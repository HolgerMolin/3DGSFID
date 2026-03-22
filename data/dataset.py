"""
AtlasCaptionDataset: PyTorch dataset for Gaussian Atlas .pt files paired with text captions.

Expected captions.json format (same as GaussianVerse / Cap3D):
  {
    "1876-9374307": "A blue office chair ...",
    ...
  }

Each atlas .pt file must be named "<cat_id>-<obj_id>.pt" and contain a float32
tensor of shape (H, W, C) where H = W = atlas_resolution and C = atlas_channels.

When `clip_embeddings` is provided (a dict loaded from the file produced by
data/precompute_clip.py), __getitem__ returns a pre-encoded `clip_embed` tensor
instead of the raw caption string.  This eliminates the cost of running the
frozen CLIP encoder during training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class AtlasCaptionDataset(Dataset):
    """
    Loads Gaussian Atlas tensors and their paired text captions (or precomputed
    CLIP embeddings when available).

    Args:
        atlas_dir:        Directory containing <id>.pt files.
        captions_json:    Path to captions.json mapping object-id → caption string.
        mean:             Per-channel mean tensor of shape (C,) for normalisation.
                          Pass None to skip normalisation.
        std:              Per-channel std tensor of shape (C,) for normalisation.
                          Pass None to skip normalisation.
        clip_embeddings:  Optional dict {obj_id → Tensor(D)} produced by
                          data/precompute_clip.py.  When provided the dataset
                          yields `clip_embed` tensors instead of caption strings,
                          allowing the CLIP encoder to be omitted at training time.
        ids:              Explicit list of object IDs to include (subset). If None,
                          all IDs present in both atlas_dir and captions_json are used.
    """

    def __init__(
        self,
        atlas_dir: str,
        captions_json: str,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        clip_embeddings: Optional[Dict[str, Tensor]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.atlas_dir = Path(atlas_dir)
        self.clip_embeddings = clip_embeddings  # None → fall back to caption strings

        with open(captions_json) as f:
            captions: Dict[str, str] = json.load(f)

        # Build the list of valid (id, atlas_path, caption/embed) triples.
        # When clip_embeddings is provided we require the id to be present there
        # too, so the dataset is self-consistent.
        if ids is not None:
            candidates = ids
        else:
            candidates = [p.stem for p in self.atlas_dir.glob("*.pt")]

        self.samples: List[Tuple[str, Union[str, Tensor]]] = []
        for obj_id in candidates:
            atlas_path = self.atlas_dir / f"{obj_id}.pt"
            if not atlas_path.exists():
                continue
            if clip_embeddings is not None:
                if obj_id in clip_embeddings:
                    self.samples.append((str(atlas_path), clip_embeddings[obj_id]))
            else:
                if obj_id in captions:
                    self.samples.append((str(atlas_path), captions[obj_id]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid (atlas, caption) pairs found in '{atlas_dir}'. "
                "Check that atlas filenames match keys in captions.json "
                "(and clip_embeddings_file if provided)."
            )

        # Normalisation statistics — shape (C, 1, 1) for broadcasting over (C, H, W)
        self.mean: Optional[Tensor] = mean.view(-1, 1, 1).float() if mean is not None else None
        self.std: Optional[Tensor] = std.view(-1, 1, 1).float() if std is not None else None

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        atlas_path, label = self.samples[idx]

        # Shape on disk: (H, W, C) → transpose to (C, H, W) for ConvNeXt
        atlas: Tensor = torch.load(atlas_path, map_location="cpu", weights_only=True)
        atlas = atlas.permute(2, 0, 1).float()  # (C, H, W)

        if self.mean is not None and self.std is not None:
            atlas = (atlas - self.mean) / (self.std + 1e-6)

        if self.clip_embeddings is not None:
            # label is a precomputed Tensor(D)
            return {"atlas": atlas, "clip_embed": label.float()}
        else:
            # label is a raw caption string
            return {"atlas": atlas, "caption": label}

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def uses_precomputed_embeddings(self) -> bool:
        return self.clip_embeddings is not None

    @property
    def num_channels(self) -> int:
        """Return C by peeking at the first sample (without normalising)."""
        atlas: Tensor = torch.load(self.samples[0][0], map_location="cpu", weights_only=True)
        return atlas.shape[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_mean_std(
    atlas_dir: str,
    num_samples: int = 5000,
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-channel mean and std from a random subset of atlas files.

    Returns numpy arrays of shape (C,).
    """
    paths = list(Path(atlas_dir).glob("*.pt"))
    rng = np.random.default_rng(42)
    subset = rng.choice(paths, size=min(num_samples, len(paths)), replace=False)

    running_sum = None
    running_sq = None
    count = 0

    for path in subset:
        atlas = torch.load(path, map_location="cpu", weights_only=True).float()
        # atlas: (H, W, C)
        flat = atlas.reshape(-1, atlas.shape[-1]).numpy()  # (H*W, C)
        if running_sum is None:
            running_sum = np.zeros(flat.shape[1], dtype=np.float64)
            running_sq = np.zeros(flat.shape[1], dtype=np.float64)
        running_sum += flat.sum(axis=0)
        running_sq += (flat ** 2).sum(axis=0)
        count += flat.shape[0]

    mean = running_sum / count
    std = np.sqrt(running_sq / count - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


def load_or_compute_stats(
    atlas_dir: str,
    mean_file: Optional[str],
    std_file: Optional[str],
    num_samples: int = 5000,
    num_workers: int = 4,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Return (mean, std) tensors.

    If mean_file / std_file are provided and exist, loads them; otherwise
    computes from atlas_dir and saves to those paths (if provided).
    """
    if mean_file and std_file and os.path.exists(mean_file) and os.path.exists(std_file):
        mean = torch.from_numpy(np.load(mean_file))
        std = torch.from_numpy(np.load(std_file))
        return mean, std

    mean_np, std_np = compute_mean_std(atlas_dir, num_samples, num_workers)

    if mean_file:
        os.makedirs(os.path.dirname(mean_file) or ".", exist_ok=True)
        np.save(mean_file, mean_np)
    if std_file:
        os.makedirs(os.path.dirname(std_file) or ".", exist_ok=True)
        np.save(std_file, std_np)

    return torch.from_numpy(mean_np), torch.from_numpy(std_np)


# ──────────────────────────────────────────────────────────────────────────────
# Dataloader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from config dict.

    The config dict is the full loaded YAML; relevant keys live under
    cfg['data'] and cfg['training'].

    If data.clip_embeddings_file is set and the file exists, precomputed CLIP
    embeddings are loaded from it so the CLIP encoder is not needed at training
    time.  Run `python -m data.precompute_clip` first to generate the file.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    mean, std = load_or_compute_stats(
        atlas_dir=data_cfg["atlas_dir"],
        mean_file=data_cfg.get("mean_file"),
        std_file=data_cfg.get("std_file"),
    )

    # Optionally load precomputed CLIP embeddings
    clip_embeddings: Optional[Dict[str, Tensor]] = None
    clip_embed_file = data_cfg.get("clip_embeddings_file")
    if clip_embed_file and os.path.exists(clip_embed_file):
        print(f"Loading precomputed CLIP embeddings from {clip_embed_file}")
        clip_embeddings = torch.load(clip_embed_file, map_location="cpu", weights_only=True)
        print(f"  Loaded {len(clip_embeddings):,} embeddings")
    elif clip_embed_file:
        print(
            f"[WARNING] clip_embeddings_file '{clip_embed_file}' not found. "
            "Falling back to online CLIP encoding. "
            "Run `python -m data.precompute_clip` to generate it."
        )

    # Full dataset with all IDs
    full_dataset = AtlasCaptionDataset(
        atlas_dir=data_cfg["atlas_dir"],
        captions_json=data_cfg["captions_json"],
        mean=mean,
        std=std,
        clip_embeddings=clip_embeddings,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * data_cfg.get("train_split", 0.99))
    n_val = n_total - n_train

    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
