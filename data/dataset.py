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


# ──────────────────────────────────────────────────────────────────────────────
# PLY / sphere helpers (used by GaussianVerseDataset for live loading)
# ──────────────────────────────────────────────────────────────────────────────

def _load_ply_raw(path: str) -> np.ndarray:
    """Load a GaussianVerse PLY and return an (N, 14) float32 array.

    Channels: xyz(3) | f_dc color(3) | opacity(1) | scale(3) | rotation(4)
    """
    from plyfile import PlyData

    plydata = PlyData.read(path)
    vert = plydata.elements[0]

    def sorted_attrs(prefix: str) -> List[str]:
        names = [p.name for p in vert.properties if p.name.startswith(prefix)]
        return sorted(names, key=lambda n: int(n.split("_")[-1]))

    xyz = np.stack(
        [np.asarray(vert["x"]), np.asarray(vert["y"]), np.asarray(vert["z"])],
        axis=1,
    )
    color = np.column_stack([np.asarray(vert[n]) for n in sorted_attrs("f_dc_")])
    opacity = np.asarray(vert["opacity"])[..., np.newaxis]
    scales = np.column_stack([np.asarray(vert[n]) for n in sorted_attrs("scale_")])
    rots = np.column_stack([np.asarray(vert[n]) for n in sorted_attrs("rot")])

    return np.concatenate([xyz, color, opacity, scales, rots], axis=1).astype(np.float32)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return (n, 3) Fibonacci-distributed points on the unit sphere."""
    phi = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n)
    y = 1.0 - (indices / float(n - 1)) * 2.0
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * indices
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)


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
# Live-loading dataset (no precomputed atlases required)
# ──────────────────────────────────────────────────────────────────────────────

class GaussianVerseDataset(Dataset):
    """
    Loads Gaussian Atlas tensors on-the-fly from raw GaussianVerse scene dirs,
    avoiding the need to precompute and store atlas .pt files.

    Each scene dir is expected to contain:
      point_cloud.ply  — fixed 128×128 Gaussians in GaussianVerse format
      gs2sphere.npy    — precomputed mapping: gs2sphere[i] = sphere point index
                         for Gaussian i (i.e. the scene-to-sphere assignment).

    The sphere-to-plane mapping (sphere2plane.npy) is shared across all scenes
    and loaded once at init from sphere2plane_path.

    Atlas channel layout: xyz(3) | color(3) | opacity(1) | scale(3) | rot(4) |
                          offset(3) = 17 channels total.
    """

    def __init__(
        self,
        source_root: str,
        sphere2plane_path: str,
        captions_json: str,
        atlas_resolution: int = 128,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        clip_embeddings: Optional[Dict[str, Tensor]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.source_root = Path(source_root)
        self.grid_side = atlas_resolution

        # Sphere-to-plane mapping — shared across all scenes
        self.sphere_to_plane: np.ndarray = np.load(sphere2plane_path)

        # Canonical sphere points, lexsorted to match gs2atlas.py ordering
        n_pts = atlas_resolution * atlas_resolution
        sphere_pts = _fibonacci_sphere(n_pts)
        order = np.lexsort((sphere_pts[:, 2], sphere_pts[:, 1], sphere_pts[:, 0]))
        self.sphere_pts: np.ndarray = sphere_pts[order].astype(np.float32)  # (N, 3)

        with open(captions_json) as f:
            captions: Dict[str, str] = json.load(f)

        # obj_id format in captions.json and on disk is "cat-scene" e.g. "1877-9378643"
        if ids is not None:
            candidates = ids
        else:
            candidates = list(captions.keys())

        self.samples: List[Tuple[str, str, Union[str, Tensor]]] = []
        for obj_id in candidates:
            # Keys in captions.json use "/" separator: "1877/9378643"
            parts = obj_id.split("/", 1)
            if len(parts) != 2:
                continue
            cat_id, scene_id = parts
            scene_dir = self.source_root / cat_id / scene_id
            if not (scene_dir / "point_cloud.ply").exists():
                continue
            if not (scene_dir / "gs2sphere.npy").exists():
                continue

            if clip_embeddings is not None:
                if obj_id in clip_embeddings:
                    self.samples.append((str(scene_dir), obj_id, clip_embeddings[obj_id]))
            else:
                if obj_id in captions:
                    self.samples.append((str(scene_dir), obj_id, captions[obj_id]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid scenes found under '{source_root}'. "
                "Check that captions.json keys match 'cat-scene' directory structure "
                "and that each scene dir contains point_cloud.ply and gs2sphere.npy."
            )

        self.mean: Optional[Tensor] = mean.view(-1, 1, 1).float() if mean is not None else None
        self.std: Optional[Tensor] = std.view(-1, 1, 1).float() if std is not None else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        scene_dir, obj_id, label = self.samples[idx]

        # Load raw Gaussians: (N, 14) — xyz|color|opacity|scale|rot
        gs = _load_ply_raw(os.path.join(scene_dir, "point_cloud.ply"))

        # gs2sphere[i] = sphere point index assigned to Gaussian i (scene→sphere).
        # argsort gives the inverse: sphere_to_scene[j] = Gaussian index for sphere point j.
        gs2sphere = np.load(os.path.join(scene_dir, "gs2sphere.npy"))
        sphere_to_scene = np.argsort(gs2sphere)
        gs = gs[sphere_to_scene]  # (N, 14), now gs[j] corresponds to sphere_pts[j]

        # Append xyz offset from matched sphere point: (N, 17)
        offset = gs[:, :3] - self.sphere_pts
        gs = np.concatenate([gs, offset], axis=-1)

        # Reorder by sphere-to-plane mapping and reshape to 2D grid
        gs = gs[self.sphere_to_plane]
        gs = gs.reshape(self.grid_side, self.grid_side, -1)  # (H, W, 17)

        atlas: Tensor = torch.from_numpy(gs).float().permute(2, 0, 1)  # (C, H, W)

        if self.mean is not None and self.std is not None:
            atlas = (atlas - self.mean) / (self.std + 1e-6)

        if self.mean is None or self.std is None:
            pass  # unnormalised — add mean/std files to config when ready

        if isinstance(label, Tensor):
            return {"atlas": atlas, "clip_embed": label.float()}
        else:
            return {"atlas": atlas, "caption": label}

    @property
    def uses_precomputed_embeddings(self) -> bool:
        return isinstance(self.samples[0][2], Tensor) if self.samples else False

    @property
    def num_channels(self) -> int:
        return self.grid_side  # placeholder; actual C=17 from atlas shape


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
    atlas_dir: Optional[str],
    mean_file: Optional[str],
    std_file: Optional[str],
    num_samples: int = 5000,
    num_workers: int = 4,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Return (mean, std) tensors, or (None, None) if unavailable.

    If mean_file / std_file are provided and exist, loads them.
    If atlas_dir is provided but stat files are missing, computes from .pt files
    and saves to those paths (if provided).
    If atlas_dir is None (live_load mode) and stat files are missing, returns
    (None, None) — normalization is skipped until stat files are provided.
    """
    if mean_file and std_file and os.path.exists(mean_file) and os.path.exists(std_file):
        mean = torch.from_numpy(np.load(mean_file))
        std = torch.from_numpy(np.load(std_file))
        return mean, std

    if atlas_dir is None:
        print(
            "[INFO] live_load mode: no mean/std files found — running without normalisation. "
            "Provide data.mean_file and data.std_file in the config to enable it."
        )
        return None, None

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
    cfg['data'], cfg['training'], and (for live mode) cfg['gs2atlas'].

    Set data.live_load: true to load directly from raw GaussianVerse scene
    directories using gs2sphere.npy (no precomputed atlas .pt files needed).
    The source_root and sphere2plane_path are read from the gs2atlas config
    section in this mode.

    Set data.live_load: false (or omit) to load from precomputed .pt atlas
    files under data.atlas_dir.

    If data.clip_embeddings_file is set and the file exists, precomputed CLIP
    embeddings are loaded so the CLIP encoder is not needed at training time.
    Run `python -m data.precompute_clip` first to generate the file.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    live_load = data_cfg.get("live_load", False)

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

    if live_load:
        gs2atlas_cfg = cfg["gs2atlas"]
        mean, std = load_or_compute_stats(
            atlas_dir=None,
            mean_file=data_cfg.get("mean_file"),
            std_file=data_cfg.get("std_file"),
        )
        full_dataset = GaussianVerseDataset(
            source_root=gs2atlas_cfg["source_root"],
            sphere2plane_path=gs2atlas_cfg["sphere2plane_path"],
            captions_json=data_cfg["captions_json"],
            atlas_resolution=data_cfg.get("atlas_resolution", 128),
            mean=mean,
            std=std,
            clip_embeddings=clip_embeddings,
        )
    else:
        mean, std = load_or_compute_stats(
            atlas_dir=data_cfg["atlas_dir"],
            mean_file=data_cfg.get("mean_file"),
            std_file=data_cfg.get("std_file"),
        )
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
