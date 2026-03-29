"""
compute_fid.py: Extract atlas embeddings with a trained encoder and compute FID.

FID (Fréchet Inception Distance) between a real and a generated set of 3DGS
objects is computed by:
  1. Loading both sets of atlas .pt files.
  2. Passing them through the trained AtlasEncoder to obtain feature vectors.
  3. Fitting a multivariate Gaussian to each feature set.
  4. Computing the Fréchet distance between the two Gaussians.

Usage:
    python -m evaluation.compute_fid \\
        --config configs/config.yaml \\
        --real   /path/to/real/atlases \\
        --gen    /path/to/generated/atlases

The paths can also be set in config.yaml under `evaluation`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml
from scipy import linalg
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.atlas_encoder import build_encoder
from data.dataset import load_or_compute_stats


# ──────────────────────────────────────────────────────────────────────────────
# Minimal atlas-only dataset (no captions required for FID)
# ──────────────────────────────────────────────────────────────────────────────

class AtlasOnlyDataset(Dataset):
    """
    Loads atlas .pt files from a directory or an explicit path list (no captions).

    Args:
        atlas_dir: Directory containing <id>.pt files (ignored if `paths` is set).
        mean:      Per-channel mean tensor (C,) for normalisation, or None.
        std:       Per-channel std  tensor (C,) for normalisation, or None.
        paths:     Optional explicit list of .pt paths; when set, used instead of glob.
    """

    def __init__(
        self,
        atlas_dir: Optional[str] = None,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        paths: Optional[Sequence[Path]] = None,
    ) -> None:
        if paths is not None:
            self.paths = sorted(Path(p) for p in paths)
        elif atlas_dir is not None:
            self.paths = sorted(Path(atlas_dir).glob("*.pt"))
        else:
            raise ValueError("Provide atlas_dir or paths.")
        if len(self.paths) == 0:
            loc = atlas_dir if atlas_dir else "paths"
            raise RuntimeError(f"No .pt atlas files found for {loc!r}.")
        self.mean = mean.view(-1, 1, 1).float() if mean is not None else None
        self.std = std.view(-1, 1, 1).float() if std is not None else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        atlas = torch.load(self.paths[idx], map_location="cpu", weights_only=True)
        atlas = atlas.permute(2, 0, 1).float()  # (C, H, W)
        if self.mean is not None and self.std is not None:
            atlas = (atlas - self.mean) / (self.std + 1e-6)
        return atlas


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module,
    atlas_dir: Optional[str],
    mean: Optional[Tensor],
    std: Optional[Tensor],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    paths: Optional[Sequence[Path]] = None,
) -> np.ndarray:
    """
    Pass all atlases in `atlas_dir` (or `paths`) through `encoder` and collect embeddings.

    Returns:
        (N, D) float32 numpy array of L2-normalised embeddings.
    """
    if paths is None and atlas_dir is None:
        raise ValueError("Provide atlas_dir or paths.")
    dataset = AtlasOnlyDataset(atlas_dir=atlas_dir, mean=mean, std=std, paths=paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder.eval()
    desc = atlas_dir if atlas_dir is not None else f"{len(dataset)} atlases"
    features = []
    for batch in tqdm(loader, desc=f"Extracting [{desc}]", leave=False):
        batch = batch.to(device, non_blocking=True)
        embed = encoder(batch)          # (B, D), already L2-normalised
        features.append(embed.cpu().numpy())

    return np.concatenate(features, axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Fréchet Distance
# ──────────────────────────────────────────────────────────────────────────────

def frechet_distance(
    feats_real: np.ndarray,
    feats_gen: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute FID between two feature sets.

    FID = ||mu_r - mu_g||² + Tr(Σ_r + Σ_g - 2 * sqrt(Σ_r · Σ_g))

    Args:
        feats_real: (N_r, D) float32 embeddings for the real set.
        feats_gen:  (N_g, D) float32 embeddings for the generated set.
        eps:        Small regulariser added to covariance diagonal for stability.

    Returns:
        Scalar FID value.
    """
    mu_r = feats_real.mean(axis=0)
    mu_g = feats_gen.mean(axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_g = np.cov(feats_gen, rowvar=False)

    # Regularise to avoid singular matrices
    sigma_r += np.eye(sigma_r.shape[0]) * eps
    sigma_g += np.eye(sigma_g.shape[0]) * eps

    diff = mu_r - mu_g
    mean_sq = float(diff.dot(diff))

    # Matrix square root of sigma_r @ sigma_g
    cov_product = sigma_r @ sigma_g
    sqrt_cov, _ = linalg.sqrtm(cov_product, disp=False)

    if np.iscomplexobj(sqrt_cov):
        # Discard negligible imaginary parts caused by numerical error
        sqrt_cov = sqrt_cov.real

    trace_term = float(np.trace(sigma_r + sigma_g - 2.0 * sqrt_cov))
    fid = mean_sq + trace_term
    return fid


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 3DGS FID between two atlas sets")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override evaluation.checkpoint in config")
    parser.add_argument("--real", type=str, default=None,
                        help="Override evaluation.real_atlas_dir in config")
    parser.add_argument("--gen", type=str, default=None,
                        help="Override evaluation.gen_atlas_dir in config")
    parser.add_argument("--save-features", type=str, default=None,
                        help="Directory to save extracted .npy feature arrays")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg["data"]

    checkpoint_path = args.checkpoint or eval_cfg.get("checkpoint")
    real_dir = args.real or eval_cfg.get("real_atlas_dir")
    gen_dir = args.gen or eval_cfg.get("gen_atlas_dir")
    batch_size = eval_cfg.get("batch_size", 128)
    num_workers = eval_cfg.get("num_workers", 4)

    if not checkpoint_path:
        raise ValueError("Provide --checkpoint or set evaluation.checkpoint in config.")
    if not real_dir:
        raise ValueError("Provide --real or set evaluation.real_atlas_dir in config.")
    if not gen_dir:
        raise ValueError("Provide --gen or set evaluation.gen_atlas_dir in config.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load normalisation stats (use training stats if available) ────────────
    mean, std = load_or_compute_stats(
        atlas_dir=real_dir,
        mean_file=data_cfg.get("mean_file"),
        std_file=data_cfg.get("std_file"),
    )

    # ── Build & load encoder ──────────────────────────────────────────────────
    print(f"Loading encoder from {checkpoint_path}")
    encoder = build_encoder(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    # ── Extract features ──────────────────────────────────────────────────────
    print("Extracting features for real atlases...")
    feats_real = extract_features(encoder, real_dir, mean, std, batch_size, num_workers, device)

    print("Extracting features for generated atlases...")
    feats_gen = extract_features(encoder, gen_dir, mean, std, batch_size, num_workers, device)

    print(f"Real set:      {feats_real.shape[0]} samples")
    print(f"Generated set: {feats_gen.shape[0]} samples")
    print(f"Embedding dim: {feats_real.shape[1]}")

    # ── Optionally save features ──────────────────────────────────────────────
    save_dir = args.save_features or eval_cfg.get("feature_save_dir")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "real_features.npy"), feats_real)
        np.save(os.path.join(save_dir, "gen_features.npy"), feats_gen)
        print(f"Features saved to {save_dir}")

    # ── Compute FID ───────────────────────────────────────────────────────────
    print("Computing FID...")
    fid = frechet_distance(feats_real, feats_gen)
    print(f"\n{'='*40}")
    print(f"  3DGS-FID = {fid:.4f}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
