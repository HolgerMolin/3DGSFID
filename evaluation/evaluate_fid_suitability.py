"""
evaluate_fid_suitability.py: Diagnostics for whether a trained encoder behaves well as an FID backbone.

1. Same-dataset subset FID: split real atlases into two disjoint halves (same size; one file
   dropped if the pool size is odd). Low FID means the encoder treats both halves as similar —
   desirable for a stable metric. Very small subsets yield noisy covariances.

2. Degradation monotonicity: FID(reference clean features vs features of the same atlases with
   additive Gaussian noise in *normalised* space) at increasing noise std. Scores should rise
   monotonically (within --monotone-tol).

Data source (matches training):
  - If data.live_load is true in the config (default for this repo), atlases are built on-the-fly
    from GaussianVerse scene dirs (gs2atlas.source_root, sphere2plane, captions) — same as training.
  - If data.live_load is false, or you pass --atlas-dir, uses precomputed *.pt under that directory.

Evaluate a checkpoint (live mode, typical):
    cd /home/hmolin/3DGSFID && python -m evaluation.evaluate_fid_suitability \\
        --config configs/config.yaml \\
        --checkpoint ./checkpoints/best.pth

Precomputed atlases only:
    python -m evaluation.evaluate_fid_suitability --config configs/config.yaml \\
        --checkpoint ./checkpoints/best.pth --atlas-dir /path/to/atlases
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import GaussianVerseDataset, load_or_compute_stats
from evaluation.compute_fid import extract_features, frechet_distance
from models.atlas_encoder import build_encoder


def _load_clip_embeddings_if_any(data_cfg: dict) -> Optional[Dict[str, Tensor]]:
    path = data_cfg.get("clip_embeddings_file")
    if path and os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


def build_gaussian_verse_dataset(cfg: dict, mean: Optional[Tensor], std: Optional[Tensor]) -> GaussianVerseDataset:
    """Same scene list and preprocessing as training when data.live_load is true."""
    data_cfg = cfg["data"]
    gs2atlas_cfg = cfg["gs2atlas"]
    clip_embeddings = _load_clip_embeddings_if_any(data_cfg)
    return GaussianVerseDataset(
        source_root=gs2atlas_cfg["source_root"],
        sphere2plane_path=gs2atlas_cfg["sphere2plane_path"],
        captions_json=data_cfg["captions_json"],
        atlas_resolution=data_cfg.get("atlas_resolution", 128),
        mean=mean,
        std=std,
        clip_embeddings=clip_embeddings,
    )


class IndexedLiveAtlasDataset(Dataset):
    """On-the-fly atlases from GaussianVerseDataset, fixed index order."""

    def __init__(self, base: GaussianVerseDataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tensor:
        return self.base[self.indices[i]]["atlas"]


class NoisyIndexedLiveAtlasDataset(Dataset):
    def __init__(
        self,
        base: GaussianVerseDataset,
        indices: Sequence[int],
        noise_std: float,
    ) -> None:
        self.base = base
        self.indices = list(indices)
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tensor:
        atlas = self.base[self.indices[i]]["atlas"]
        if self.noise_std > 0:
            atlas = atlas + self.noise_std * torch.randn_like(atlas)
        return atlas


@torch.no_grad()
def extract_features_tensor_dataset(
    encoder: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    encoder.eval()
    features = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device, non_blocking=True)
        embed = encoder(batch)
        features.append(embed.cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32)


class NoisyAtlasDataset(Dataset):
    """Same loading/normalisation as FID atlases, then optional additive noise after normalisation."""

    def __init__(
        self,
        paths: Sequence[Path],
        mean: Optional[Tensor],
        std: Optional[Tensor],
        noise_std: float,
    ) -> None:
        self.paths = sorted(Path(p) for p in paths)
        self.mean = mean.view(-1, 1, 1).float() if mean is not None else None
        self.std = std.view(-1, 1, 1).float() if std is not None else None
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        atlas = torch.load(self.paths[idx], map_location="cpu", weights_only=True)
        atlas = atlas.permute(2, 0, 1).float()
        if self.mean is not None and self.std is not None:
            atlas = (atlas - self.mean) / (self.std + 1e-6)
        if self.noise_std > 0:
            atlas = atlas + self.noise_std * torch.randn_like(atlas)
        return atlas


@torch.no_grad()
def extract_features_noisy_pt(
    encoder: torch.nn.Module,
    paths: Sequence[Path],
    mean: Optional[Tensor],
    std: Optional[Tensor],
    noise_std: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    dataset = NoisyAtlasDataset(paths, mean=mean, std=std, noise_std=noise_std)
    return extract_features_tensor_dataset(
        encoder,
        dataset,
        batch_size,
        num_workers,
        device,
        desc=f"Noisy σ={noise_std:g} [{len(dataset)} .pt]",
    )


def split_subsets_pool(
    pool: List,
    subset_fraction: float,
) -> Tuple[List, List, List]:
    """
    Disjoint A, B of equal size. k = min(floor(N * fraction), N // 2).
    pool is already shuffled; returns (a, b, unused) slices of that list.
    """
    n = len(pool)
    if n < 2:
        raise ValueError(f"Need at least 2 items in pool; got {n}.")
    k = int(n * subset_fraction)
    k = min(max(k, 1), n // 2)
    a = pool[:k]
    b = pool[k : 2 * k]
    unused = pool[2 * k :]
    return a, b, unused


def build_shuffled_index_pool(n_total: int, seed: int, max_samples: Optional[int]) -> List[int]:
    rng = random.Random(seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    if max_samples is not None:
        idx = idx[:max_samples]
    return idx


def monotonicity_violations(values: List[float], tol: float) -> int:
    return sum(1 for i in range(len(values) - 1) if values[i] > values[i + 1] + tol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate encoder suitability for FID (subset FID + noise monotonicity)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override evaluation.checkpoint in config")
    parser.add_argument(
        "--atlas-dir",
        type=str,
        default=None,
        help="Precomputed *.pt atlas directory. If omitted and data.live_load is true, "
        "builds atlases on-the-fly from gs2atlas (same as training).",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for shuffling and subset split")
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=0.5,
        help="Target fraction for subset A; B matches size, capped at half the pool",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="After shuffling, use at most this many scenes (.pt or live) for both analyses",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0,0.02,0.05,0.1,0.2",
        help="Comma-separated noise stds in normalised atlas space",
    )
    parser.add_argument(
        "--monotone-tol",
        type=float,
        default=1e-3,
        help="Allow FID to decrease by at most this much between consecutive noise levels",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write metrics as CSV",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg["data"]

    checkpoint_path = args.checkpoint or eval_cfg.get("checkpoint")
    batch_size = eval_cfg.get("batch_size", 128)
    num_workers = eval_cfg.get("num_workers", 4)

    if args.atlas_dir is not None:
        use_live = False
        atlas_dir_resolved = args.atlas_dir
    elif data_cfg.get("live_load", False):
        use_live = True
        atlas_dir_resolved = None
    else:
        use_live = False
        atlas_dir_resolved = eval_cfg.get("real_atlas_dir") or data_cfg.get("atlas_dir")
        if not atlas_dir_resolved:
            raise ValueError(
                "Precomputed atlas mode: set --atlas-dir, or evaluation.real_atlas_dir, "
                "or data.atlas_dir — or set data.live_load: true for on-the-fly atlases."
            )

    if not checkpoint_path:
        raise ValueError("Provide --checkpoint or set evaluation.checkpoint in config.")

    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]
    if not noise_levels:
        raise ValueError("--noise-levels must contain at least one value.")
    if not (0 < args.subset_fraction <= 1):
        raise ValueError("--subset-fraction must be in (0, 1].")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data: {'on-the-fly GaussianVerse (same as training)' if use_live else 'precomputed .pt'}")

    mean, std = load_or_compute_stats(
        atlas_dir=atlas_dir_resolved if not use_live else None,
        mean_file=data_cfg.get("mean_file"),
        std_file=data_cfg.get("std_file"),
    )

    print(f"Loading encoder from {checkpoint_path}")
    encoder = build_encoder(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    gv: Optional[GaussianVerseDataset] = None
    if use_live:
        print("Building GaussianVerse dataset (live atlases)...")
        gv = build_gaussian_verse_dataset(cfg, mean, std)
        n_total = len(gv)
        pool = build_shuffled_index_pool(n_total, args.seed, args.max_samples)
    else:
        paths = sorted(Path(atlas_dir_resolved).glob("*.pt"))
        rng = random.Random(args.seed)
        rng.shuffle(paths)
        if args.max_samples is not None:
            paths = paths[: args.max_samples]
        pool = paths

    if len(pool) < 2:
        raise RuntimeError(
            "Need at least 2 samples in the pool. "
            + (f"GaussianVerseDataset has {len(gv) if gv else 0} scenes." if use_live else f"No .pt in {atlas_dir_resolved!r}.")
        )

    pool_a, pool_b, unused = split_subsets_pool(pool, args.subset_fraction)
    print(f"\nPool: {len(pool)} samples (after shuffle" +
          (f", max_samples={args.max_samples}" if args.max_samples else "") + ")")
    print(f"Subset A: {len(pool_a)}, Subset B: {len(pool_b)}, unused: {len(unused)}")
    if unused:
        print(
            "(Unused items keep A and B the same size; odd total drops one, "
            "or shrink when 2*k < N.)"
        )

    print("\n--- Same-dataset subset FID ---")
    if use_live:
        assert gv is not None
        feats_a = extract_features_tensor_dataset(
            encoder,
            IndexedLiveAtlasDataset(gv, pool_a),
            batch_size,
            num_workers,
            device,
            desc=f"Subset A [{len(pool_a)} live]",
        )
        feats_b = extract_features_tensor_dataset(
            encoder,
            IndexedLiveAtlasDataset(gv, pool_b),
            batch_size,
            num_workers,
            device,
            desc=f"Subset B [{len(pool_b)} live]",
        )
    else:
        feats_a = extract_features(
            encoder, None, mean, std, batch_size, num_workers, device, paths=pool_a
        )
        feats_b = extract_features(
            encoder, None, mean, std, batch_size, num_workers, device, paths=pool_b
        )
    fid_sub = frechet_distance(feats_a, feats_b)
    print(f"FID(subset_A, subset_B) = {fid_sub:.6f}")
    print(
        "Lower is better: encoder treats two random halves of the same corpus as closer "
        "(more stable FID backbone)."
    )

    print("\n--- Monotonicity vs additive noise (normalised space) ---")
    noise_sorted = sorted(noise_levels)
    fids_by_sigma: List[Tuple[float, float]] = []
    print("Extracting clean reference features...")
    if use_live:
        assert gv is not None
        feats_ref = extract_features_tensor_dataset(
            encoder,
            IndexedLiveAtlasDataset(gv, pool),
            batch_size,
            num_workers,
            device,
            desc=f"Clean ref [{len(pool)} live]",
        )
        for sigma in noise_sorted:
            feats_n = extract_features_tensor_dataset(
                encoder,
                NoisyIndexedLiveAtlasDataset(gv, pool, sigma),
                batch_size,
                num_workers,
                device,
                desc=f"Noisy σ={sigma:g} [{len(pool)} live]",
            )
            fid_n = frechet_distance(feats_ref, feats_n)
            fids_by_sigma.append((sigma, fid_n))
    else:
        feats_ref = extract_features(
            encoder, None, mean, std, batch_size, num_workers, device, paths=pool
        )
        for sigma in noise_sorted:
            feats_n = extract_features_noisy_pt(
                encoder, pool, mean, std, sigma, batch_size, num_workers, device
            )
            fid_n = frechet_distance(feats_ref, feats_n)
            fids_by_sigma.append((sigma, fid_n))

    print(f"\n{'sigma':>10}  {'FID(ref, noisy)':>18}")
    print("-" * 32)
    for sigma, fid_n in fids_by_sigma:
        print(f"{sigma:10.6g}  {fid_n:18.6f}")

    fid_values = [f for _, f in fids_by_sigma]
    viol = monotonicity_violations(fid_values, args.monotone_tol)
    mono_ok = viol == 0
    print(f"\nMonotonic (non-decreasing FID vs σ, tol={args.monotone_tol:g}): {mono_ok}")
    print(f"Violations (consecutive pairs): {viol}")
    if noise_sorted[0] == 0:
        print(f"Sanity: σ=0 FID should be ~0; got {fids_by_sigma[0][1]:.6f}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["data_mode", "live" if use_live else "precomputed_pt"])
            w.writerow(["subset_fid", f"{fid_sub:.8f}"])
            w.writerow(["subset_a_count", len(pool_a)])
            w.writerow(["subset_b_count", len(pool_b)])
            w.writerow(["pool_count", len(pool)])
            for sigma, fid_n in fids_by_sigma:
                w.writerow([f"noise_fid_sigma_{sigma:g}", f"{fid_n:.8f}"])
            w.writerow(["monotone_violations", viol])
            w.writerow(["monotone_ok", int(mono_ok)])
        print(f"\nWrote CSV to {out}")


if __name__ == "__main__":
    main()
