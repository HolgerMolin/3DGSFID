"""
gs2atlas: Convert 3D Gaussian splat PLY files to Gaussian Atlas tensors.

Adapted from the 3DGen-Playground reference implementation:
  https://github.com/tiangexiang/3DGen-Playground

Pipeline per scene:
  Load PLY → filter by opacity → pad/prune to resolution² →
  lexsort by xyz → OT match to canonical sphere →
  reorder by sphere-to-plane mapping → save (H, W, C) tensor as .pt
"""

import os
import argparse
import concurrent.futures
import time
from pathlib import Path

import numpy as np
import torch
import ot
from lapjv import lapjv
from plyfile import PlyData
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# PLY loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ply(path: str, max_sh_degree: int = 0) -> torch.Tensor:
    """
    Load a 3DGS PLY file and return a (N, C) float32 tensor.

    Channels: xyz(3) | color_DC(3) | opacity(1) | scale(3) | rotation(4)
    """
    plydata = PlyData.read(path)
    vert = plydata.elements[0]

    xyz = np.stack(
        [np.asarray(vert["x"]), np.asarray(vert["y"]), np.asarray(vert["z"])],
        axis=1,
    )
    opacity = np.asarray(vert["opacity"])[..., np.newaxis]

    def sorted_attrs(prefix: str):
        names = [p.name for p in vert.properties if p.name.startswith(prefix)]
        return sorted(names, key=lambda n: int(n.split("_")[-1]))

    color = np.column_stack([np.asarray(vert[n]) for n in sorted_attrs("color_")])
    scales = np.column_stack([np.asarray(vert[n]) for n in sorted_attrs("scale_")])
    rots = np.column_stack([np.asarray(vert[n]) for n in sorted_attrs("rot")])

    data = np.concatenate([xyz, color, opacity, scales, rots], axis=1)
    return torch.tensor(data, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Sphere utilities
# ──────────────────────────────────────────────────────────────────────────────

def fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    """Return (n, 3) Fibonacci-distributed points on a unit sphere."""
    phi = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n)
    y = 1.0 - (indices / float(n - 1)) * 2.0
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * indices
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    points = np.stack([x, y, z], axis=1) * radius
    return points


# ──────────────────────────────────────────────────────────────────────────────
# Optimal transport matching
# ──────────────────────────────────────────────────────────────────────────────

def ot_match(scene_pts: np.ndarray, sphere_pts: np.ndarray):
    """
    One-shot linear assignment between scene_pts and sphere_pts.

    Uses float32 for the cost matrix (half the memory of float64) and int32
    after scaling, which is sufficient precision for lapjv.

    Returns (corrs_sphere_to_scene, corrs_scene_to_sphere).
    """
    cost_matrix = ot.dist(scene_pts, sphere_pts, metric="sqeuclidean").astype(np.float32)
    scaled = np.rint(cost_matrix * 1000).astype(np.int32)
    x, y, _ = lapjv(scaled)
    return y, x  # corrs_sphere_to_scene, corrs_scene_to_sphere


# ──────────────────────────────────────────────────────────────────────────────
# Single-scene conversion
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(
    scene_dir: str,
    save_root: str,
    sphere_pts: np.ndarray,
    sphere_to_plane: np.ndarray,
    ply_iteration: int = 30000,
    max_sh_degree: int = 0,
    resolution: int = 128 * 128,
) -> None:
    """Convert one GaussianVerse scene directory to an atlas .pt file."""
    parts = Path(scene_dir).parts
    scene_name = f"{parts[-2]}-{parts[-1]}"
    save_path = os.path.join(save_root, scene_name + ".pt")

    if os.path.exists(save_path):
        print(f"[SKIP] {scene_name}.pt already exists")
        return

    ply_path = os.path.join(
        scene_dir, f"point_cloud/iteration_{ply_iteration}/point_cloud.ply"
    )
    if not os.path.exists(ply_path):
        print(f"[MISSING] {ply_path}")
        return

    t0 = time.time()
    gs = load_ply(ply_path, max_sh_degree).numpy()  # (N, C_raw)

    # Filter by opacity > 0
    visible = gs[:, 6] > 0.0
    gs = gs[visible]
    n_vis = gs.shape[0]

    # Pad (duplicate small Gaussians with zeroed opacity) or prune (remove smallest)
    if n_vis < resolution:
        deficit = resolution - n_vis
        mean_scales = gs[:, 7:10].mean(axis=1)
        pad_idx = np.argsort(mean_scales)[:deficit]
        pad = gs[pad_idx].copy()
        pad[:, 6] = 0.0  # zero out opacity of padding
        gs = np.concatenate([gs, pad], axis=0)
    elif n_vis > resolution:
        surplus = n_vis - resolution
        mean_scales = gs[:, 7:10].mean(axis=1)
        keep_idx = np.argsort(mean_scales)[surplus:]
        gs = gs[keep_idx]

    # Lexsort scene points by (x, y, z) to get a deterministic ordering
    pts = gs[:, :3]
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    pts = pts[order]
    gs = gs[order]

    # OT: match scene points to canonical sphere points
    print(f"[OT] {scene_name}")
    sphere_to_scene, _ = ot_match(pts, sphere_pts)
    gs = gs[sphere_to_scene]

    # Append per-Gaussian offset from its matched sphere point
    offset = gs[:, :3] - sphere_pts
    gs = np.concatenate([gs, offset], axis=-1)  # (resolution, C_raw + 3)

    # Reorder by sphere-to-plane mapping → 2-D grid
    gs = gs[sphere_to_plane]
    side = int(np.sqrt(resolution))
    gs = gs.reshape(side, side, -1)

    torch.save(torch.from_numpy(gs).float(), save_path)
    print(f"[DONE] {scene_name}.pt  ({time.time() - t0:.1f}s)")


def _worker_wrapper(args):
    scene_dir, save_root, sphere_pts, sphere_to_plane, cfg = args
    process_scene(
        scene_dir,
        save_root,
        sphere_pts,
        sphere_to_plane,
        ply_iteration=cfg["ply_iteration"],
        max_sh_degree=cfg["max_sh_degree"],
        resolution=cfg["atlas_resolution"] ** 2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert 3DGS PLYs to Gaussian Atlas .pt files")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--txt_file", type=str, required=True,
                        help="Text file listing relative scene paths (one per line)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_full = yaml.safe_load(f)
    cfg = cfg_full["gs2atlas"]
    cfg["atlas_resolution"] = cfg_full["data"]["atlas_resolution"]

    os.makedirs(cfg["save_root"], exist_ok=True)

    # Canonical sphere points (lexsorted to match scene-point sorting)
    n_pts = cfg["atlas_resolution"] ** 2
    sphere_pts = fibonacci_sphere(n_pts)
    order = np.lexsort((sphere_pts[:, 2], sphere_pts[:, 1], sphere_pts[:, 0]))
    sphere_pts = sphere_pts[order]

    sphere_to_plane = np.load(cfg["sphere2plane_path"])

    with open(args.txt_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    end = len(lines) if args.end_idx == -1 else args.end_idx
    lines = lines[args.start_idx : end]
    print(f"Processing {len(lines)} scenes with {cfg['num_workers']} workers")

    task_args = [
        (os.path.join(cfg["source_root"], line), cfg["save_root"], sphere_pts, sphere_to_plane, cfg)
        for line in lines
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["num_workers"]) as executor:
        list(executor.map(_worker_wrapper, task_args))


if __name__ == "__main__":
    main()
