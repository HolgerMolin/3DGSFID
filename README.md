# 3DGS-FID

A training pipeline for computing FID scores natively over 3D Gaussian Splat distributions.

A ConvNeXt network is trained to embed **Gaussian Atlases** — 2D grid representations of 3DGS objects — into an embedding space aligned with frozen **CLIP** text embeddings via symmetric InfoNCE contrastive loss. The trained encoder is then used as the feature extractor for **Fréchet Inception Distance (FID)** computation between real and generated 3DGS sets.

Gaussian atlases are computed following the method from [Gaussian Atlas (ICCV 2025)](https://arxiv.org/abs/2503.15877) and the [3DGen-Playground](https://github.com/tiangexiang/3DGen-Playground) reference implementation.

---

## Architecture

```
PLY files (GaussianVerse)
        │
        ▼  gs2atlas preprocessing
Atlas .pt (128 × 128 × 17)
        │
        ▼  ConvNeXt encoder + projection head
Atlas embedding (512-dim, L2-norm'd)
        │
        │◄──────── InfoNCE contrastive loss ────────►│
        │                                             │
Text captions ──► Frozen CLIP text encoder ──► CLIP embedding (512-dim)

At evaluation:
Atlas embeddings ──► FID computation between real and generated sets
```

The atlas tensor has **C = 17 channels** per Gaussian:
`xyz (3) | color DC (3) | opacity (1) | scale (3) | rotation (4) | offset-from-sphere (3)`

---

## Installation

```bash
# 1. Clone
git clone https://github.com/your-org/3DGSFID.git
cd 3DGSFID

# 2. Create environment
conda create -n 3dgsfid python=3.10
conda activate 3dgsfid

# 3. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt
```

---

## Project Layout

```
3DGSFID/
├── configs/
│   └── config.yaml            # all hyperparameters and paths
├── data/
│   ├── gs2atlas.py            # PLY → atlas .pt preprocessing
│   └── dataset.py             # AtlasCaptionDataset + dataloader factory
├── models/
│   ├── atlas_encoder.py       # ConvNeXt with 17-channel stem + projection head
│   └── clip_encoder.py        # Frozen CLIP text encoder wrapper
├── training/
│   ├── losses.py              # Symmetric InfoNCE + retrieval accuracy metrics
│   └── train.py               # Main training loop
├── evaluation/
│   └── compute_fid.py         # Feature extraction + FID computation
└── requirements.txt
```

---

## Quick Start

### Step 1 — Configure paths

Copy and edit `configs/config.yaml`:

```yaml
data:
  atlas_dir: "/path/to/precomputed/atlases"
  captions_json: "/path/to/captions.json"

gs2atlas:
  source_root: "/path/to/gaussianverse"
  save_root: "/path/to/atlases"
  sphere2plane_path: "/path/to/sphere2plane.npy"  # download below
```

Download `sphere2plane.npy` from GaussianVerse:
```bash
wget https://downloads.cs.stanford.edu/vision/gaussianverse/sphere2plane.npy
```

### Step 2 — Preprocess: PLY → Atlas

Create a text file listing scene paths (one per line, relative to `source_root`):

```
1876/9374307
1876/9381042
...
```

Then run:

```bash
python -m data.gs2atlas \
    --config configs/config.yaml \
    --txt_file scene_list.txt
```

This produces one `<cat_id>-<obj_id>.pt` file per scene in `gs2atlas.save_root`.

### Step 3 — Train

```bash
python -m training.train --config configs/config.yaml

# Resume from checkpoint
python -m training.train --config configs/config.yaml --resume checkpoints/epoch_0009.pth

# Disable W&B logging
python -m training.train --config configs/config.yaml --no-wandb
```

Checkpoints are saved to `training.output_dir` (default: `./checkpoints`).
The best checkpoint (lowest validation loss) is always saved as `best.pth`.

### Step 4 — Compute FID

```bash
python -m evaluation.compute_fid \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pth \
    --real /path/to/real/atlases \
    --gen  /path/to/generated/atlases
```

Output:
```
========================================
  3DGS-FID = 12.3456
========================================
```

Optionally save extracted feature arrays for later use:
```bash
python -m evaluation.compute_fid ... --save-features ./features/
```

---

## Configuration Reference

Key fields in `configs/config.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `data` | `atlas_channels` | `17` | Number of channels per atlas pixel |
| `data` | `atlas_resolution` | `128` | Grid side length (128 → 128×128) |
| `data` | `train_split` | `0.99` | Fraction of data used for training |
| `model` | `backbone` | `convnext_base` | timm backbone name |
| `model` | `embed_dim` | `512` | Output embedding dimension |
| `model` | `clip_model` | `ViT-B/32` | CLIP model for caption embeddings |
| `training` | `batch_size` | `256` | Samples per batch |
| `training` | `epochs` | `50` | Number of training epochs |
| `training` | `lr` | `1e-4` | Peak learning rate (AdamW) |
| `training` | `warmup_epochs` | `2` | Linear warmup duration |
| `training` | `mixed_precision` | `true` | Enable torch.amp (CUDA only) |

---

## Data Sources

This project is designed for use with [GaussianVerse](https://arxiv.org/abs/2503.15877) (205K high-quality 3DGS fittings) from the [3DGen-Playground](https://github.com/tiangexiang/3DGen-Playground).

Captions can be sourced from [Cap3D](https://github.com/crockwell/Cap3D) or [3DTopia](https://github.com/3DTopia/3DTopia), formatted as a flat JSON mapping `"<obj_id>": "<caption>"`.

---

## Citation

If you use this project, please also cite the works it builds upon:

```bibtex
@inproceedings{gaussianatlas2025,
  title     = {Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation},
  author    = {Xiang, Tiange and Li, Kai and Long, Chengjiang and H{\"a}ne, Christian
               and Guo, Peihong and Delp, Scott and Adeli, Ehsan and Fei-Fei, Li},
  booktitle = {ICCV},
  year      = {2025}
}

@misc{xiang2025_3dgen_playground,
  author       = {Tiange Xiang},
  title        = {3DGen-Playground},
  year         = {2025},
  howpublished = {\url{https://github.com/tiangexiang/3DGen-Playground}}
}
```
