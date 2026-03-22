"""
precompute_clip.py: Pre-encode all captions with CLIP and save to disk.

This is a one-time preprocessing step that eliminates the cost of running the
frozen CLIP text encoder during every training step.

Output: a single .pt file containing a dict { obj_id (str) -> embed (Tensor, D) }
All embeddings are float32 and L2-normalised (unit norm).

Usage:
    python -m data.precompute_clip --config configs/config.yaml

The output path is read from config under data.clip_embeddings_file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def precompute(
    captions_json: str,
    output_path: str,
    clip_model: str = "ViT-B/32",
    batch_size: int = 512,
    device: str = "cuda",
) -> None:
    import clip  # imported here so the module can be imported without clip installed

    print(f"Loading CLIP {clip_model} on {device} ...")
    model, _ = clip.load(clip_model, device=device, jit=False)
    model.eval()

    with open(captions_json) as f:
        captions: dict[str, str] = json.load(f)

    obj_ids = list(captions.keys())
    caption_texts = [captions[k] for k in obj_ids]

    print(f"Encoding {len(obj_ids):,} captions in batches of {batch_size} ...")
    all_embeds: list[torch.Tensor] = []

    for start in tqdm(range(0, len(caption_texts), batch_size)):
        batch_texts = caption_texts[start : start + batch_size]
        tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        with torch.no_grad():
            embed = model.encode_text(tokens).float()
            embed = F.normalize(embed, dim=-1)
        all_embeds.append(embed.cpu())

    all_embeds_cat = torch.cat(all_embeds, dim=0)  # (N, D)

    embed_dict: dict[str, torch.Tensor] = {
        obj_id: all_embeds_cat[i] for i, obj_id in enumerate(obj_ids)
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embed_dict, output_path)
    print(f"Saved {len(embed_dict):,} embeddings → {output_path}")
    print(f"Embedding dim: {all_embeds_cat.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-encode captions with CLIP")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    output_path = data_cfg.get("clip_embeddings_file")
    if not output_path:
        raise ValueError(
            "Set data.clip_embeddings_file in config.yaml to specify the output path."
        )

    precompute(
        captions_json=data_cfg["captions_json"],
        output_path=output_path,
        clip_model=cfg["model"].get("clip_model", "ViT-B/32"),
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
