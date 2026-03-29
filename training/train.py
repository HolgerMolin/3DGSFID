"""
Training script: align ConvNeXt atlas embeddings with frozen CLIP text embeddings.

Usage:
    python -m training.train --config configs/config.yaml

Optional flags:
    --no-wandb     Disable Weights & Biases logging
    --resume PATH  Resume from a checkpoint file
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import build_dataloaders
from models.atlas_encoder import build_encoder
from models.clip_encoder import build_clip_encoder
from training.losses import AlignmentLoss, retrieval_accuracy

# Sentinel: indicates that CLIP embeddings come pre-loaded in each batch
_PRECOMPUTED = object()


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    epoch: int,
    encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"] + 1, ckpt["best_val_loss"]


# ──────────────────────────────────────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    encoder: nn.Module,
    clip_encoder,  # CLIPTextEncoder instance, or _PRECOMPUTED sentinel
    loader,
    loss_fn: AlignmentLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
    epoch: int,
    wandb_run=None,
    global_step: int = 0,
) -> tuple[float, int]:
    """Run one train or validation epoch. Returns (mean_loss, updated_global_step).

    When clip_encoder is _PRECOMPUTED, each batch is expected to contain a
    'clip_embed' tensor (produced by AtlasCaptionDataset with precomputed
    embeddings) and no CLIP forward pass is performed.
    """
    is_train = optimizer is not None
    encoder.train(is_train)
    use_precomputed = clip_encoder is _PRECOMPUTED
    if not use_precomputed:
        clip_encoder.eval()

    train_cfg = cfg["training"]
    log_interval = train_cfg.get("log_interval", 50)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"

    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        bar = tqdm(loader, desc=f"{'Train' if is_train else 'Val'} epoch {epoch}", leave=False)
        for batch in bar:
            atlases = batch["atlas"].to(device, non_blocking=True)  # (B, C, H, W)

            with autocast("cuda", enabled=use_amp):
                atlas_embed = encoder(atlases)

                if use_precomputed:
                    clip_embed = batch["clip_embed"].to(device, non_blocking=True)
                else:
                    captions: list[str] = batch["caption"]
                    clip_embed = clip_encoder(captions).to(device)

                loss_dict = loss_fn(
                    atlas_embed,
                    clip_embed,
                    temperature=encoder.temperature if is_train else encoder.temperature.detach(),
                )
                loss = loss_dict["loss"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                if global_step % log_interval == 0:
                    acc = retrieval_accuracy(
                        atlas_embed.detach(), clip_embed.detach(), top_k=(1, 5, 10)
                    )
                    log_data = {
                        "train/loss": loss.item(),
                        "train/infonce": loss_dict["infonce"].item(),
                        "train/temperature": encoder.temperature.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        **{f"train/{k}": v for k, v in acc.items()},
                        "step": global_step,
                    }
                    if "mse" in loss_dict:
                        log_data["train/mse"] = loss_dict["mse"].item()
                    if wandb_run:
                        wandb_run.log(log_data)
                    bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        T=f"{encoder.temperature.item():.3f}",
                        a2t1=f"{acc.get('atlas2text_top1', 0.0):.1f}%",
                        a2t5=f"{acc.get('atlas2text_top5', 0.0):.1f}%",
                        a2t10=f"{acc.get('atlas2text_top10', 0.0):.1f}%",
                    )

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), global_step


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train atlas → CLIP contrastive encoder")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=train_cfg.get("wandb_project", "3dgs-fid"),
                name=train_cfg.get("wandb_run_name"),
                config=cfg,
            )
        except Exception as e:
            print(f"[WARNING] W&B init failed ({e}); continuing without logging.")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = steps_per_epoch * train_cfg.get("warmup_epochs", 2)

    # ── Models ────────────────────────────────────────────────────────────────
    print("Building models...")
    encoder = build_encoder(cfg).to(device)

    # Skip loading CLIP entirely when precomputed embeddings are available.
    # The dataset already yields 'clip_embed' tensors in that case.
    data_cfg = cfg["data"]
    clip_embed_file = data_cfg.get("clip_embeddings_file")
    import os as _os
    if clip_embed_file and _os.path.exists(clip_embed_file):
        print("Precomputed CLIP embeddings detected — CLIP encoder will not be loaded.")
        clip_encoder = _PRECOMPUTED
    else:
        clip_encoder = build_clip_encoder(cfg, device=str(device))

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = AlignmentLoss(mse_weight=train_cfg.get("mse_weight", 0.0))

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.05),
        betas=(0.9, 0.999),
    )
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler("cuda", enabled=train_cfg.get("mixed_precision", True) and device.type == "cuda")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, encoder, optimizer, scheduler, scaler, device
        )
        global_step = start_epoch * steps_per_epoch

    # ── Training loop ─────────────────────────────────────────────────────────
    save_interval = train_cfg.get("save_interval", 5)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        train_loss, global_step = run_epoch(
            encoder, clip_encoder, train_loader, loss_fn,
            optimizer, scheduler, scaler, device, cfg,
            epoch=epoch, wandb_run=wandb_run, global_step=global_step,
        )

        val_loss, _ = run_epoch(
            encoder, clip_encoder, val_loader, loss_fn,
            optimizer=None, scheduler=scheduler, scaler=scaler,
            device=device, cfg=cfg, epoch=epoch,
            wandb_run=wandb_run, global_step=global_step,
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  T={encoder.temperature.item():.4f}"
        )

        if wandb_run:
            wandb_run.log({"epoch": epoch, "val/loss": val_loss, "train/epoch_loss": train_loss})

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                str(output_dir / "best.pth"),
                epoch, encoder, optimizer, scheduler, scaler, best_val_loss,
            )
            print(f"  ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                str(output_dir / f"epoch_{epoch:04d}.pth"),
                epoch, encoder, optimizer, scheduler, scaler, best_val_loss,
            )

    print("Training complete.")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
