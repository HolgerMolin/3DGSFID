"""
AtlasEncoder: ConvNeXt backbone adapted for multi-channel Gaussian Atlas input.

The first convolutional layer is re-initialised to accept `in_chans` input
channels (default 17) rather than the standard 3 RGB channels.  Pretrained
weights for all other layers are preserved.

Output: L2-normalised embedding of dimension `embed_dim` (default 512,
matching CLIP ViT-B/32 text embeddings).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AtlasEncoder(nn.Module):
    """
    ConvNeXt encoder for Gaussian Atlas tensors.

    Args:
        backbone:   timm model name, e.g. "convnext_base".
        in_chans:   Number of atlas channels (typically 17).
        embed_dim:  Output embedding dimension (must match CLIP text dim).
        pretrained: Whether to initialise the backbone from ImageNet weights.
    """

    def __init__(
        self,
        backbone: str = "convnext_base",
        in_chans: int = 17,
        embed_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        # Load with 3-channel weights first, then adapt the stem.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=3,       # load standard pretrained weights
            num_classes=0,    # remove classification head → returns feature map
        )

        # Re-initialise the first conv to accept `in_chans` channels.
        # Strategy: average the pretrained RGB weights across the channel dim
        # and replicate/scale them to `in_chans` so training starts close to
        # the pretrained representation.
        stem_conv: nn.Conv2d = self._get_stem_conv()
        old_weight = stem_conv.weight.data.clone()  # (out, 3, kH, kW)
        out_ch, _, kH, kW = old_weight.shape

        # Replicate by tiling and then rescaling by 3 / in_chans
        repeats = math.ceil(in_chans / 3)
        new_weight = old_weight.repeat(1, repeats, 1, 1)[:, :in_chans, :, :]
        new_weight = new_weight * (3.0 / in_chans)

        new_conv = nn.Conv2d(
            in_chans,
            out_ch,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=stem_conv.bias is not None,
        )
        new_conv.weight = nn.Parameter(new_weight)
        if stem_conv.bias is not None:
            new_conv.bias = nn.Parameter(stem_conv.bias.data.clone())

        self._set_stem_conv(new_conv)

        # ── Projection head ───────────────────────────────────────────────────
        backbone_dim = self.backbone.num_features  # e.g. 1024 for convnext_base
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
            nn.Linear(backbone_dim, embed_dim),
        )

        # Learnable temperature (log-scale), initialised following OpenAI CLIP
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, atlas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atlas: (B, C, H, W) float32 tensor.
        Returns:
            embed: (B, embed_dim) L2-normalised embedding.
        """
        features = self.backbone(atlas)          # (B, backbone_dim)
        embed = self.proj(features)              # (B, embed_dim)
        embed = F.normalize(embed, dim=-1)       # unit sphere
        return embed

    @property
    def temperature(self) -> torch.Tensor:
        """Clipped temperature scalar (≥ 0.01 for numerical stability)."""
        return self.log_temperature.exp().clamp(min=0.01)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers for stem conv access (works across timm versions)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_stem_conv(self) -> nn.Conv2d:
        # ConvNeXt stem layout: backbone.stem[0] or backbone.stem_0
        if hasattr(self.backbone, "stem"):
            stem = self.backbone.stem
            if isinstance(stem, nn.Sequential):
                return stem[0]
            if isinstance(stem, nn.Conv2d):
                return stem
        # Fallback: walk first named conv
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                return module
        raise RuntimeError("Could not find stem Conv2d in backbone.")

    def _set_stem_conv(self, new_conv: nn.Conv2d) -> None:
        if hasattr(self.backbone, "stem"):
            stem = self.backbone.stem
            if isinstance(stem, nn.Sequential):
                stem[0] = new_conv
                return
            if isinstance(stem, nn.Conv2d):
                self.backbone.stem = new_conv
                return
        # Fallback: find the attribute name of the first conv and replace it
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                parts = name.split(".")
                parent = self.backbone
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_conv)
                return
        raise RuntimeError("Could not set stem Conv2d in backbone.")


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_encoder(cfg: dict) -> AtlasEncoder:
    """Construct AtlasEncoder from the 'model' section of the YAML config."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return AtlasEncoder(
        backbone=model_cfg.get("backbone", "convnext_base"),
        in_chans=data_cfg.get("atlas_channels", 17),
        embed_dim=model_cfg.get("embed_dim", 512),
        pretrained=model_cfg.get("pretrained", True),
    )
