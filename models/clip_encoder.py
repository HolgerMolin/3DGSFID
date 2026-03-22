"""
CLIPTextEncoder: frozen CLIP text encoder wrapper.

Loads OpenAI CLIP and exposes only the text encoding pathway.
All parameters are frozen — this module is used purely for supervision.
"""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder that returns L2-normalised embeddings.

    Args:
        model_name: CLIP model identifier, e.g. "ViT-B/32".
        device:     Torch device string.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu") -> None:
        super().__init__()
        model, _ = clip.load(model_name, device=device, jit=False)
        self.clip_model = model
        self.embed_dim: int = model.text_projection.shape[1]  # 512 for ViT-B/32

        # Freeze all parameters — CLIP is only used for target embeddings
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        self.clip_model.eval()

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, captions: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode a list of caption strings into L2-normalised embeddings.

        Args:
            captions: List of strings or pre-tokenised tensor.
        Returns:
            (B, embed_dim) float32 tensor on the same device as the model.
        """
        if isinstance(captions, list):
            tokens = clip.tokenize(captions, truncate=True).to(
                next(self.clip_model.parameters()).device
            )
        else:
            tokens = captions

        embed = self.clip_model.encode_text(tokens).float()
        embed = F.normalize(embed, dim=-1)
        return embed

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, mode: bool = True):
        """Always keep CLIP in eval mode regardless of the outer training flag."""
        super().train(mode)
        self.clip_model.eval()
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_clip_encoder(cfg: dict, device: str = "cpu") -> CLIPTextEncoder:
    """Construct CLIPTextEncoder from the 'model' section of the YAML config."""
    model_name = cfg["model"].get("clip_model", "ViT-B/32")
    return CLIPTextEncoder(model_name=model_name, device=device)
