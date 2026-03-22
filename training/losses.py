"""
Contrastive losses for aligning Gaussian Atlas embeddings with CLIP embeddings.

Primary loss: symmetric InfoNCE (as used in CLIP / ALIGN).
  Given a batch of N (atlas_embed, clip_embed) pairs, the loss encourages
  each atlas embedding to be close to its paired caption embedding and far
  from all others in the batch.

  L = 0.5 * (cross_entropy(logits, labels) + cross_entropy(logits.T, labels))
  where logits[i, j] = dot(atlas_i, clip_j) / temperature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (contrastive) loss.

    The temperature is read from the encoder's `log_temperature` parameter so
    it participates in backprop.  Alternatively, a fixed scalar can be provided.

    Args:
        temperature: Fixed temperature for the softmax.  If None, temperature
                     must be supplied in `forward()` via the `temperature` kwarg
                     (e.g. from `encoder.temperature`).
    """

    def __init__(self, temperature: float | None = None) -> None:
        super().__init__()
        self._fixed_temperature = temperature

    def forward(
        self,
        atlas_embed: Tensor,
        clip_embed: Tensor,
        temperature: Tensor | float | None = None,
    ) -> Tensor:
        """
        Args:
            atlas_embed: (B, D) L2-normalised atlas embeddings.
            clip_embed:  (B, D) L2-normalised CLIP text embeddings.
            temperature: Scalar temperature.  Falls back to self._fixed_temperature
                         if not provided.
        Returns:
            Scalar loss.
        """
        if temperature is None:
            if self._fixed_temperature is not None:
                temperature = self._fixed_temperature
            else:
                raise ValueError(
                    "temperature must be provided either at construction or in forward()."
                )

        B = atlas_embed.shape[0]
        labels = torch.arange(B, device=atlas_embed.device)

        # Cosine similarity matrix — (B, B)
        # atlas_embed and clip_embed are already L2-normalised
        logits = torch.matmul(atlas_embed, clip_embed.t()) / temperature  # (B, B)

        # Symmetric cross-entropy
        loss_atlas_to_clip = F.cross_entropy(logits, labels)
        loss_clip_to_atlas = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_atlas_to_clip + loss_clip_to_atlas)

        return loss


class AlignmentLoss(nn.Module):
    """
    Combined loss: InfoNCE + optional MSE alignment regulariser.

    The MSE term pulls atlas embeddings directly towards their paired CLIP
    embeddings in embedding space, providing a softer signal than pure
    contrastive learning.

    Args:
        mse_weight: Weight for the MSE regulariser (0 to disable).
        temperature: Fixed temperature (or None to use encoder's learnable one).
    """

    def __init__(
        self,
        mse_weight: float = 0.0,
        temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.infonce = InfoNCELoss(temperature=temperature)
        self.mse_weight = mse_weight

    def forward(
        self,
        atlas_embed: Tensor,
        clip_embed: Tensor,
        temperature: Tensor | float | None = None,
    ) -> dict[str, Tensor]:
        """
        Returns a dict with keys 'loss', 'infonce', and (if mse_weight>0) 'mse'.
        """
        contrastive = self.infonce(atlas_embed, clip_embed, temperature=temperature)
        result = {"infonce": contrastive, "loss": contrastive}

        if self.mse_weight > 0.0:
            mse = F.mse_loss(atlas_embed, clip_embed.detach())
            result["mse"] = mse
            result["loss"] = contrastive + self.mse_weight * mse

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieval_accuracy(
    atlas_embed: Tensor,
    clip_embed: Tensor,
    top_k: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """
    Compute atlas→text and text→atlas top-k retrieval accuracy.

    Args:
        atlas_embed: (B, D) L2-normalised.
        clip_embed:  (B, D) L2-normalised.
        top_k:       Tuple of k values to evaluate.

    Returns:
        Dict with keys like "atlas2text_top1", "text2atlas_top5", etc.
    """
    sim = torch.matmul(atlas_embed, clip_embed.t())  # (B, B)
    B = sim.shape[0]
    labels = torch.arange(B, device=sim.device)
    results: dict[str, float] = {}

    for direction, logits in [("atlas2text", sim), ("text2atlas", sim.t())]:
        for k in top_k:
            _, preds = logits.topk(k, dim=1)
            correct = preds.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
            results[f"{direction}_top{k}"] = correct * 100.0

    return results
