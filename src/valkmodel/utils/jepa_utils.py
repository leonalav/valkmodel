from __future__ import annotations

import torch
import torch.nn.functional as F


def create_jepa_pairs(
    latent_state: torch.Tensor,
    horizons: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latent_state.ndim != 3:
        raise ValueError("latent_state must have shape (batch, sequence, latent_dim)")
    if horizons.shape != latent_state.shape[:2]:
        raise ValueError("horizons must have shape (batch, sequence)")
    if torch.any(horizons < 1):
        raise ValueError("horizons must be positive")

    batch_size, seq_len, latent_dim = latent_state.shape
    positions = torch.arange(seq_len, device=latent_state.device).view(1, seq_len).expand(batch_size, seq_len)
    target_positions = positions + horizons.to(device=latent_state.device)
    valid_mask = target_positions < seq_len
    clamped_positions = target_positions.clamp(max=seq_len - 1)
    batch_indices = torch.arange(batch_size, device=latent_state.device).view(batch_size, 1).expand(batch_size, seq_len)
    future = latent_state[batch_indices, clamped_positions]
    future = torch.where(valid_mask.unsqueeze(-1), future, torch.zeros_like(future))
    return latent_state, future.view(batch_size, seq_len, latent_dim), valid_mask


def compute_normalized_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if mask.shape != predictions.shape[:2]:
        raise ValueError("mask must have shape (batch, sequence)")
    active = mask.to(dtype=torch.bool)
    if not active.any():
        return predictions.sum() * 0
    pred = F.normalize(predictions[active], dim=-1, eps=eps)
    target = F.normalize(targets[active], dim=-1, eps=eps)
    return (pred - target).pow(2).sum(dim=-1).mean()


def compute_jepa_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    collapse_variance_threshold: float = 1e-4,
) -> dict[str, torch.Tensor]:
    active = mask.to(dtype=torch.bool)
    if not active.any():
        zero = predictions.sum() * 0
        false_flag = torch.zeros((), dtype=torch.bool, device=predictions.device)
        return {
            "prediction_variance": zero,
            "target_variance": zero,
            "cosine_mean": zero,
            "prediction_collapsed": false_flag,
            "target_collapsed": false_flag,
        }
    pred = predictions[active]
    target = targets[active]
    pred_norm = F.normalize(pred, dim=-1, eps=eps)
    target_norm = F.normalize(target, dim=-1, eps=eps)
    prediction_variance = pred.var(dim=0, unbiased=False).mean()
    target_variance = target.var(dim=0, unbiased=False).mean()
    return {
        "prediction_variance": prediction_variance,
        "target_variance": target_variance,
        "cosine_mean": (pred_norm * target_norm).sum(dim=-1).mean(),
        "prediction_collapsed": prediction_variance < collapse_variance_threshold,
        "target_collapsed": target_variance < collapse_variance_threshold,
    }
