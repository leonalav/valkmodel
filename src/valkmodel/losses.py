from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_weighted_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tool_mask: torch.Tensor | None = None,
    tool_weight: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shift_labels)
    active = shift_labels.ne(ignore_index)
    weights = torch.ones_like(per_token)
    if tool_mask is not None:
        if tool_mask.shape != labels.shape:
            raise ValueError("tool_mask must have the same shape as labels")
        weights = torch.where(tool_mask[:, 1:].to(dtype=torch.bool), torch.full_like(weights, tool_weight), weights)
    weights = weights * active.to(dtype=weights.dtype)
    denominator = weights.sum()
    if denominator == 0:
        return logits.sum() * 0
    return (per_token * weights).sum() / denominator


def compute_branch_value_loss(
    branch_values: torch.Tensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if branch_values.ndim != 3:
        raise ValueError("branch_values must have shape (batch, sequence, num_branches)")
    if target_values.shape != branch_values.shape[:2]:
        raise ValueError("target_values must have shape (batch, sequence)")
    if mask is None:
        active = torch.ones_like(target_values, dtype=torch.bool)
    else:
        if mask.shape != target_values.shape:
            raise ValueError("mask must have shape (batch, sequence)")
        active = mask.to(dtype=torch.bool)
    if not active.any():
        return branch_values.sum() * 0
    branch_probs = torch.softmax(branch_values, dim=-1)
    expected_values = (branch_probs * branch_values).sum(dim=-1)
    return F.mse_loss(expected_values[active], target_values[active])
