from __future__ import annotations

import torch


def create_tool_mask(
    input_ids: torch.Tensor,
    tool_call_token_id: int | None = None,
    tool_result_token_id: int | None = None,
    reasoning_start_token_id: int | None = None,
    reasoning_end_token_id: int | None = None,
    branch_marker_token_id: int | None = None,
    tool_call_span: int = 32,
    tool_result_span: int = 64,
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence)")
    if tool_call_span < 0 or tool_result_span < 0:
        raise ValueError("tool spans must be nonnegative")

    _mark_token_windows(mask, input_ids, tool_call_token_id, tool_call_span)
    _mark_token_windows(mask, input_ids, tool_result_token_id, tool_result_span)
    _mark_token_windows(mask, input_ids, branch_marker_token_id, 0)
    _mark_reasoning_spans(mask, input_ids, reasoning_start_token_id, reasoning_end_token_id)
    return mask


def _mark_token_windows(mask: torch.Tensor, input_ids: torch.Tensor, token_id: int | None, span: int) -> None:
    if token_id is None:
        return
    seq_len = input_ids.shape[1]
    for batch_idx, position in (input_ids == token_id).nonzero(as_tuple=False):
        start = int(position)
        end = min(seq_len, start + span + 1)
        mask[int(batch_idx), start:end] = True


def _mark_reasoning_spans(
    mask: torch.Tensor,
    input_ids: torch.Tensor,
    start_token_id: int | None,
    end_token_id: int | None,
) -> None:
    if start_token_id is None:
        return
    for batch_idx in range(input_ids.shape[0]):
        inside = False
        for position in range(input_ids.shape[1]):
            token = int(input_ids[batch_idx, position])
            if token == start_token_id:
                inside = True
            if inside:
                mask[batch_idx, position] = True
            if end_token_id is not None and token == end_token_id and inside:
                inside = False
