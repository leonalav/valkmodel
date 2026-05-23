from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def _empty_pack(max_seq_len: int, pad_token_id: int) -> dict[str, torch.Tensor | list[tuple[int, int]]]:
    return {
        "input_ids": torch.full((max_seq_len,), pad_token_id, dtype=torch.long),
        "labels": torch.full((max_seq_len,), IGNORE_INDEX, dtype=torch.long),
        "attention_mask": torch.zeros(max_seq_len, dtype=torch.long),
        "document_ids": torch.full((max_seq_len,), -1, dtype=torch.long),
        "document_boundaries": [],
    }


def _append_pack(
    packs: list[dict[str, torch.Tensor | list[tuple[int, int]]]],
    tokens: list[int],
    doc_ids: list[int],
    boundaries: list[tuple[int, int]],
    max_seq_len: int,
    pad_token_id: int,
) -> None:
    pad_len = max_seq_len - len(tokens)
    input_ids = torch.tensor(tokens + [pad_token_id] * pad_len, dtype=torch.long)
    labels = torch.tensor(tokens + [IGNORE_INDEX] * pad_len, dtype=torch.long)
    attention_mask = torch.tensor([1] * len(tokens) + [0] * pad_len, dtype=torch.long)
    document_ids = torch.tensor(doc_ids + [-1] * pad_len, dtype=torch.long)
    packs.append(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "document_ids": document_ids,
            "document_boundaries": boundaries,
        }
    )


def pack_documents(
    documents: Sequence[Sequence[int]],
    max_seq_len: int,
    pad_token_id: int,
    separator_token_id: int | None = None,
) -> list[dict[str, torch.Tensor | list[tuple[int, int]]]]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    packs: list[dict[str, torch.Tensor | list[tuple[int, int]]]] = []
    tokens: list[int] = []
    doc_ids: list[int] = []
    boundaries: list[tuple[int, int]] = []

    for doc_index, document in enumerate(documents):
        start = 0
        document_tokens = list(document)
        while start < len(document_tokens):
            remaining = max_seq_len - len(tokens)
            if remaining == 0:
                _append_pack(packs, tokens, doc_ids, boundaries, max_seq_len, pad_token_id)
                tokens, doc_ids, boundaries = [], [], []
                remaining = max_seq_len

            take = min(remaining, len(document_tokens) - start)
            chunk = document_tokens[start : start + take]
            boundary_start = len(tokens)
            tokens.extend(chunk)
            doc_ids.extend([doc_index] * len(chunk))
            boundaries.append((boundary_start, boundary_start + len(chunk)))
            start += take

            if len(tokens) == max_seq_len:
                _append_pack(packs, tokens, doc_ids, boundaries, max_seq_len, pad_token_id)
                tokens, doc_ids, boundaries = [], [], []

        if separator_token_id is not None and doc_index != len(documents) - 1:
            if len(tokens) == max_seq_len:
                _append_pack(packs, tokens, doc_ids, boundaries, max_seq_len, pad_token_id)
                tokens, doc_ids, boundaries = [], [], []
            tokens.append(separator_token_id)
            doc_ids.append(-1)

    if tokens and len(tokens) == max_seq_len:
        _append_pack(packs, tokens, doc_ids, boundaries, max_seq_len, pad_token_id)
    if not packs:
        packs.append(_empty_pack(max_seq_len, pad_token_id))
    return packs


def create_document_attention_mask(document_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if document_ids.shape != attention_mask.shape:
        raise ValueError("document_ids and attention_mask must have the same shape")
    active = attention_mask.to(dtype=torch.bool)
    same_document = document_ids.unsqueeze(-1) == document_ids.unsqueeze(-2)
    valid_documents = document_ids.ge(0).unsqueeze(-1) & document_ids.ge(0).unsqueeze(-2)
    active_pairs = active.unsqueeze(-1) & active.unsqueeze(-2)
    return same_document & valid_documents & active_pairs


class PackedDataset(Dataset):
    def __init__(
        self,
        documents: Sequence[Sequence[int]],
        max_seq_len: int,
        pad_token_id: int,
        separator_token_id: int | None = None,
    ):
        self.packs = pack_documents(documents, max_seq_len, pad_token_id, separator_token_id)

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[tuple[int, int]]]:
        return self.packs[index]
