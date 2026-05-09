from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .dataset_registry import DatasetRegistry
from .eval_collator import EvalDataCollator
from .mixture_config import load_mixture_config
from .mixture_dataset import StreamingMixtureDataset
from .packed_collator import PackedDataCollator


def build_training_dataloader(
    mixture_config_path: str | Path,
    tokenizer: Any,
    block_size: int,
    batch_size: int,
    registry: DatasetRegistry | None = None,
    document_streams: Mapping[str, Iterator[str]] | None = None,
    num_workers: int = 0,
) -> DataLoader:
    _, mixture = load_mixture_config(mixture_config_path, registry or DatasetRegistry())
    dataset = StreamingMixtureDataset(
        mixture=mixture,
        tokenizer=tokenizer,
        block_size=block_size,
        document_streams=document_streams,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=PackedDataCollator(tokenizer=tokenizer, block_size=block_size),
    )


def build_eval_dataloader(examples: list[dict[str, list[int]]], tokenizer: Any, batch_size: int) -> DataLoader:
    return DataLoader(
        examples,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=EvalDataCollator(tokenizer),
    )
