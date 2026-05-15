"""
ShardDataset: reads pre-tokenized Parquet shards for packed training.

Directory layout expected:
    <shard_root>/
        stage_<block_size>/
            shard_0000.parquet
            shard_0001.parquet
            ...

Each Parquet file must have an `input_ids` column where every row is a
list of exactly `block_size` integers (already packed by the pretokenizer).

The dataset yields dicts with keys:
    input_ids       : LongTensor[block_size]
    labels          : LongTensor[block_size]  (copy of input_ids)
    attention_mask  : LongTensor[block_size]  (all ones — no padding in packed shards)

This matches the contract expected by PackedDataCollator and the training loop.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class ShardDataset(Dataset):
    """
    Map-style dataset over pre-tokenized Parquet shards for a given block_size.

    Parameters
    ----------
    shard_root:
        Root directory containing stage_<block_size>/ subdirectories.
    block_size:
        Context length for the current curriculum stage.  The dataset reads
        from ``shard_root / f"stage_{block_size}"``.
    """

    def __init__(self, shard_root: str | Path, block_size: int) -> None:
        self.shard_root = Path(shard_root)
        self.block_size = block_size
        self._stage_dir = self.shard_root / f"stage_{block_size}"
        if not self._stage_dir.exists():
            raise FileNotFoundError(
                f"ShardDataset: stage directory not found: {self._stage_dir}"
            )
        shard_files = sorted(self._stage_dir.glob("*.parquet"))
        if not shard_files:
            raise FileNotFoundError(
                f"ShardDataset: no .parquet files found in {self._stage_dir}"
            )
        # Load all shards into memory as a list of rows for O(1) random access.
        # For very large datasets this should be replaced with an index-based
        # lazy reader, but for the training path this is sufficient.
        self._rows: list[list[int]] = []
        for shard_file in shard_files:
            table = pq.read_table(shard_file, columns=["input_ids"])
            for row in table.to_pydict()["input_ids"]:
                self._rows.append(list(row))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self._rows[index], dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones(self.block_size, dtype=torch.long),
        }
