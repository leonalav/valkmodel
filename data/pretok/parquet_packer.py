from __future__ import annotations

from typing import IO

import pyarrow as pa
import pyarrow.parquet as pq

from data.token_stream_builder import TokenStreamBuilder

_SCHEMA = pa.schema([
    pa.field("block_index", pa.int64()),
    pa.field("input_ids", pa.list_(pa.int32())),
])


class ParquetPacker:
    def __init__(self, sink: str | IO[bytes], *, block_size: int, eos_token_id: int) -> None:
        self._writer = pq.ParquetWriter(sink, _SCHEMA)
        self._stream = TokenStreamBuilder(eos_token_id=eos_token_id)
        self._block_size = block_size
        self._block_index = 0
        self._flushed = False

    def add_document(self, token_ids: list[int]) -> None:
        self._stream.add_document(token_ids)
        self._emit_ready_blocks()

    def _emit_ready_blocks(self) -> None:
        for block in self._stream.iter_blocks(self._block_size):
            batch = pa.record_batch(
                {
                    "block_index": pa.array([self._block_index], type=pa.int64()),
                    "input_ids": pa.array([block], type=pa.list_(pa.int32())),
                },
                schema=_SCHEMA,
            )
            self._writer.write_batch(batch)
            self._block_index += 1

    def flush(self) -> None:
        if self._flushed:
            return
        self._emit_ready_blocks()
        self._writer.close()
        self._flushed = True

    def __enter__(self) -> "ParquetPacker":
        return self

    def __exit__(self, *_: object) -> None:
        self.flush()
