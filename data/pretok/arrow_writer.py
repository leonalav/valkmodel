from __future__ import annotations

from typing import IO

import pyarrow as pa
import pyarrow.ipc as ipc

_SCHEMA = pa.schema([
    pa.field("doc_index", pa.int64()),
    pa.field("token_ids", pa.list_(pa.int32())),
    pa.field("token_count", pa.int32()),
])


class ArrowDocWriter:
    def __init__(self, sink: str | IO[bytes]) -> None:
        self._writer = ipc.new_file(sink, _SCHEMA)
        self._doc_index = 0
        self._closed = False

    def add_document(self, token_ids: list[int]) -> None:
        batch = pa.record_batch(
            {
                "doc_index": pa.array([self._doc_index], type=pa.int64()),
                "token_ids": pa.array([token_ids], type=pa.list_(pa.int32())),
                "token_count": pa.array([len(token_ids)], type=pa.int32()),
            },
            schema=_SCHEMA,
        )
        self._writer.write_batch(batch)
        self._doc_index += 1

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        self._closed = True

    def __enter__(self) -> "ArrowDocWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
