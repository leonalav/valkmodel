from __future__ import annotations

import io

import pyarrow.parquet as pq

from data.pretok.parquet_packer import ParquetPacker


class TestParquetPacker:
    def test_packs_documents_with_one_eos_between_each_document(self):
        sink = io.BytesIO()
        packer = ParquetPacker(sink, block_size=4, eos_token_id=99)
        packer.add_document([1, 2, 3])
        packer.add_document([4])
        packer.flush()
        sink.seek(0)

        table = pq.read_table(sink)

        assert table.column("input_ids").to_pylist() == [[1, 2, 3, 99]]

    def test_discards_partial_tail_without_padding(self):
        sink = io.BytesIO()
        packer = ParquetPacker(sink, block_size=5, eos_token_id=99)
        packer.add_document([1, 2])
        packer.flush()
        sink.seek(0)

        table = pq.read_table(sink)

        assert len(table) == 0
