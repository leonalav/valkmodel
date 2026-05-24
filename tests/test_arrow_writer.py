from __future__ import annotations

import io

import pyarrow.ipc as ipc

from data.pretok.arrow_writer import ArrowDocWriter


class TestArrowDocWriter:
    def test_writes_exact_token_ids_without_eos_injection(self):
        sink = io.BytesIO()
        writer = ArrowDocWriter(sink)
        writer.add_document([11, 12, 13])
        writer.add_document([21])
        writer.close()
        sink.seek(0)

        with ipc.open_file(sink) as reader:
            table = reader.read_all()

        assert table.column("doc_index").to_pylist() == [0, 1]
        assert table.column("token_ids").to_pylist() == [[11, 12, 13], [21]]
        assert table.column("token_count").to_pylist() == [3, 1]

    def test_accepts_context_manager_usage(self):
        sink = io.BytesIO()

        with ArrowDocWriter(sink) as writer:
            writer.add_document([1, 2])

        sink.seek(0)
        with ipc.open_file(sink) as reader:
            table = reader.read_all()

        assert table.column("token_ids").to_pylist() == [[1, 2]]
