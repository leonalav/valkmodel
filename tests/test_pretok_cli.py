from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def _install_fake_tokenizer(monkeypatch):
    module = types.ModuleType("data.tokenizer_setup")

    class FakeTokenizer:
        eos_token_id = 99
        bos_token_id = 1
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) for ch in text]

    module.load_tokenizer = lambda *args, **kwargs: FakeTokenizer()
    module.get_tokenizer_metadata = lambda tokenizer: {
        "vocab_size": 256,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    monkeypatch.setitem(sys.modules, "data.tokenizer_setup", module)


class TestPretokCLI:
    def test_iter_tokenized_documents_for_spec_extracts_and_filters_rows(self, monkeypatch):
        _install_fake_tokenizer(monkeypatch)
        from data.dataset_registry import DatasetSpec
        from data.pretok.cli import iter_tokenized_documents_for_spec

        rows = iter([
            {"text": "ok"},
            {"text": ""},
            {"text": 123},
        ])
        monkeypatch.setattr("data.pretok.cli.load_stream_rows", lambda spec: rows)

        spec = DatasetSpec(name="fineweb_edu", hf_path="fake/fineweb")
        tokenizer = types.SimpleNamespace(encode=lambda text, add_special_tokens=False: [1, 2, 3] if text == "ok" else [])

        docs = list(iter_tokenized_documents_for_spec(spec, tokenizer))

        assert docs == [[1, 2, 3]]

    def test_build_command_writes_arrow_and_stage_parquet_outputs(self, tmp_path, monkeypatch):
        _install_fake_tokenizer(monkeypatch)
        from data.dataset_registry import DatasetRegistry, DatasetSpec
        from data.pretok.cli import main

        out_dir = tmp_path / "out"
        registry = DatasetRegistry({
            "fineweb_edu": DatasetSpec(name="fineweb_edu", hf_path="fake/fineweb", subset="sample-10BT"),
        })

        monkeypatch.setattr(
            "data.pretok.cli.build_pretok_registry",
            lambda _registry: registry,
        )
        monkeypatch.setattr(
            "data.pretok.cli.iter_tokenized_documents_for_spec",
            lambda spec, tokenizer, limit=None: iter([[1, 2, 3], [4]]),
        )

        exit_code = main([
            "build",
            "--output-dir", str(out_dir),
            "--tokenizer", "fake-tokenizer",
            "--dataset", "fineweb_edu",
            "--stage", "4",
            "--num-workers", "2",
            "--shard-size", "7",
        ])

        assert exit_code == 0

        arrow_path = out_dir / "canonical" / "fineweb_edu.arrow"
        parquet_path = out_dir / "packed" / "stage_4" / "fineweb_edu.parquet"
        manifest_path = out_dir / "manifest.json"

        assert arrow_path.exists()
        assert parquet_path.exists()
        assert manifest_path.exists()

        with ipc.open_file(arrow_path) as reader:
            arrow_table = reader.read_all()
        parquet_table = pq.read_table(parquet_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert arrow_table.column("token_ids").to_pylist() == [[1, 2, 3], [4]]
        assert parquet_table.column("input_ids").to_pylist() == [[1, 2, 3, 99]]
        assert manifest["datasets"] == ["fineweb_edu"]
        assert manifest["stages"] == [4]

    def test_publish_command_uploads_directory_to_hf_repo(self, tmp_path, monkeypatch):
        from data.pretok.cli import main

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "manifest.json").write_text("{}", encoding="utf-8")
        uploads: list[dict[str, object]] = []

        monkeypatch.setattr(
            "data.pretok.cli.upload_output_to_hf",
            lambda output_dir, repo_id, revision=None: uploads.append(
                {
                    "output_dir": Path(output_dir),
                    "repo_id": repo_id,
                    "revision": revision,
                }
            ),
        )

        exit_code = main([
            "publish",
            "--output-dir", str(out_dir),
            "--repo-id", "leonidas123/valkmodel-data",
        ])

        assert exit_code == 0
        assert uploads == [
            {
                "output_dir": out_dir,
                "repo_id": "leonidas123/valkmodel-data",
                "revision": None,
            }
        ]
