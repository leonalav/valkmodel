from __future__ import annotations

import sys
import types


def _install_fake_tokenizer(monkeypatch):
    module = types.ModuleType("data.tokenizer_setup")

    class FakeTokenizer:
        eos_token_id = 99
        bos_token_id = 1
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

    module.load_tokenizer = lambda *args, **kwargs: FakeTokenizer()
    module.get_tokenizer_metadata = lambda tokenizer: {
        "vocab_size": 256,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    monkeypatch.setitem(sys.modules, "data.tokenizer_setup", module)


class TestPretokLogging:
    def test_build_prints_start_finish_and_step_messages(self, tmp_path, monkeypatch, capsys):
        _install_fake_tokenizer(monkeypatch)
        from data.dataset_registry import DatasetRegistry, DatasetSpec
        from data.pretok.cli import main

        registry = DatasetRegistry({
            "fineweb_edu": DatasetSpec(name="fineweb_edu", hf_path="fake/fineweb", subset="sample-10BT"),
        })
        monkeypatch.setattr("data.pretok.cli.build_pretok_registry", lambda _registry: registry)
        monkeypatch.setattr("data.pretok.cli.iter_tokenized_documents_for_spec", lambda spec, tokenizer, limit=None: iter([[1, 2, 3], [4]]))

        exit_code = main([
            "build",
            "--output-dir", str(tmp_path / "out"),
            "--tokenizer", "fake-tokenizer",
            "--dataset", "fineweb_edu",
            "--stage", "4",
            "--num-workers", "1",
        ])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "pretok build start" in out.lower()
        assert "fineweb_edu" in out
        assert "stage=4" in out.lower()
        assert "pretok build complete" in out.lower()

    def test_tqdm_is_forced_on_for_build_loops(self, tmp_path, monkeypatch):
        _install_fake_tokenizer(monkeypatch)
        from data.dataset_registry import DatasetRegistry, DatasetSpec
        from data.pretok import cli

        registry = DatasetRegistry({
            "fineweb_edu": DatasetSpec(name="fineweb_edu", hf_path="fake/fineweb", subset="sample-10BT"),
        })
        monkeypatch.setattr("data.pretok.cli.build_pretok_registry", lambda _registry: registry)
        monkeypatch.setattr("data.pretok.cli.iter_tokenized_documents_for_spec", lambda spec, tokenizer, limit=None: iter([[1, 2, 3], [4]]))

        seen: list[dict[str, object]] = []
        original_tqdm = cli._tqdm

        def wrapped_tqdm(iterable, **kwargs):
            seen.append(dict(kwargs))
            return original_tqdm(iterable, **kwargs)

        monkeypatch.setattr(cli, "_tqdm", wrapped_tqdm)

        exit_code = cli.main([
            "build",
            "--output-dir", str(tmp_path / "out"),
            "--tokenizer", "fake-tokenizer",
            "--dataset", "fineweb_edu",
            "--stage", "4",
            "--num-workers", "1",
        ])

        assert exit_code == 0
        assert seen
        assert all(call.get("disable") is False for call in seen)
        assert {call.get("desc") for call in seen} >= {"datasets", "tokenize:fineweb_edu", "pack:fineweb_edu"}
