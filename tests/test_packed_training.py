"""
Task 3 TDD tests: --use-streaming / --no-use-streaming flag, dataloader routing,
curriculum-triggered dataloader rebuild, and packed batch field contract.

RED phase: all tests written before any production code changes.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Helpers shared across test groups
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class OneBatchDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        return {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}


def tiny_collate(examples):
    keys = examples[0].keys()
    return {key: torch.stack([example[key] for example in examples]) for key in keys}


def write_yaml(path, content):
    path.write_text(content, encoding="utf-8")


def write_json(path, payload):
    import json
    path.write_text(json.dumps(payload), encoding="utf-8")


def install_fake_data_modules(monkeypatch, *, use_streaming=True, packed_shard_root=None):
    """
    Install lightweight fake data/tokenizer modules and return a call-capture dict.
    The fake build_training_dataloader records which mode was requested.
    """
    calls = {}
    data_module = types.ModuleType("data")
    dataloader_builder = types.ModuleType("data.dataloader_builder")
    tokenizer_setup = types.ModuleType("data.tokenizer_setup")

    def fake_load_tokenizer(tokenizer_name_or_path, use_fast=True, **kwargs):
        calls["tokenizer"] = tokenizer_name_or_path
        tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)
        tok.__len__ = lambda self: 32
        return tok

    def fake_build_training_dataloader(
        mixture_config_path,
        tokenizer,
        block_size,
        batch_size,
        use_streaming=True,
        packed_shard_root=None,
        **kwargs,
    ):
        calls["dataloader"] = {
            "mixture_config_path": mixture_config_path,
            "block_size": block_size,
            "batch_size": batch_size,
            "use_streaming": use_streaming,
            "packed_shard_root": packed_shard_root,
        }
        return DataLoader(OneBatchDataset(), batch_size=batch_size, collate_fn=tiny_collate)

    tokenizer_setup.load_tokenizer = fake_load_tokenizer
    dataloader_builder.build_training_dataloader = fake_build_training_dataloader
    data_module.tokenizer_setup = tokenizer_setup
    data_module.dataloader_builder = dataloader_builder
    monkeypatch.setitem(sys.modules, "data", data_module)
    monkeypatch.setitem(sys.modules, "data.tokenizer_setup", tokenizer_setup)
    monkeypatch.setitem(sys.modules, "data.dataloader_builder", dataloader_builder)
    return calls


# ---------------------------------------------------------------------------
# Group 1: CLI flag parsing
# ---------------------------------------------------------------------------

class TestCLIFlagParsing:
    """--use-streaming / --no-use-streaming and --packed-shard-root parsing."""

    def test_use_streaming_defaults_to_true(self):
        from valkmodel.training.train_cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--preset", "130m", "--train-data", "t.json", "--output-dir", "out"])

        assert args.use_streaming is True

    def test_no_use_streaming_sets_flag_false(self):
        from valkmodel.training.train_cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([
            "--preset", "130m",
            "--train-data", "t.json",
            "--output-dir", "out",
            "--no-use-streaming",
        ])

        assert args.use_streaming is False

    def test_packed_shard_root_defaults_to_none(self):
        from valkmodel.training.train_cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--preset", "130m", "--train-data", "t.json", "--output-dir", "out"])

        assert args.packed_shard_root is None

    def test_packed_shard_root_accepts_path_string(self):
        from valkmodel.training.train_cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([
            "--preset", "130m",
            "--train-data", "t.json",
            "--output-dir", "out",
            "--packed-shard-root", "/data/shards",
        ])

        assert args.packed_shard_root == "/data/shards"

    def test_use_streaming_flag_is_stored_in_training_args(self):
        """TrainingArguments must carry use_streaming and packed_shard_root."""
        from valkmodel.training.trainer import TrainingArguments

        ta = TrainingArguments(use_streaming=False, packed_shard_root="/shards")

        assert ta.use_streaming is False
        assert ta.packed_shard_root == "/shards"

    def test_training_args_use_streaming_defaults_to_true(self):
        from valkmodel.training.trainer import TrainingArguments

        ta = TrainingArguments()

        assert ta.use_streaming is True
        assert ta.packed_shard_root is None


# ---------------------------------------------------------------------------
# Group 2: Dataloader routing (streaming vs packed)
# ---------------------------------------------------------------------------

class TestDataloaderRouting:
    """build_dataloader_from_config must pass use_streaming and packed_shard_root through."""

    def _make_config_files(self, tmp_path):
        model_path = tmp_path / "model.json"
        training_path = tmp_path / "training.yaml"
        data_path = tmp_path / "data.yaml"
        tokenizer_path = tmp_path / "tokenizer.yaml"
        write_json(model_path, {
            "model_type": "valkmodel",
            "hidden_size": 32, "num_hidden_layers": 1,
            "num_heads": 3, "head_dim": 8, "num_v_heads": 3,
            "intermediate_size": 64, "vocab_size": 32,
            "gdn_backend": "fla",
        })
        write_yaml(training_path, "training:\n  max_steps: 1\n  learning_rate: 0.001\n")
        write_yaml(data_path, "mixture:\n  creative_writing: 1.0\n")
        write_yaml(tokenizer_path, "tokenizer_name_or_path: fake\nuse_fast: true\n")
        return model_path, training_path, data_path, tokenizer_path

    def test_streaming_mode_is_default_when_no_flag_given(self, tmp_path, monkeypatch):
        calls = install_fake_data_modules(monkeypatch)
        model_path, training_path, data_path, tokenizer_path = self._make_config_files(tmp_path)
        from valkmodel.training.train_cli import build_arg_parser, build_trainer_from_args

        args = build_arg_parser().parse_args([
            "--training-config", str(training_path),
            "--data-config", str(data_path),
            "--tokenizer-config", str(tokenizer_path),
            "--model-config", str(model_path),
            "--output-dir", str(tmp_path / "out"),
            "--batch-size", "1", "--seq-len", "4", "--device", "cpu",
        ])
        build_trainer_from_args(args)

        assert calls["dataloader"]["use_streaming"] is True
        assert calls["dataloader"]["packed_shard_root"] is None

    def test_no_use_streaming_routes_to_packed_mode(self, tmp_path, monkeypatch):
        calls = install_fake_data_modules(monkeypatch)
        model_path, training_path, data_path, tokenizer_path = self._make_config_files(tmp_path)
        from valkmodel.training.train_cli import build_arg_parser, build_trainer_from_args

        args = build_arg_parser().parse_args([
            "--training-config", str(training_path),
            "--data-config", str(data_path),
            "--tokenizer-config", str(tokenizer_path),
            "--model-config", str(model_path),
            "--output-dir", str(tmp_path / "out"),
            "--batch-size", "1", "--seq-len", "4", "--device", "cpu",
            "--no-use-streaming",
            "--packed-shard-root", "/data/shards",
        ])
        build_trainer_from_args(args)

        assert calls["dataloader"]["use_streaming"] is False
        assert calls["dataloader"]["packed_shard_root"] == Path("A:/data/shards")

    def test_build_training_dataloader_streaming_path_uses_mixture_dataset(self, tmp_path):
        """Streaming path must produce a DataLoader backed by StreamingMixtureDataset."""
        import yaml
        from data.dataloader_builder import build_training_dataloader
        from data.mixture_dataset import StreamingMixtureDataset

        data_path = tmp_path / "data.yaml"
        data_path.write_text("mixture:\n  creative_writing: 1.0\n", encoding="utf-8")
        tok = types.SimpleNamespace(
            pad_token_id=0, eos_token_id=2, bos_token_id=1,
            encode=lambda text, add_special_tokens=False: [1, 2, 3],
        )

        loader = build_training_dataloader(
            data_path, tok, block_size=4, batch_size=1, use_streaming=True
        )

        assert isinstance(loader, DataLoader)
        assert isinstance(loader.dataset, StreamingMixtureDataset)

    def test_build_training_dataloader_packed_path_uses_shard_dataset(self, tmp_path):
        """Packed path must produce a DataLoader backed by ShardDataset."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from data.dataloader_builder import build_training_dataloader
        from data.pretok.shard_dataset import ShardDataset

        shard_dir = tmp_path / "shards" / "stage_4"
        shard_dir.mkdir(parents=True)
        # Write a minimal parquet shard with input_ids column
        block_size = 4
        rows = {"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]]}
        table = pa.table(rows)
        pq.write_table(table, shard_dir / "shard_0000.parquet")

        tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)
        data_path = tmp_path / "data.yaml"
        data_path.write_text("mixture:\n  creative_writing: 1.0\n", encoding="utf-8")

        loader = build_training_dataloader(
            data_path, tok, block_size=block_size, batch_size=1,
            use_streaming=False,
            packed_shard_root=tmp_path / "shards",
        )

        assert isinstance(loader, DataLoader)
        assert isinstance(loader.dataset, ShardDataset)


# ---------------------------------------------------------------------------
# Group 3: Curriculum-triggered dataloader rebuild
# ---------------------------------------------------------------------------

class TestCurriculumDataloaderRebuild:
    """When curriculum stage changes context length, trainer must rebuild its dataloader."""

    def _make_tiny_trainer(self, tmp_path, stages, steps_per_stage, *, use_streaming=True, packed_shard_root=None):
        from valkmodel import ValkModelConfig, ValkModelForCausalLM
        from valkmodel.training import TrainingArguments, ValkTrainer

        config = ValkModelConfig(
            vocab_size=32, hidden_size=32, num_hidden_layers=1,
            num_heads=2, head_dim=8, num_v_heads=2,
            intermediate_size=64, max_position_embeddings=256,
            use_short_conv=True, use_gate=False,
        )
        model = ValkModelForCausalLM(config)
        args = TrainingArguments(
            num_training_steps=steps_per_stage * len(stages),
            batch_size=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
            use_curriculum=True,
            curriculum_stages=stages,
            curriculum_steps_per_stage=steps_per_stage,
            use_streaming=use_streaming,
            packed_shard_root=packed_shard_root,
        )
        return model, args

    def test_trainer_exposes_dataloader_factory_callable(self, tmp_path):
        """Trainer must accept a dataloader_factory kwarg for curriculum rebuilds."""
        from valkmodel import ValkModelConfig, ValkModelForCausalLM
        from valkmodel.training import TrainingArguments, ValkTrainer

        config = ValkModelConfig(
            vocab_size=32, hidden_size=32, num_hidden_layers=1,
            num_heads=2, head_dim=8, num_v_heads=2,
            intermediate_size=64, max_position_embeddings=256,
            use_short_conv=True, use_gate=False,
        )
        model = ValkModelForCausalLM(config)
        args = TrainingArguments(
            num_training_steps=1, batch_size=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )
        factory_calls = []

        def fake_factory(block_size: int) -> DataLoader:
            factory_calls.append(block_size)
            ds = OneBatchDataset()
            return DataLoader(ds, batch_size=1, collate_fn=tiny_collate)

        loader = fake_factory(4)
        trainer = ValkTrainer(
            model=model, train_dataset=None, args=args,
            train_dataloader=loader,
            dataloader_factory=fake_factory,
        )

        assert trainer.dataloader_factory is fake_factory

    def test_trainer_rebuilds_dataloader_on_stage_transition(self, tmp_path):
        """
        When curriculum advances to a new stage, the trainer must call
        dataloader_factory(new_block_size) and replace self.train_dataloader.
        """
        from valkmodel import ValkModelConfig, ValkModelForCausalLM
        from valkmodel.training import TrainingArguments, ValkTrainer

        stages = [4, 8]
        steps_per_stage = 2
        config = ValkModelConfig(
            vocab_size=32, hidden_size=32, num_hidden_layers=1,
            num_heads=2, head_dim=8, num_v_heads=2,
            intermediate_size=64, max_position_embeddings=256,
            use_short_conv=True, use_gate=False,
        )
        model = ValkModelForCausalLM(config)
        args = TrainingArguments(
            num_training_steps=steps_per_stage * len(stages),
            batch_size=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
            use_curriculum=True,
            curriculum_stages=stages,
            curriculum_steps_per_stage=steps_per_stage,
        )
        factory_calls: list[int] = []

        def fake_factory(block_size: int) -> DataLoader:
            factory_calls.append(block_size)
            # Return a dataset whose items match the requested block_size
            class FixedSeqDataset(Dataset):
                def __len__(self): return 4
                def __getitem__(self, i):
                    ids = torch.ones(block_size, dtype=torch.long)
                    return {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}

            return DataLoader(FixedSeqDataset(), batch_size=1, collate_fn=tiny_collate)

        initial_loader = fake_factory(stages[0])
        factory_calls.clear()  # reset — only count rebuilds during train()

        trainer = ValkTrainer(
            model=model, train_dataset=None, args=args,
            train_dataloader=initial_loader,
            dataloader_factory=fake_factory,
        )
        trainer.train()

        # Factory must have been called at least once for the stage-2 block_size
        assert stages[1] in factory_calls, (
            f"Expected factory called with block_size={stages[1]}, got calls={factory_calls}"
        )

    def test_trainer_does_not_rebuild_dataloader_without_factory(self, tmp_path):
        """Without a factory, curriculum still advances but no rebuild occurs (no crash)."""
        from valkmodel import ValkModelConfig, ValkModelForCausalLM
        from valkmodel.training import TrainingArguments, ValkTrainer

        stages = [4, 8]
        steps_per_stage = 2
        config = ValkModelConfig(
            vocab_size=32, hidden_size=32, num_hidden_layers=1,
            num_heads=2, head_dim=8, num_v_heads=2,
            intermediate_size=64, max_position_embeddings=256,
            use_short_conv=True, use_gate=False,
        )
        model = ValkModelForCausalLM(config)
        args = TrainingArguments(
            num_training_steps=steps_per_stage * len(stages),
            batch_size=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
            use_curriculum=True,
            curriculum_stages=stages,
            curriculum_steps_per_stage=steps_per_stage,
        )

        class FixedSeqDataset(Dataset):
            def __len__(self): return 8
            def __getitem__(self, i):
                ids = torch.ones(4, dtype=torch.long)
                return {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}

        loader = DataLoader(FixedSeqDataset(), batch_size=1, collate_fn=tiny_collate)
        trainer = ValkTrainer(
            model=model, train_dataset=None, args=args,
            train_dataloader=loader,
        )

        # Must not raise
        metrics = trainer.train()
        assert "train_loss" in metrics

    def test_build_trainer_from_args_wires_dataloader_factory_in_config_first_mode(
        self, tmp_path, monkeypatch
    ):
        """
        build_trainer_from_args must attach a dataloader_factory to the trainer
        when config-first mode is used, so curriculum rebuilds work end-to-end.
        """
        calls = install_fake_data_modules(monkeypatch)
        model_path = tmp_path / "model.json"
        training_path = tmp_path / "training.yaml"
        data_path = tmp_path / "data.yaml"
        tokenizer_path = tmp_path / "tokenizer.yaml"
        write_json(model_path, {
            "model_type": "valkmodel",
            "hidden_size": 32, "num_hidden_layers": 1,
            "num_heads": 3, "head_dim": 8, "num_v_heads": 3,
            "intermediate_size": 64, "vocab_size": 32,
            "gdn_backend": "fla",
        })
        write_yaml(training_path, "training:\n  max_steps: 1\n  learning_rate: 0.001\n")
        write_yaml(data_path, "mixture:\n  creative_writing: 1.0\n")
        write_yaml(tokenizer_path, "tokenizer_name_or_path: fake\nuse_fast: true\n")
        from valkmodel.training.train_cli import build_arg_parser, build_trainer_from_args

        args = build_arg_parser().parse_args([
            "--training-config", str(training_path),
            "--data-config", str(data_path),
            "--tokenizer-config", str(tokenizer_path),
            "--model-config", str(model_path),
            "--output-dir", str(tmp_path / "out"),
            "--batch-size", "1", "--seq-len", "4", "--device", "cpu",
        ])
        trainer = build_trainer_from_args(args)

        assert trainer.dataloader_factory is not None
        assert callable(trainer.dataloader_factory)


# ---------------------------------------------------------------------------
# Group 4: Packed batch field contract
# ---------------------------------------------------------------------------

class TestPackedBatchFields:
    """ShardDataset + DataLoader must produce exactly input_ids, labels, attention_mask."""

    def _write_parquet_shard(self, shard_dir: Path, block_size: int, n_rows: int = 4):
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir.mkdir(parents=True, exist_ok=True)
        rows = {"input_ids": [list(range(i, i + block_size)) for i in range(n_rows)]}
        table = pa.table(rows)
        pq.write_table(table, shard_dir / "shard_0000.parquet")

    def test_shard_dataset_yields_required_fields(self, tmp_path):
        from data.pretok.shard_dataset import ShardDataset

        block_size = 8
        shard_dir = tmp_path / "shards" / "stage_8"
        self._write_parquet_shard(shard_dir, block_size)

        ds = ShardDataset(shard_root=tmp_path / "shards", block_size=block_size)
        item = ds[0]

        assert set(item.keys()) == {"input_ids", "labels", "attention_mask"}

    def test_shard_dataset_input_ids_are_long_tensors(self, tmp_path):
        from data.pretok.shard_dataset import ShardDataset

        block_size = 8
        shard_dir = tmp_path / "shards" / "stage_8"
        self._write_parquet_shard(shard_dir, block_size)

        ds = ShardDataset(shard_root=tmp_path / "shards", block_size=block_size)
        item = ds[0]

        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long

    def test_shard_dataset_labels_match_input_ids(self, tmp_path):
        from data.pretok.shard_dataset import ShardDataset

        block_size = 8
        shard_dir = tmp_path / "shards" / "stage_8"
        self._write_parquet_shard(shard_dir, block_size)

        ds = ShardDataset(shard_root=tmp_path / "shards", block_size=block_size)
        item = ds[0]

        assert item["labels"].tolist() == item["input_ids"].tolist()

    def test_shard_dataset_attention_mask_is_all_ones(self, tmp_path):
        from data.pretok.shard_dataset import ShardDataset

        block_size = 8
        shard_dir = tmp_path / "shards" / "stage_8"
        self._write_parquet_shard(shard_dir, block_size)

        ds = ShardDataset(shard_root=tmp_path / "shards", block_size=block_size)
        item = ds[0]

        assert item["attention_mask"].tolist() == [1] * block_size

    def test_shard_dataset_selects_correct_stage_directory(self, tmp_path):
        """ShardDataset must look in stage_{block_size}/ subdirectory."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from data.pretok.shard_dataset import ShardDataset

        block_size = 16
        wrong_dir = tmp_path / "shards" / "stage_8"
        right_dir = tmp_path / "shards" / "stage_16"
        wrong_dir.mkdir(parents=True)
        right_dir.mkdir(parents=True)

        # Only write data in the correct stage dir
        rows = {"input_ids": [list(range(16))]}
        pq.write_table(pa.table(rows), right_dir / "shard_0000.parquet")

        ds = ShardDataset(shard_root=tmp_path / "shards", block_size=block_size)

        assert len(ds) == 1
        assert ds[0]["input_ids"].tolist() == list(range(16))

    def test_packed_dataloader_batch_has_correct_shape(self, tmp_path):
        """DataLoader over ShardDataset must produce (batch, block_size) shaped tensors."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from data.pretok.shard_dataset import ShardDataset

        block_size = 8
        batch_size = 2
        shard_dir = tmp_path / "shards" / "stage_8"
        shard_dir.mkdir(parents=True)
        rows = {"input_ids": [list(range(i, i + block_size)) for i in range(4)]}
        pq.write_table(pa.table(rows), shard_dir / "shard_0000.parquet")

        ds = ShardDataset(shard_root=tmp_path / "shards", block_size=block_size)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=tiny_collate)
        batch = next(iter(loader))

        assert batch["input_ids"].shape == torch.Size([batch_size, block_size])
        assert batch["labels"].shape == torch.Size([batch_size, block_size])
        assert batch["attention_mask"].shape == torch.Size([batch_size, block_size])
