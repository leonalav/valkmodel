import json
import sys
import types

import torch
from torch.utils.data import DataLoader, Dataset

import valkmodel.training.train_cli as train_cli
from valkmodel.training.train_cli import (
    build_arg_parser,
    build_trainer_from_args,
    ensure_project_data_import_path,
    load_token_documents,
    load_training_config,
    resolve_path,
    resolve_training_preset,
)


class OneBatchDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        return {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}


def tiny_collate(examples):
    keys = examples[0].keys()
    return {key: torch.stack([example[key] for example in examples]) for key in keys}


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_yaml(path, content):
    path.write_text(content, encoding="utf-8")


def install_fake_data_modules(monkeypatch):
    calls = {}
    data_module = types.ModuleType("data")
    dataloader_builder = types.ModuleType("data.dataloader_builder")
    tokenizer_setup = types.ModuleType("data.tokenizer_setup")

    def fake_load_tokenizer(tokenizer_name_or_path, use_fast=True, **kwargs):
        calls["tokenizer"] = {"tokenizer_name_or_path": tokenizer_name_or_path, "use_fast": use_fast, **kwargs}
        tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)
        tokenizer.__len__ = lambda: 32
        return tokenizer

    def fake_build_training_dataloader(mixture_config_path, tokenizer, block_size, batch_size, **kwargs):
        calls["dataloader"] = {
            "mixture_config_path": mixture_config_path,
            "tokenizer": tokenizer,
            "block_size": block_size,
            "batch_size": batch_size,
            **kwargs,
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


def test_train_cli_adds_project_root_for_top_level_data_package(monkeypatch):
    src_path = str(resolve_path("src"))
    root_path = str(resolve_path("."))
    monkeypatch.setattr(sys, "path", [src_path])

    ensure_project_data_import_path()

    assert sys.path[0] == root_path


def test_train_cli_parser_accepts_fallback_token_training_args():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--preset",
            "130m",
            "--train-data",
            "train.json",
            "--output-dir",
            "out",
            "--num-steps",
            "2",
            "--batch-size",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert args.preset == "130m"
    assert args.train_data == "train.json"
    assert args.output_dir == "out"
    assert args.num_steps == 2
    assert args.batch_size == 1
    assert args.device == "cpu"


def test_train_cli_parser_accepts_config_first_args():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--training-config",
            "configs/training_130m_probe.yaml",
            "--data-config",
            "configs/data_mix_fast.yaml",
            "--tokenizer-config",
            "configs/tokenizer_llama3.yaml",
            "--model-config",
            "configs/valkmodel_tiny_130m.json",
            "--output-dir",
            "out",
            "--device",
            "cuda",
            "--bf16",
        ]
    )

    assert args.training_config == "configs/training_130m_probe.yaml"
    assert args.data_config == "configs/data_mix_fast.yaml"
    assert args.tokenizer_config == "configs/tokenizer_llama3.yaml"
    assert args.model_config == "configs/valkmodel_tiny_130m.json"
    assert args.device == "cuda"
    assert args.bf16 is True


def test_train_cli_resolves_builtin_training_preset():
    preset = resolve_training_preset("130m_probe")

    assert preset["training"].name == "training_130m_probe.yaml"
    assert preset["data"].name == "data_mix_fast.yaml"
    assert preset["tokenizer"].name == "tokenizer_llama3.yaml"
    assert preset["model"].name == "valkmodel_tiny_130m.json"


def test_train_cli_loads_json_and_jsonl_token_documents(tmp_path):
    json_path = tmp_path / "tokens.json"
    json_path.write_text(json.dumps([[1, 2, 3], {"input_ids": [4, 5, 6]}]), encoding="utf-8")
    jsonl_path = tmp_path / "tokens.jsonl"
    jsonl_path.write_text(json.dumps({"tokens": [7, 8, 9]}) + "\n" + json.dumps([10, 11, 12]) + "\n", encoding="utf-8")

    assert load_token_documents(str(json_path)) == [[1, 2, 3], [4, 5, 6]]
    assert load_token_documents(str(jsonl_path)) == [[7, 8, 9], [10, 11, 12]]


def test_train_cli_resolves_user_and_environment_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("VALK_DATA_DIR", str(tmp_path))
    data_path = tmp_path / "tokens.json"

    assert resolve_path("$VALK_DATA_DIR/tokens.json") == data_path.resolve()


def test_training_yaml_maps_supported_fields_and_reports_ignored_fields(tmp_path):
    training_path = tmp_path / "training.yaml"
    write_yaml(
        training_path,
        """
model:
  hidden_size: 32
  num_hidden_layers: 1
  num_heads: 3
  head_dim: 8
  num_v_heads: 3
  intermediate_size: 64
  vocab_size: 32
  branch_entropy_weight: 0.003
  moba_layers: [0]
training:
  learning_rate: 0.001
  warmup_steps: 3
  max_steps: 7
  weight_decay: 0.2
  grad_clip: 0.5
  balance_loss_weight: 0.01
""",
    )

    config, training_args, metadata = load_training_config(training_path)

    assert config.hidden_size == 32
    assert config.branch_entropy_weight == 0.003
    assert training_args.learning_rate == 0.001
    assert training_args.warmup_steps == 3
    assert training_args.num_training_steps == 7
    assert training_args.weight_decay == 0.2
    assert training_args.max_grad_norm == 0.5
    assert "model.moba_layers" in metadata["ignored_config_fields"]
    assert "training.balance_loss_weight" in metadata["ignored_config_fields"]


def test_run_preset_configs_use_current_valkmodel_fields():
    legacy_fields = {
        "moba_layers",
        "mla_layers",
        "kda_layers",
        "num_experts",
        "expert_top_k",
        "expert_usage_alert_threshold",
        "balance_loss_weight",
    }

    for preset_name in ("130m_probe", "1b_main"):
        preset = resolve_training_preset(preset_name)
        config, _, metadata = load_training_config(preset["training"], preset["model"])

        assert config.num_heads * config.head_dim == int(0.75 * config.hidden_size)
        assert not any(field in metadata["ignored_config_fields"] for field in legacy_fields)


def test_train_cli_builds_config_first_trainer_with_existing_data_pipeline(tmp_path, monkeypatch):
    calls = install_fake_data_modules(monkeypatch)
    model_path = tmp_path / "model.json"
    training_path = tmp_path / "training.yaml"
    data_path = tmp_path / "data.yaml"
    tokenizer_path = tmp_path / "tokenizer.yaml"
    write_json(
        model_path,
        {
            "model_type": "valkmodel",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_heads": 3,
            "head_dim": 8,
            "num_v_heads": 3,
            "intermediate_size": 64,
            "vocab_size": 32,
            "moba_layers": [0],
        },
    )
    write_yaml(training_path, "training:\n  max_steps: 1\n  grad_clip: 0.5\n  learning_rate: 0.001\n")
    write_yaml(data_path, "mixture:\n  creative_writing: 1.0\n")
    write_yaml(tokenizer_path, "tokenizer_name_or_path: fake-tokenizer\ntrust_remote_code: false\nuse_fast: true\n")
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--training-config",
            str(training_path),
            "--data-config",
            str(data_path),
            "--tokenizer-config",
            str(tokenizer_path),
            "--model-config",
            str(model_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--batch-size",
            "1",
            "--seq-len",
            "4",
            "--device",
            "cpu",
        ]
    )

    trainer = build_trainer_from_args(args)

    assert trainer.args.num_training_steps == 1
    assert trainer.args.max_grad_norm == 0.5
    assert trainer.train_dataset is None
    assert calls["tokenizer"] == {"tokenizer_name_or_path": "fake-tokenizer", "use_fast": True, "trust_remote_code": False}
    assert calls["dataloader"]["mixture_config_path"] == data_path.resolve()
    assert calls["dataloader"]["block_size"] == 4
    assert calls["dataloader"]["batch_size"] == 1


def test_train_cli_main_relaunches_multi_gpu_runs_under_torchrun(monkeypatch):
    calls = {}

    monkeypatch.setattr(train_cli, "should_use_ddp", lambda device=None: (True, 4))
    monkeypatch.setattr(train_cli, "is_ddp_environment", lambda: False)
    def fake_relaunch(num_gpus):
        calls["num_gpus"] = num_gpus
        raise SystemExit(0)

    monkeypatch.setattr(train_cli, "relaunch_with_torchrun", fake_relaunch)

    try:
        train_cli.main(["--run-preset", "130m_probe", "--output-dir", "out"])
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["num_gpus"] == 4


def test_train_cli_main_uses_ddp_trainer_inside_torchrun(monkeypatch, tmp_path, capsys):
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([[3, 4, 5, 6]]), encoding="utf-8")
    calls = {}

    class FakeDDPTrainer:
        @classmethod
        def from_trainer(cls, trainer):
            calls["wrapped"] = trainer
            return cls()

        def train(self):
            return {"train_loss": 1.25, "global_step": 1}

    monkeypatch.setattr(train_cli, "should_use_ddp", lambda device=None: (True, 2))
    monkeypatch.setattr(train_cli, "is_ddp_environment", lambda: True)
    monkeypatch.setattr(train_cli, "get_rank", lambda: 0)
    monkeypatch.setattr(train_cli, "DDPValkTrainer", FakeDDPTrainer)

    metrics = train_cli.main(
        [
            "--preset",
            "130m",
            "--train-data",
            str(train_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-steps",
            "1",
            "--batch-size",
            "1",
            "--seq-len",
            "4",
            "--device",
            "cpu",
            "--hidden-size",
            "32",
            "--num-hidden-layers",
            "1",
            "--num-heads",
            "3",
            "--head-dim",
            "8",
            "--num-v-heads",
            "3",
            "--intermediate-size",
            "64",
            "--vocab-size",
            "32",
        ]
    )

    assert metrics == {"train_loss": 1.25, "global_step": 1}
    assert "wrapped" in calls
    assert '"global_step": 1' in capsys.readouterr().out


def test_train_cli_builds_tiny_trainer_from_args(tmp_path):
    train_path = tmp_path / "train.json"
    eval_path = tmp_path / "eval.json"
    train_path.write_text(json.dumps([[3, 4, 5, 6], [7, 8, 9, 10]]), encoding="utf-8")
    eval_path.write_text(json.dumps([[11, 12, 13, 14]]), encoding="utf-8")
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--preset",
            "130m",
            "--train-data",
            str(train_path),
            "--eval-data",
            str(eval_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-steps",
            "1",
            "--batch-size",
            "1",
            "--seq-len",
            "4",
            "--device",
            "cpu",
            "--hidden-size",
            "32",
            "--num-hidden-layers",
            "1",
            "--num-heads",
            "3",
            "--head-dim",
            "8",
            "--num-v-heads",
            "3",
            "--intermediate-size",
            "64",
            "--vocab-size",
            "32",
        ]
    )

    trainer = build_trainer_from_args(args)

    assert trainer.args.num_training_steps == 1
    assert trainer.args.batch_size == 1
    assert trainer.model.config.hidden_size == 32
    batch = next(iter(trainer.train_dataloader))
    assert batch["input_ids"].shape == torch.Size([1, 4])
    assert trainer.eval_dataloader is not None
