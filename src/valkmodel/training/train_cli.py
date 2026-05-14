from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

import torch
from torch.utils.data import Dataset

from ..configuration_valkmodel import ValkModelConfig
from ..modeling_valkmodel import ValkModelForCausalLM
from .ddp_launcher import get_rank, is_ddp_environment, relaunch_with_torchrun, should_use_ddp
from .ddp_trainer import DDPValkTrainer
from .trainer import TrainingArguments, ValkTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"
TRAINING_PRESETS = {
    "130m_probe": {
        "training": CONFIG_ROOT / "training_130m_probe.yaml",
        "data": CONFIG_ROOT / "data_mix_fast.yaml",
        "tokenizer": CONFIG_ROOT / "tokenizer_llama3.yaml",
        "model": CONFIG_ROOT / "valkmodel_tiny_130m.json",
    },
    "1b_main": {
        "training": CONFIG_ROOT / "training_1b_main.yaml",
        "data": CONFIG_ROOT / "data_mix_30b.yaml",
        "tokenizer": CONFIG_ROOT / "tokenizer_llama3.yaml",
        "model": CONFIG_ROOT / "valkmodel_base_1b.json",
    },
}
MODEL_OVERRIDE_KEYS = ("hidden_size", "num_hidden_layers", "num_heads", "head_dim", "num_v_heads", "intermediate_size", "vocab_size")
TRAINING_ALIASES = {"max_steps": "num_training_steps", "grad_clip": "max_grad_norm"}


class TokenListDataset(Dataset):
    def __init__(self, documents: list[list[int]], seq_len: int, pad_token_id: int = 0):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if not documents:
            raise ValueError("at least one token document is required")
        self.examples = [self._build_example(document, seq_len, pad_token_id) for document in documents]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]

    def _build_example(self, tokens: list[int], seq_len: int, pad_token_id: int) -> dict[str, torch.Tensor]:
        if not tokens:
            raise ValueError("token documents must not be empty")
        clipped = tokens[:seq_len]
        padded = clipped + [pad_token_id] * (seq_len - len(clipped))
        labels = padded.copy()
        if len(clipped) < seq_len:
            labels[len(clipped) :] = [-100] * (seq_len - len(clipped))
        input_ids = torch.tensor(padded, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": (input_ids != pad_token_id).long(),
        }


def ensure_project_data_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ValkModel from repo configs or local tokenized smoke-test data")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--preset", choices=["130m", "260m", "520m", "780m", "1.2b", "2.8b", "5b", "8b"])
    config_group.add_argument("--config-file")
    config_group.add_argument("--model-config")
    parser.add_argument("--run-preset", choices=sorted(TRAINING_PRESETS))
    parser.add_argument("--training-config")
    parser.add_argument("--data-config")
    parser.add_argument("--tokenizer-config")
    parser.add_argument("--train-data")
    parser.add_argument("--eval-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--jepa-warmup-steps", type=int, default=10_000)
    parser.add_argument("--branch-warmup-steps", type=int, default=10_000)
    parser.add_argument("--tool-loss-weight", type=float)
    parser.add_argument("--jepa-ema-update-every", type=int, default=1)
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--curriculum-stages", type=int, nargs="+")
    parser.add_argument("--curriculum-steps-per-stage", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-hidden-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--num-v-heads", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--vocab-size", type=int)
    return parser


def resolve_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_training_preset(name: str) -> dict[str, Path]:
    if name not in TRAINING_PRESETS:
        raise ValueError(f"unknown training preset: {name}")
    return {key: resolve_path(value) for key, value in TRAINING_PRESETS[name].items()}


def load_training_config(path: str | Path, model_config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> tuple[ValkModelConfig, TrainingArguments, dict[str, Any]]:
    payload = load_yaml_config(path)
    yaml_model_payload = dict(payload.get("model", {}))
    model_payload = dict(yaml_model_payload)
    overridden_yaml_model_fields: list[str] = []
    if model_config_path is not None:
        with resolve_path(model_config_path).open("r", encoding="utf-8") as handle:
            json_model_payload = json.load(handle)
        overridden_yaml_model_fields = sorted(key for key in yaml_model_payload if key in json_model_payload and yaml_model_payload[key] != json_model_payload[key])
        model_payload.update(json_model_payload)
    if overrides:
        model_payload.update({key: value for key, value in overrides.items() if value is not None})
    config, ignored_model = _build_model_config(model_payload)
    training_args, ignored_training = _build_training_args(payload.get("training", {}))
    ignored_fields = sorted(set(ignored_model + overridden_yaml_model_fields))
    return config, training_args, {"ignored_config_fields": [f"model.{field}" for field in ignored_fields] + [f"training.{field}" for field in ignored_training]}


def load_tokenizer_from_config(path: str | Path):
    ensure_project_data_import_path()
    from data.tokenizer_setup import load_tokenizer

    payload = load_yaml_config(path)
    tokenizer_name_or_path = payload.pop("tokenizer_name_or_path")
    return load_tokenizer(tokenizer_name_or_path, **payload)


def build_dataloader_from_config(path: str | Path, tokenizer: Any, block_size: int, batch_size: int):
    ensure_project_data_import_path()
    from data.dataloader_builder import build_training_dataloader

    return build_training_dataloader(resolve_path(path), tokenizer, block_size=block_size, batch_size=batch_size)


def load_token_documents(path: str) -> list[list[int]]:
    data_path = resolve_path(path)
    if data_path.suffix == ".jsonl":
        documents = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    documents.append(_extract_tokens(json.loads(stripped)))
        return documents
    with data_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, dict):
        return [_extract_tokens(loaded)]
    if isinstance(loaded, list):
        if not loaded:
            return []
        if all(isinstance(item, int) for item in loaded):
            return [[int(item) for item in loaded]]
        return [_extract_tokens(item) for item in loaded]
    raise ValueError("token data must be a JSON list, JSON object, or JSONL records")


def build_trainer_from_args(args: argparse.Namespace) -> ValkTrainer:
    paths = _resolve_config_paths(args)
    output_dir = resolve_path(args.output_dir)
    if paths is not None:
        model_overrides = _model_overrides_from_args(args)
        config, training_args, metadata = load_training_config(paths["training"], paths.get("model"), model_overrides)
        _apply_cli_training_overrides(training_args, args)
        _set_output_dirs(training_args, output_dir)
        tokenizer = load_tokenizer_from_config(paths["tokenizer"])
        train_dataloader = build_dataloader_from_config(paths["data"], tokenizer, block_size=args.seq_len, batch_size=training_args.batch_size)
        model = ValkModelForCausalLM(config)
        backends = set(layer.attn.backend for layer in model.model.layers)
        print(f"[DEBUG] GDN backends in use: {backends}")
        assert backends == {"fla"}, f"Expected fla, got {backends}"
        trainer = ValkTrainer(model=model, train_dataset=None, args=training_args, train_dataloader=train_dataloader)
        trainer.config_metadata = metadata
        print(f"training_config={paths['training']}")
        print(f"data_config={paths['data']}")
        print(f"tokenizer_config={paths['tokenizer']}")
        if paths.get("model") is not None:
            print(f"model_config={paths['model']}")
    else:
        config = _load_config(args)
        pad_token_id = config.pad_token_id if config.pad_token_id is not None else 0
        train_dataset = TokenListDataset(load_token_documents(args.train_data), args.seq_len, pad_token_id)
        eval_dataset = TokenListDataset(load_token_documents(args.eval_data), args.seq_len, pad_token_id) if args.eval_data else None
        training_args = _training_args_from_cli(args)
        _set_output_dirs(training_args, output_dir)
        model = ValkModelForCausalLM(config)
        backends = set(layer.attn.backend for layer in model.model.layers)
        print(f"[DEBUG] GDN backends in use: {backends}")
        assert backends == {"fla"}, f"Expected fla, got {backends}"
        trainer = ValkTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, args=training_args)
    if args.resume_from:
        trainer.load_checkpoint(str(resolve_path(args.resume_from)))
    return trainer


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    use_ddp, num_gpus = should_use_ddp(args.device)
    if use_ddp and not is_ddp_environment():
        relaunch_with_torchrun(num_gpus)
    trainer = build_trainer_from_args(args)
    if is_ddp_environment():
        trainer = DDPValkTrainer.from_trainer(trainer)
    metrics = trainer.train()
    if not is_ddp_environment() or get_rank() == 0:
        print(json.dumps(metrics, sort_keys=True))
    return metrics


def _resolve_config_paths(args: argparse.Namespace) -> dict[str, Path] | None:
    if args.run_preset:
        paths = resolve_training_preset(args.run_preset)
        if args.training_config:
            paths["training"] = resolve_path(args.training_config)
        if args.data_config:
            paths["data"] = resolve_path(args.data_config)
        if args.tokenizer_config:
            paths["tokenizer"] = resolve_path(args.tokenizer_config)
        if args.model_config:
            paths["model"] = resolve_path(args.model_config)
        return paths
    if args.training_config or args.data_config or args.tokenizer_config:
        missing = [name for name in ("training_config", "data_config", "tokenizer_config") if getattr(args, name) is None]
        if missing:
            raise ValueError(f"config-first training requires: {', '.join(missing)}")
        paths = {
            "training": resolve_path(args.training_config),
            "data": resolve_path(args.data_config),
            "tokenizer": resolve_path(args.tokenizer_config),
        }
        if args.model_config:
            paths["model"] = resolve_path(args.model_config)
        return paths
    return None


def _load_config(args: argparse.Namespace) -> ValkModelConfig:
    overrides = _model_overrides_from_args(args)
    if args.config_file:
        with resolve_path(args.config_file).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        values.update(overrides)
        config, _ = _build_model_config(values)
        return config
    if args.preset is None:
        raise ValueError("fallback token-list mode requires --preset or --config-file")
    return ValkModelConfig.from_preset(args.preset, **overrides)


def _model_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key in MODEL_OVERRIDE_KEYS if (value := getattr(args, key)) is not None}


def _build_model_config(values: dict[str, Any]) -> tuple[ValkModelConfig, list[str]]:
    valid_keys = set(inspect.signature(ValkModelConfig.__init__).parameters) - {"self", "kwargs"}
    filtered = {key: value for key, value in values.items() if key in valid_keys and key != "model_type"}
    ignored = sorted(key for key in values if key not in valid_keys and key != "model_type")
    return ValkModelConfig(**filtered), ignored


def _build_training_args(values: dict[str, Any]) -> tuple[TrainingArguments, list[str]]:
    valid_keys = set(inspect.signature(TrainingArguments).parameters)
    mapped: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in values.items():
        target_key = TRAINING_ALIASES.get(key, key)
        if target_key in valid_keys:
            mapped[target_key] = value
        else:
            ignored.append(key)
    return TrainingArguments(**mapped), sorted(ignored)


def _training_args_from_cli(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_training_steps=args.num_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        seed=args.seed,
        bf16=args.bf16,
        device=args.device,
        jepa_warmup_steps=args.jepa_warmup_steps,
        branch_warmup_steps=args.branch_warmup_steps,
        tool_loss_weight=args.tool_loss_weight,
        jepa_ema_update_every=args.jepa_ema_update_every,
        use_curriculum=args.use_curriculum,
        curriculum_stages=args.curriculum_stages,
        curriculum_steps_per_stage=args.curriculum_steps_per_stage,
    )


def _apply_cli_training_overrides(training_args: TrainingArguments, args: argparse.Namespace) -> None:
    defaults = build_arg_parser().parse_args(["--preset", "130m", "--train-data", "dummy", "--output-dir", "dummy"])
    override_map = {
        "learning_rate": "learning_rate",
        "min_learning_rate": "min_learning_rate",
        "weight_decay": "weight_decay",
        "warmup_steps": "warmup_steps",
        "num_steps": "num_training_steps",
        "batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "eval_steps": "eval_steps",
        "save_steps": "save_steps",
        "log_steps": "log_steps",
        "seed": "seed",
        "bf16": "bf16",
        "device": "device",
        "jepa_warmup_steps": "jepa_warmup_steps",
        "branch_warmup_steps": "branch_warmup_steps",
        "tool_loss_weight": "tool_loss_weight",
        "jepa_ema_update_every": "jepa_ema_update_every",
        "use_curriculum": "use_curriculum",
        "curriculum_stages": "curriculum_stages",
        "curriculum_steps_per_stage": "curriculum_steps_per_stage",
    }
    for arg_name, field_name in override_map.items():
        value = getattr(args, arg_name)
        if value != getattr(defaults, arg_name):
            setattr(training_args, field_name, value)


def _set_output_dirs(training_args: TrainingArguments, output_dir: Path) -> None:
    training_args.checkpoint_dir = str(output_dir / "checkpoints")
    training_args.log_dir = str(output_dir / "logs")


def _extract_tokens(record: Any) -> list[int]:
    if isinstance(record, list) and all(isinstance(item, int) for item in record):
        return [int(item) for item in record]
    if isinstance(record, dict):
        for key in ("input_ids", "tokens", "ids"):
            value = record.get(key)
            if isinstance(value, list) and all(isinstance(item, int) for item in value):
                return [int(item) for item in value]
    raise ValueError("each token document must be a list of ints or an object with input_ids, tokens, or ids")


if __name__ == "__main__":
    main()
