from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .curriculum import ContextCurriculum
from .profiling import TrainingProfiler


@dataclass
class TrainingArguments:
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    num_training_steps: int = 100_000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_steps: int = 1000
    save_steps: int = 5000
    log_steps: int = 10
    seed: int = 42
    bf16: bool = False
    device: str | None = None
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    jepa_warmup_steps: int = 10_000
    branch_warmup_steps: int = 10_000
    tool_loss_weight: float | None = None
    jepa_ema_update_every: int = 1
    use_curriculum: bool = False
    curriculum_stages: list[int] | None = None
    curriculum_steps_per_stage: int = 1000
    use_streaming: bool = True
    packed_shard_root: str | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be nonnegative and learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.warmup_steps < 0 or self.num_training_steps <= 0:
            raise ValueError("warmup_steps must be nonnegative and num_training_steps must be positive")
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if self.eval_steps <= 0 or self.save_steps <= 0 or self.log_steps <= 0:
            raise ValueError("eval_steps, save_steps, and log_steps must be positive")
        if self.jepa_warmup_steps < 0 or self.branch_warmup_steps < 0:
            raise ValueError("auxiliary warmup steps must be nonnegative")
        if self.jepa_ema_update_every <= 0:
            raise ValueError("jepa_ema_update_every must be positive")


class ValkTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset | None,
        args: TrainingArguments | None = None,
        eval_dataset: Dataset | None = None,
        train_dataloader: DataLoader | None = None,
        eval_dataloader: DataLoader | None = None,
        dataloader_factory: Any | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = TrainingArguments() if args is None else args
        torch.manual_seed(self.args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("ValkTrainer requires CUDA because ValkModel only supports gdn_backend='fla'")
        self.device = torch.device(self.args.device or "cuda")
        if self.device.type != "cuda":
            raise ValueError("ValkTrainer requires a CUDA device")
        self.model.to(self.device)
        self.dataloader_factory = dataloader_factory
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        elif train_dataset is not None:
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self._collate_batch)
        else:
            raise ValueError("train_dataset or train_dataloader is required")
        if eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader
        elif eval_dataset is not None:
            self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=self._collate_batch)
        else:
            self.eval_dataloader = None
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.profiler = TrainingProfiler(max_grad_norm=self.args.max_grad_norm)
        self.curriculum = (
            ContextCurriculum(self.args.curriculum_stages, self.args.curriculum_steps_per_stage)
            if self.args.use_curriculum
            else None
        )
        self.global_step = 0
        self.last_grad_norm = 0.0
        self.last_metrics: dict[str, float] = {}
        Path(self.args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.args.log_dir).mkdir(parents=True, exist_ok=True)

    def train(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_iter = iter(self.train_dataloader)
        last_loss = 0.0
        _prev_stage_index: int = (
            self.curriculum._stage_index(self.global_step) if self.curriculum is not None else -1
        )
        while self.global_step < self.args.num_training_steps:
            step_started = time.perf_counter()
            accumulated_loss = 0.0
            token_count = 0
            latest_outputs = None
            autocast_ctx = (
                torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.args.bf16 and self.device.type == "cuda"
                else torch.amp.autocast(device_type=self.device.type, enabled=False)
            )
            accumulated_loss_tensor = torch.zeros((), device=self.device)
            for _ in range(self.args.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                batch = self._move_batch_to_device(batch)
                token_count += int(batch["input_ids"].numel())
                with autocast_ctx:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        labels=batch.get("labels"),
                        attention_mask=batch.get("attention_mask"),
                        training_lambdas=self.get_auxiliary_loss_weights(self.global_step),
                    )
                latest_outputs = outputs
                loss = outputs.loss / self.args.gradient_accumulation_steps
                loss.backward()
                accumulated_loss_tensor = accumulated_loss_tensor + loss.detach()
            accumulated_loss = float(accumulated_loss_tensor.item())
            self.last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm).item()
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            if self.model.training and getattr(self.model, "jepa_module", None) is not None and self.global_step % self.args.jepa_ema_update_every == 0:
                self.model.jepa_module.update_target_encoder()
            # Curriculum dataloader rebuild on stage transition
            if self.curriculum is not None and self.dataloader_factory is not None:
                new_stage_index = self.curriculum._stage_index(self.global_step)
                if new_stage_index != _prev_stage_index:
                    new_block_size = self.curriculum.get_current_context_length(self.global_step)
                    self.train_dataloader = self.dataloader_factory(new_block_size)
                    data_iter = iter(self.train_dataloader)
                    _prev_stage_index = new_stage_index
            elapsed = max(time.perf_counter() - step_started, 1e-12)
            last_loss = accumulated_loss
            self._log_step(last_loss, self.last_grad_norm, token_count / elapsed, latest_outputs)
            if self.eval_dataloader is not None and self.global_step % self.args.eval_steps == 0:
                self.last_metrics.update(self.evaluate())
            if self.global_step % self.args.save_steps == 0:
                self.save_checkpoint(str(Path(self.args.checkpoint_dir) / f"step_{self.global_step}"))
        return {"train_loss": last_loss, "global_step": self.global_step, **self.last_metrics}

    def evaluate(self) -> dict[str, float]:
        if self.eval_dataloader is None:
            raise ValueError("eval_dataset is required for evaluation")
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._move_batch_to_device(batch)
                outputs = self.model(input_ids=batch["input_ids"], labels=batch.get("labels"), attention_mask=batch.get("attention_mask"))
                labels = batch.get("labels")
                active_tokens = int(labels.ne(-100).sum().item()) if labels is not None else int(batch["input_ids"].numel())
                total_loss += float(outputs.loss.detach().cpu()) * active_tokens
                total_tokens += active_tokens
        if was_training:
            self.model.train()
        eval_loss = total_loss / max(total_tokens, 1)
        return {"eval_loss": eval_loss, "eval_perplexity": math.exp(min(eval_loss, 20.0))}

    def get_auxiliary_loss_weights(self, step: int) -> dict[str, float]:
        config = self.model.config
        weights = {
            "tool": config.tool_loss_weight if self.args.tool_loss_weight is None else self.args.tool_loss_weight,
            "jepa": self._linear_warmup(0.0, config.jepa_loss_weight, step, self.args.jepa_warmup_steps),
            "branch": self._linear_warmup(0.0, config.branch_diversity_weight, step, self.args.branch_warmup_steps),
            "branch_entropy": self._linear_warmup(0.0, config.branch_entropy_weight, step, self.args.branch_warmup_steps),
        }
        return weights

    def save_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path / "model.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), checkpoint_path / "scheduler.pt")
        rng_state = torch.get_rng_state().tolist()
        state = {"global_step": self.global_step, "rng_state": rng_state}
        if self.curriculum is not None:
            state["curriculum_stage"] = self.curriculum.get_stage_info(self.global_step)
        with (checkpoint_path / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(state, handle)
        with (checkpoint_path / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.model.config.to_dict(), handle, indent=2)

    def load_checkpoint(self, path: str) -> int:
        checkpoint_path = Path(path)
        self.model.load_state_dict(torch.load(checkpoint_path / "model.pt", map_location=self.device))
        self.optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt", map_location=self.device))
        self.scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt", map_location=self.device))
        with (checkpoint_path / "trainer_state.json").open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        self.global_step = int(state["global_step"])
        torch.set_rng_state(torch.tensor(state["rng_state"], dtype=torch.uint8))
        return self.global_step

    def _create_optimizer(self) -> torch.optim.AdamW:
        decay_params = []
        no_decay_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim < 2 or "bias" in name or "norm" in name.lower() or "embeddings" in name:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(step: int) -> float:
            if self.args.warmup_steps > 0 and step < self.args.warmup_steps:
                return max(step, 1) / self.args.warmup_steps
            progress = (step - self.args.warmup_steps) / max(1, self.args.num_training_steps - self.args.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            min_ratio = self.args.min_learning_rate / self.args.learning_rate
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _log_step(self, loss: float, grad_norm: float, tokens_per_sec: float, outputs: Any) -> None:
        latent_state = getattr(outputs, "latent_state", None)
        latent_norm = float(latent_state.norm().detach().cpu()) if latent_state is not None else 0.0
        latent_variance = float(latent_state.var(unbiased=False).detach().cpu()) if latent_state is not None else None
        memory_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0
        context_length = (
            self.curriculum.get_current_context_length(self.global_step)
            if self.curriculum is not None
            else int(getattr(outputs, "logits").shape[1])
        )
        learning_rate = float(self.scheduler.get_last_lr()[0])
        perplexity = math.exp(min(float(loss), 20.0))
        weights = self.get_auxiliary_loss_weights(self.global_step)
        jepa_metrics = getattr(outputs, "jepa_metrics", None) or {}
        jepa_loss = self._to_float(getattr(outputs, "jepa_loss", None))
        branch_metrics = self._aggregate_branch_metrics(getattr(outputs, "branch_metrics", None))
        metrics = {
            "loss": loss,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "tokens_per_sec": tokens_per_sec,
            "memory_mb": memory_mb,
            "context_length": context_length,
            "latent_state_norm": latent_norm,
            "latent_state_variance": latent_variance,
            "tool_weight": weights["tool"],
            "jepa_weight": weights["jepa"],
            "branch_weight": weights["branch"],
            "branch_entropy_weight": weights["branch_entropy"],
            "jepa_loss": jepa_loss,
            "jepa_prediction_variance": self._to_float(jepa_metrics.get("prediction_variance")),
            "jepa_target_variance": self._to_float(jepa_metrics.get("target_variance")),
            "jepa_cosine_mean": self._to_float(jepa_metrics.get("cosine_mean")),
            **branch_metrics,
        }
        self.profiler.log_step(
            step=self.global_step,
            loss=loss,
            grad_norm=grad_norm,
            latent_state_norm=latent_norm,
            memory_mb=memory_mb,
            tokens_per_sec=tokens_per_sec,
            context_length=context_length,
            latent_state_variance=latent_variance,
            learning_rate=learning_rate,
            perplexity=perplexity,
            tool_weight=weights["tool"],
            jepa_weight=weights["jepa"],
            branch_weight=weights["branch"],
            branch_entropy_weight=weights["branch_entropy"],
            jepa_loss=jepa_loss,
            jepa_prediction_variance=metrics["jepa_prediction_variance"],
            jepa_target_variance=metrics["jepa_target_variance"],
            jepa_cosine_mean=metrics["jepa_cosine_mean"],
            branch_entropy_mean=metrics.get("branch_entropy_mean"),
            branch_diversity_loss_mean=metrics.get("branch_diversity_loss_mean"),
            branch_variance_mean=metrics.get("branch_variance_mean"),
        )
        self.last_metrics = metrics
        if self.global_step % self.args.log_steps == 0:
            print(self._format_metrics(metrics))

    def _aggregate_branch_metrics(self, branch_metrics: Any) -> dict[str, float | None]:
        if not branch_metrics:
            return {"branch_entropy_mean": None, "branch_diversity_loss_mean": None, "branch_variance_mean": None}
        return {
            "branch_entropy_mean": self._mean_metric(branch_metrics, "branch_entropy"),
            "branch_diversity_loss_mean": self._mean_metric(branch_metrics, "diversity_loss"),
            "branch_variance_mean": self._mean_metric(branch_metrics, "branch_variance"),
        }

    def _mean_metric(self, metrics: Any, key: str) -> float | None:
        values = [self._to_float(metric.get(key)) for metric in metrics if isinstance(metric, dict) and metric.get(key) is not None]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    def _format_metrics(self, metrics: dict[str, Any]) -> str:
        fields = [
            f"step={self.global_step}",
            f"loss={metrics['loss']:.4f}",
            f"ppl={metrics['perplexity']:.2f}",
            f"lr={metrics['learning_rate']:.3e}",
            f"grad={metrics['grad_norm']:.4f}",
            f"tok/s={metrics['tokens_per_sec']:.1f}",
            f"mem={metrics['memory_mb']:.1f}MB",
            f"ctx={metrics['context_length']}",
            f"tool_w={metrics['tool_weight']:.4f}",
            f"jepa_w={metrics['jepa_weight']:.4f}",
            f"branch_w={metrics['branch_weight']:.4f}",
            f"branch_entropy_w={metrics['branch_entropy_weight']:.4f}",
        ]
        for key in (
            "jepa_loss",
            "jepa_prediction_variance",
            "jepa_target_variance",
            "jepa_cosine_mean",
            "branch_entropy_mean",
            "branch_diversity_loss_mean",
            "branch_variance_mean",
        ):
            value = metrics.get(key)
            if value is not None:
                fields.append(f"{key}={value:.4f}")
        return " | ".join(fields)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _collate_batch(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        keys = examples[0].keys()
        batch: dict[str, Any] = {}
        for key in keys:
            values = [example[key] for example in examples]
            if torch.is_tensor(values[0]):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values
        return batch

    def _linear_warmup(self, start: float, end: float, step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0 or step >= warmup_steps:
            return end
        return start + (end - start) * (step / warmup_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.args)
