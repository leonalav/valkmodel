from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TrainingStepMetrics:
    step: int
    loss: float
    grad_norm: float
    latent_state_norm: float
    memory_mb: float
    tokens_per_sec: float
    context_length: int
    latent_state_variance: float | None = None
    learning_rate: float | None = None
    perplexity: float | None = None
    tool_weight: float | None = None
    jepa_weight: float | None = None
    branch_weight: float | None = None
    jepa_loss: float | None = None
    jepa_prediction_variance: float | None = None
    jepa_target_variance: float | None = None
    jepa_cosine_mean: float | None = None
    branch_entropy_mean: float | None = None
    branch_diversity_loss_mean: float | None = None
    branch_variance_mean: float | None = None


class TrainingProfiler:
    def __init__(
        self,
        max_grad_norm: float = 100.0,
        min_latent_variance: float = 0.0,
        min_tokens_per_sec: float = 0.0,
    ):
        self.max_grad_norm = max_grad_norm
        self.min_latent_variance = min_latent_variance
        self.min_tokens_per_sec = min_tokens_per_sec
        self.records: list[TrainingStepMetrics] = []

    def log_step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        latent_state_norm: float,
        memory_mb: float,
        tokens_per_sec: float,
        context_length: int,
        latent_state_variance: float | None = None,
        learning_rate: float | None = None,
        perplexity: float | None = None,
        tool_weight: float | None = None,
        jepa_weight: float | None = None,
        branch_weight: float | None = None,
        jepa_loss: float | None = None,
        jepa_prediction_variance: float | None = None,
        jepa_target_variance: float | None = None,
        jepa_cosine_mean: float | None = None,
        branch_entropy_mean: float | None = None,
        branch_diversity_loss_mean: float | None = None,
        branch_variance_mean: float | None = None,
    ) -> None:
        self.records.append(
            TrainingStepMetrics(
                step=step,
                loss=float(loss),
                grad_norm=float(grad_norm),
                latent_state_norm=float(latent_state_norm),
                memory_mb=float(memory_mb),
                tokens_per_sec=float(tokens_per_sec),
                context_length=int(context_length),
                latent_state_variance=self._optional_float(latent_state_variance),
                learning_rate=self._optional_float(learning_rate),
                perplexity=self._optional_float(perplexity),
                tool_weight=self._optional_float(tool_weight),
                jepa_weight=self._optional_float(jepa_weight),
                branch_weight=self._optional_float(branch_weight),
                jepa_loss=self._optional_float(jepa_loss),
                jepa_prediction_variance=self._optional_float(jepa_prediction_variance),
                jepa_target_variance=self._optional_float(jepa_target_variance),
                jepa_cosine_mean=self._optional_float(jepa_cosine_mean),
                branch_entropy_mean=self._optional_float(branch_entropy_mean),
                branch_diversity_loss_mean=self._optional_float(branch_diversity_loss_mean),
                branch_variance_mean=self._optional_float(branch_variance_mean),
            )
        )

    def summarize_by_context_length(self) -> dict[int, dict[str, float | int]]:
        grouped: dict[int, list[TrainingStepMetrics]] = {}
        for record in self.records:
            grouped.setdefault(record.context_length, []).append(record)

        summary: dict[int, dict[str, float | int]] = {}
        for context_length, records in grouped.items():
            count = len(records)
            values: dict[str, float | int] = {
                "steps": count,
                "loss_mean": sum(record.loss for record in records) / count,
                "grad_norm_mean": sum(record.grad_norm for record in records) / count,
                "latent_state_norm_mean": sum(record.latent_state_norm for record in records) / count,
                "memory_mb_peak": max(record.memory_mb for record in records),
                "tokens_per_sec_mean": sum(record.tokens_per_sec for record in records) / count,
            }
            for field_name in (
                "learning_rate",
                "perplexity",
                "tool_weight",
                "jepa_weight",
                "branch_weight",
                "jepa_loss",
                "jepa_prediction_variance",
                "jepa_target_variance",
                "jepa_cosine_mean",
                "branch_entropy_mean",
                "branch_diversity_loss_mean",
                "branch_variance_mean",
            ):
                present = [getattr(record, field_name) for record in records if getattr(record, field_name) is not None]
                if present:
                    summary_key = field_name if field_name.endswith("_mean") else f"{field_name}_mean"
                    values[summary_key] = sum(present) / len(present)
            summary[context_length] = values
        return summary

    def detect_anomalies(self) -> list[str]:
        anomalies: list[str] = []
        for record in self.records:
            if not math.isfinite(record.loss):
                anomalies.append(f"step {record.step}: loss is not finite")
            if not math.isfinite(record.grad_norm) or record.grad_norm > self.max_grad_norm:
                anomalies.append(f"step {record.step}: grad_norm is unstable")
            if record.latent_state_variance is not None and record.latent_state_variance <= self.min_latent_variance:
                anomalies.append(f"step {record.step}: latent variance is too low")
            if record.tokens_per_sec < self.min_tokens_per_sec:
                anomalies.append(f"step {record.step}: throughput is too low")
            if record.perplexity is not None and not math.isfinite(record.perplexity):
                anomalies.append(f"step {record.step}: perplexity is not finite")
            if record.jepa_prediction_variance is not None and record.jepa_prediction_variance <= self.min_latent_variance:
                anomalies.append(f"step {record.step}: JEPA prediction variance is too low")
            if record.jepa_target_variance is not None and record.jepa_target_variance <= self.min_latent_variance:
                anomalies.append(f"step {record.step}: JEPA target variance is too low")
            if record.branch_entropy_mean is not None and record.branch_entropy_mean <= 1e-12:
                anomalies.append(f"step {record.step}: branch entropy is too low")
        return anomalies

    def _optional_float(self, value: float | None) -> float | None:
        return None if value is None else float(value)
