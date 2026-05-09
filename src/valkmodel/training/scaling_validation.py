from __future__ import annotations

from dataclasses import dataclass

from ..configuration_valkmodel import ValkModelConfig


DEFAULT_GPU_MEMORY_GB = {"rtx_5090": 32.0, "l40": 48.0, "h100": 80.0}
DEFAULT_GPU_PEAK_TFLOPS = {"rtx_5090": 100.0, "l40": 90.5, "h100": 989.0}


@dataclass
class HardwareProfile:
    preset_name: str
    num_parameters: int
    context_length: int
    batch_size: int
    memory_gb: float
    tokens_per_sec: float
    mfu_estimate: float
    gpu_type: str
    eval_loss: float | None = None


class ScalingValidator:
    def __init__(
        self,
        gpu_memory_gb: dict[str, float] | None = None,
        gpu_peak_tflops: dict[str, float] | None = None,
        safety_margin: float = 1.2,
    ):
        if safety_margin <= 0:
            raise ValueError("safety_margin must be positive")
        self.gpu_memory_gb = dict(DEFAULT_GPU_MEMORY_GB if gpu_memory_gb is None else gpu_memory_gb)
        self.gpu_peak_tflops = dict(DEFAULT_GPU_PEAK_TFLOPS if gpu_peak_tflops is None else gpu_peak_tflops)
        self.safety_margin = safety_margin

    def estimate_training_memory(
        self,
        config: ValkModelConfig,
        batch_size: int,
        seq_len: int,
        dtype_bytes: int = 2,
        activation_multiplier: float = 4.0,
    ) -> dict[str, float]:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        num_parameters = config.estimate_parameters()
        params_gb = num_parameters * dtype_bytes / 1024**3
        gradients_gb = num_parameters * dtype_bytes / 1024**3
        master_weights_gb = num_parameters * 4 / 1024**3
        optimizer_gb = 2 * num_parameters * 4 / 1024**3
        activations_gb = (
            batch_size
            * seq_len
            * config.hidden_size
            * config.num_hidden_layers
            * activation_multiplier
            * dtype_bytes
            / 1024**3
        )
        cache_gb = self.estimate_gdn_cache_memory(config, batch_size, dtype_bytes)
        branch_gb = 0.0
        if config.use_latent_branching:
            branching_layers = config.latent_branching_layers if config.latent_branching_layers is not None else [config.num_hidden_layers // 2]
            branch_gb = (
                len(branching_layers)
                * batch_size
                * seq_len
                * config.num_branches
                * config.latent_state_dim
                * dtype_bytes
                / 1024**3
            )
        total_gb = self.safety_margin * (params_gb + gradients_gb + master_weights_gb + optimizer_gb + activations_gb + cache_gb + branch_gb)
        return {
            "params_gb": params_gb,
            "gradients_gb": gradients_gb,
            "master_weights_gb": master_weights_gb,
            "optimizer_gb": optimizer_gb,
            "activations_gb": activations_gb,
            "gdn_cache_gb": cache_gb,
            "branch_gb": branch_gb,
            "total_gb": total_gb,
        }

    def estimate_gdn_cache_memory(self, config: ValkModelConfig, batch_size: int, dtype_bytes: int = 2) -> float:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return config.num_hidden_layers * batch_size * config.num_v_heads * config.head_v_dim * dtype_bytes / 1024**3

    def estimate_training_flops(self, config: ValkModelConfig, tokens: int) -> int:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        return 6 * config.estimate_parameters() * tokens

    def compute_mfu(self, num_parameters: int, tokens_per_sec: float, gpu_type: str) -> float:
        if num_parameters <= 0 or tokens_per_sec <= 0:
            raise ValueError("num_parameters and tokens_per_sec must be positive")
        if gpu_type not in self.gpu_peak_tflops:
            raise ValueError(f"unknown gpu_type: {gpu_type}")
        actual_flops_per_sec = 6 * num_parameters * tokens_per_sec
        peak_flops_per_sec = self.gpu_peak_tflops[gpu_type] * 1_000_000_000_000
        return actual_flops_per_sec / peak_flops_per_sec

    def validate_scaling_curve(self, profiles: list[HardwareProfile]) -> dict[str, bool]:
        ordered = sorted(profiles, key=lambda profile: profile.num_parameters)
        losses = [profile.eval_loss for profile in ordered if profile.eval_loss is not None]
        return {"eval_loss_monotonic": all(left >= right for left, right in zip(losses, losses[1:]))}

    def recommend_training_config(
        self,
        config: ValkModelConfig,
        context_length: int,
        gpu_type: str,
        dtype_bytes: int = 2,
    ) -> dict[str, float | int | bool]:
        if gpu_type not in self.gpu_memory_gb:
            raise ValueError(f"unknown gpu_type: {gpu_type}")
        memory_budget = self.gpu_memory_gb[gpu_type]
        max_batch_size = 0
        for batch_size in range(1, 1025):
            estimate = self.estimate_training_memory(config, batch_size, context_length, dtype_bytes)
            if estimate["total_gb"] <= memory_budget:
                max_batch_size = batch_size
            else:
                break
        recommended_batch = max(1, max_batch_size)
        memory_estimate = self.estimate_training_memory(config, recommended_batch, context_length, dtype_bytes)["total_gb"]
        return {"fits": max_batch_size > 0, "max_batch_size": max_batch_size, "estimated_memory_gb": memory_estimate}
