from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AblationResult:
    variant_name: str
    preset_name: str
    eval_loss: float
    tool_benchmark_score: float
    general_benchmark_score: float
    training_tokens: int


class AblationTracker:
    def __init__(self):
        self.results: list[AblationResult] = []

    def log_result(self, result: AblationResult) -> None:
        self.results.append(result)

    def compare_variants(self, preset_name: str, baseline_variant: str = "baseline_gdn") -> dict[str, dict[str, float]]:
        preset_results = [result for result in self.results if result.preset_name == preset_name]
        baseline = self._find_result(preset_results, baseline_variant)
        comparison: dict[str, dict[str, float]] = {}
        for result in preset_results:
            if result.variant_name == baseline_variant:
                continue
            comparison[result.variant_name] = {
                "eval_loss_delta": round(result.eval_loss - baseline.eval_loss, 12),
                "tool_benchmark_score_delta": round(result.tool_benchmark_score - baseline.tool_benchmark_score, 12),
                "general_benchmark_score_delta": round(result.general_benchmark_score - baseline.general_benchmark_score, 12),
                "training_tokens_delta": result.training_tokens - baseline.training_tokens,
            }
        return comparison

    def detect_regressions(self, baseline_variant: str = "baseline_gdn", threshold: float = 0.05) -> list[str]:
        if threshold < 0:
            raise ValueError("threshold must be nonnegative")
        regressions: list[str] = []
        presets = sorted({result.preset_name for result in self.results})
        for preset in presets:
            preset_results = [result for result in self.results if result.preset_name == preset]
            baseline = self._find_result(preset_results, baseline_variant)
            for result in preset_results:
                if result.variant_name == baseline_variant:
                    continue
                if self._relative_increase(result.eval_loss, baseline.eval_loss) > threshold:
                    regressions.append(f"{preset}/{result.variant_name}: eval_loss regression")
                if self._relative_drop(result.tool_benchmark_score, baseline.tool_benchmark_score) > threshold:
                    regressions.append(f"{preset}/{result.variant_name}: tool_benchmark_score regression")
                if self._relative_drop(result.general_benchmark_score, baseline.general_benchmark_score) > threshold:
                    regressions.append(f"{preset}/{result.variant_name}: general_benchmark_score regression")
        return regressions

    def _find_result(self, results: list[AblationResult], variant_name: str) -> AblationResult:
        for result in results:
            if result.variant_name == variant_name:
                return result
        raise ValueError(f"missing baseline variant: {variant_name}")

    def _relative_increase(self, value: float, baseline: float) -> float:
        return (value - baseline) / max(abs(baseline), 1e-12)

    def _relative_drop(self, value: float, baseline: float) -> float:
        return (baseline - value) / max(abs(baseline), 1e-12)
