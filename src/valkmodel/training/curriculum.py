from __future__ import annotations


class ContextCurriculum:
    def __init__(self, stages: list[int] | None = None, steps_per_stage: int = 1000):
        stages = [8192, 32768, 131072, 272000, 512000, 1000000] if stages is None else stages
        if not stages or any(stage <= 0 for stage in stages):
            raise ValueError("stages must contain positive context lengths")
        if steps_per_stage <= 0:
            raise ValueError("steps_per_stage must be positive")
        self.stages = list(stages)
        self.steps_per_stage = steps_per_stage

    def _stage_index(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be nonnegative")
        return min(step // self.steps_per_stage, len(self.stages) - 1)

    def get_current_context_length(self, step: int) -> int:
        return self.stages[self._stage_index(step)]

    def should_validate(self, step: int) -> bool:
        if step < 0:
            raise ValueError("step must be nonnegative")
        return step % self.steps_per_stage == 0

    def get_stage_info(self, step: int) -> dict[str, int]:
        stage_index = self._stage_index(step)
        stage_start_step = stage_index * self.steps_per_stage
        return {
            "stage_index": stage_index,
            "context_length": self.stages[stage_index],
            "stage_start_step": stage_start_step,
            "stage_end_step": stage_start_step + self.steps_per_stage,
        }
