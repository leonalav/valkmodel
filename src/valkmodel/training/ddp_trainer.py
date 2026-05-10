from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .ddp_launcher import get_local_rank, get_rank, get_world_size
from .trainer import ValkTrainer


class DDPValkTrainer(ValkTrainer):
    def __init__(self, *args: Any, initialize_distributed: bool = True, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        if initialize_distributed and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=self._distributed_backend())
        if torch.cuda.is_available() and self.device.type != "cpu":
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.model.to(self.device)
        device_ids = [self.local_rank] if self.device.type == "cuda" else None
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids)
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                sampler=DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True),
                collate_fn=self._collate_batch,
            )
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.args.batch_size,
                sampler=DistributedSampler(self.eval_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False),
                collate_fn=self._collate_batch,
            )

    @classmethod
    def from_trainer(cls, trainer: ValkTrainer) -> "DDPValkTrainer":
        return cls(
            model=trainer.model,
            train_dataset=trainer.train_dataset,
            args=trainer.args,
            eval_dataset=trainer.eval_dataset,
            train_dataloader=trainer.train_dataloader if trainer.train_dataset is None else None,
            eval_dataloader=trainer.eval_dataloader if trainer.eval_dataset is None else None,
        )

    def train(self) -> dict[str, float]:
        try:
            return super().train()
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    def save_checkpoint(self, path: str) -> None:
        if not self.is_rank_zero():
            return
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.unwrap_model().state_dict(), checkpoint_path / "model.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), checkpoint_path / "scheduler.pt")
        rng_state = torch.get_rng_state().tolist()
        state = {"global_step": self.global_step, "rng_state": rng_state}
        if self.curriculum is not None:
            state["curriculum_stage"] = self.curriculum.get_stage_info(self.global_step)
        with (checkpoint_path / "trainer_state.json").open("w", encoding="utf-8") as handle:
            import json

            json.dump(state, handle)
        with (checkpoint_path / "config.json").open("w", encoding="utf-8") as handle:
            import json

            json.dump(self.unwrap_model().config.to_dict(), handle, indent=2)

    def _log_step(self, loss: float, grad_norm: float, tokens_per_sec: float, outputs: Any) -> None:
        if self.is_rank_zero():
            super()._log_step(loss, grad_norm, tokens_per_sec, outputs)

    def get_auxiliary_loss_weights(self, step: int) -> dict[str, float]:
        config = self.unwrap_model().config
        return {
            "tool": config.tool_loss_weight if self.args.tool_loss_weight is None else self.args.tool_loss_weight,
            "jepa": self._linear_warmup(0.0, config.jepa_loss_weight, step, self.args.jepa_warmup_steps),
            "branch": self._linear_warmup(0.0, config.branch_diversity_weight, step, self.args.branch_warmup_steps),
            "branch_entropy": self._linear_warmup(0.0, config.branch_entropy_weight, step, self.args.branch_warmup_steps),
        }

    def unwrap_model(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def _distributed_backend(self) -> str:
        if sys.platform == "win32" or not torch.cuda.is_available():
            return "gloo"
        return "nccl"
