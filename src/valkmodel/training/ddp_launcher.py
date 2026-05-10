from __future__ import annotations

import os
import subprocess
import sys

import torch


def should_use_ddp(device: str | None = None) -> tuple[bool, int]:
    if device == "cpu" or not torch.cuda.is_available():
        return False, 0 if not torch.cuda.is_available() else torch.cuda.device_count()
    num_gpus = torch.cuda.device_count()
    return num_gpus > 1, num_gpus


def is_ddp_environment() -> bool:
    return all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def relaunch_with_torchrun(num_gpus: int) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(num_gpus),
        "-m",
        "valkmodel.training.train_cli",
        *sys.argv[1:],
    ]
    result = subprocess.run(command)
    raise SystemExit(result.returncode)
