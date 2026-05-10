import sys
import types

import pytest

from valkmodel.training import ddp_launcher


def test_ddp_launcher_uses_ddp_only_for_multi_cuda_without_cpu_device(monkeypatch):
    monkeypatch.setattr(ddp_launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ddp_launcher.torch.cuda, "device_count", lambda: 2)

    assert ddp_launcher.should_use_ddp(device=None) == (True, 2)
    assert ddp_launcher.should_use_ddp(device="cuda") == (True, 2)
    assert ddp_launcher.should_use_ddp(device="cpu") == (False, 2)


def test_ddp_launcher_does_not_use_ddp_for_single_gpu_or_cpu(monkeypatch):
    monkeypatch.setattr(ddp_launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ddp_launcher.torch.cuda, "device_count", lambda: 1)

    assert ddp_launcher.should_use_ddp(device=None) == (False, 1)

    monkeypatch.setattr(ddp_launcher.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ddp_launcher.torch.cuda, "device_count", lambda: 0)

    assert ddp_launcher.should_use_ddp(device=None) == (False, 0)


def test_ddp_launcher_detects_torchrun_environment(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert ddp_launcher.is_ddp_environment() is False
    assert ddp_launcher.get_rank() == 0
    assert ddp_launcher.get_world_size() == 1
    assert ddp_launcher.get_local_rank() == 0

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert ddp_launcher.is_ddp_environment() is True
    assert ddp_launcher.get_rank() == 3
    assert ddp_launcher.get_world_size() == 8
    assert ddp_launcher.get_local_rank() == 1


def test_ddp_launcher_relaunches_with_torchrun_without_shell(monkeypatch):
    calls = {}

    def fake_run(command):
        calls["command"] = command
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr(ddp_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["valkmodel-train", "--run-preset", "130m_probe", "--output-dir", "out"])

    with pytest.raises(SystemExit) as exit_info:
        ddp_launcher.relaunch_with_torchrun(4)

    assert exit_info.value.code == 7
    assert calls["command"][:6] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "4",
        "-m",
    ]
    assert calls["command"][6:] == ["valkmodel.training.train_cli", "--run-preset", "130m_probe", "--output-dir", "out"]
