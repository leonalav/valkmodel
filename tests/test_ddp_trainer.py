import types

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from valkmodel import ValkModelConfig, ValkModelForCausalLM
from valkmodel.training import TrainingArguments, ValkTrainer
from valkmodel.training.ddp_trainer import DDPValkTrainer


class TinyTokenDataset(Dataset):
    def __init__(self, vocab_size=97, length=4, seq_len=6):
        self.examples = []
        for idx in range(length):
            ids = (torch.arange(seq_len) + idx + 3) % vocab_size
            self.examples.append({"input_ids": ids.long(), "labels": ids.long(), "attention_mask": torch.ones(seq_len, dtype=torch.long)})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def tiny_config():
    return ValkModelConfig(
        vocab_size=97,
        hidden_size=64,
        num_hidden_layers=1,
        num_heads=3,
        head_dim=16,
        num_v_heads=3,
        intermediate_size=128,
        max_position_embeddings=128,
        use_short_conv=True,
    )


def fake_ddp_environment(monkeypatch, rank="0", local_rank="0", world_size="2"):
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("LOCAL_RANK", local_rank)
    monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda backend: None)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)


class FakeDDP(torch.nn.Module):
    def __init__(self, module, device_ids=None):
        super().__init__()
        self.module = module
        self.device_ids = device_ids

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_ddp_trainer_converts_existing_trainer_and_wraps_model(monkeypatch, tmp_path):
    fake_ddp_environment(monkeypatch, rank="0", local_rank="1", world_size="2")
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)
    base = ValkTrainer(
        model=ValkModelForCausalLM(tiny_config()).cuda(),
        train_dataset=TinyTokenDataset(),
        args=TrainingArguments(num_training_steps=1, batch_size=2, checkpoint_dir=str(tmp_path), device="cuda"),
    )

    trainer = DDPValkTrainer.from_trainer(base)

    assert isinstance(trainer.model, FakeDDP)
    assert trainer.model.device_ids == [1]
    assert trainer.rank == 0
    assert trainer.local_rank == 1
    assert trainer.world_size == 2


def test_ddp_trainer_uses_distributed_sampler_for_map_style_dataset(monkeypatch, tmp_path):
    fake_ddp_environment(monkeypatch)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)
    base = ValkTrainer(
        model=ValkModelForCausalLM(tiny_config()).cuda(),
        train_dataset=TinyTokenDataset(length=8),
        args=TrainingArguments(num_training_steps=1, batch_size=2, checkpoint_dir=str(tmp_path), device="cuda"),
    )

    trainer = DDPValkTrainer.from_trainer(base)

    assert isinstance(trainer.train_dataloader.sampler, DistributedSampler)
    assert trainer.train_dataloader.sampler.num_replicas == 2


def test_ddp_trainer_preserves_injected_dataloader(monkeypatch, tmp_path):
    fake_ddp_environment(monkeypatch)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)
    train_loader = DataLoader(TinyTokenDataset(length=2), batch_size=1)
    base = ValkTrainer(
        model=ValkModelForCausalLM(tiny_config()).cuda(),
        train_dataset=None,
        train_dataloader=train_loader,
        args=TrainingArguments(num_training_steps=1, batch_size=1, checkpoint_dir=str(tmp_path), device="cuda"),
    )

    trainer = DDPValkTrainer.from_trainer(base)

    assert trainer.train_dataloader is train_loader


def test_ddp_trainer_rank_zero_guards_checkpoint_and_logging(monkeypatch, tmp_path, capsys):
    fake_ddp_environment(monkeypatch, rank="1", local_rank="1", world_size="2")
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)
    base = ValkTrainer(
        model=ValkModelForCausalLM(tiny_config()).cuda(),
        train_dataset=TinyTokenDataset(),
        args=TrainingArguments(num_training_steps=1, batch_size=2, log_steps=1, checkpoint_dir=str(tmp_path), device="cuda"),
    )
    trainer = DDPValkTrainer.from_trainer(base)
    trainer.global_step = 1
    outputs = types.SimpleNamespace(logits=torch.zeros(2, 6, trainer.unwrap_model().config.vocab_size, device="cuda"), latent_state=None, jepa_loss=None, jepa_metrics=None, branch_metrics=None)

    trainer.save_checkpoint(str(tmp_path / "rank_1_checkpoint"))
    trainer._log_step(2.0, 0.5, 123.0, outputs)

    assert not (tmp_path / "rank_1_checkpoint").exists()
    assert capsys.readouterr().out == ""
