import types

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from valkmodel import ValkModelConfig, ValkModelForCausalLM
from valkmodel.training import TrainingArguments, ValkTrainer


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


def tiny_config(**overrides):
    values = {
        "vocab_size": 97,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_heads": 3,
        "head_dim": 16,
        "num_v_heads": 3,
        "intermediate_size": 128,
        "max_position_embeddings": 128,
        "use_short_conv": True,
    }
    values.update(overrides)
    return ValkModelConfig(**values)


def test_trainer_accepts_injected_dataloaders(tmp_path):
    model = ValkModelForCausalLM(tiny_config()).cuda()
    args = TrainingArguments(num_training_steps=1, batch_size=2, checkpoint_dir=str(tmp_path), device="cuda")
    train_loader = DataLoader(TinyTokenDataset(), batch_size=2, collate_fn=lambda examples: {key: torch.stack([example[key] for example in examples]) for key in examples[0]})

    trainer = ValkTrainer(model=model, train_dataset=None, args=args, train_dataloader=train_loader)

    assert trainer.train_dataset is None
    assert trainer.train_dataloader is train_loader


def test_trainer_initializes_optimizer_groups_and_scheduler(tmp_path):
    model = ValkModelForCausalLM(tiny_config()).cuda()
    args = TrainingArguments(num_training_steps=4, batch_size=2, warmup_steps=2, checkpoint_dir=str(tmp_path), device="cuda")
    trainer = ValkTrainer(model=model, train_dataset=TinyTokenDataset(), args=args)

    assert len(trainer.optimizer.param_groups) == 2
    assert trainer.optimizer.param_groups[0]["weight_decay"] == args.weight_decay
    assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0
    assert trainer.scheduler is not None


def test_auxiliary_loss_warmup_schedules_reach_targets(tmp_path):
    model = ValkModelForCausalLM(tiny_config(use_latent_state=True, use_jepa=True, latent_state_layers=[0], use_latent_branching=True, enable_unstable_latent_branching=True, latent_branching_layers=[0])).cuda()
    args = TrainingArguments(
        num_training_steps=4,
        batch_size=2,
        jepa_warmup_steps=4,
        branch_warmup_steps=2,
        checkpoint_dir=str(tmp_path),
        device="cuda",
    )
    trainer = ValkTrainer(model=model, train_dataset=TinyTokenDataset(), args=args)

    step0 = trainer.get_auxiliary_loss_weights(0)
    step2 = trainer.get_auxiliary_loss_weights(2)
    step4 = trainer.get_auxiliary_loss_weights(4)

    assert step0["jepa"] == 0.0
    assert step0["branch"] == 0.0
    assert step0["branch_entropy"] == 0.0
    assert step2["jepa"] == model.config.jepa_loss_weight * 0.5
    assert step2["branch"] == model.config.branch_diversity_weight
    assert step2["branch_entropy"] == model.config.branch_entropy_weight
    assert step4["jepa"] == model.config.jepa_loss_weight


def test_trainer_runs_steps_logs_metrics_and_updates_parameters(tmp_path):
    torch.manual_seed(0)
    model = ValkModelForCausalLM(tiny_config()).cuda()
    before = model.lm_head.weight.detach().clone()
    args = TrainingArguments(num_training_steps=2, batch_size=2, warmup_steps=1, log_steps=1, checkpoint_dir=str(tmp_path), device="cuda")
    trainer = ValkTrainer(model=model, train_dataset=TinyTokenDataset(), args=args)

    metrics = trainer.train()

    assert trainer.global_step == 2
    assert torch.isfinite(torch.tensor(metrics["train_loss"]))
    assert trainer.profiler.records
    assert not torch.allclose(before, model.lm_head.weight.detach())


def test_trainer_clips_gradients_and_records_grad_norm(tmp_path):
    model = ValkModelForCausalLM(tiny_config()).cuda()
    args = TrainingArguments(num_training_steps=1, batch_size=2, max_grad_norm=0.01, checkpoint_dir=str(tmp_path), device="cuda")
    trainer = ValkTrainer(model=model, train_dataset=TinyTokenDataset(), args=args)

    trainer.train()

    assert trainer.last_grad_norm >= 0
    clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    assert clipped_norm <= args.max_grad_norm + 1e-5


def test_trainer_updates_jepa_target_encoder_after_optimizer_step(tmp_path):
    torch.manual_seed(0)
    config = tiny_config(use_latent_state=True, latent_state_layers=[0], use_jepa=True, jepa_hidden_dim=16)
    model = ValkModelForCausalLM(config).cuda()
    before = model.jepa_module.target_encoder.weight.detach().clone()
    args = TrainingArguments(num_training_steps=1, batch_size=2, checkpoint_dir=str(tmp_path), device="cuda")
    trainer = ValkTrainer(model=model, train_dataset=TinyTokenDataset(), args=args)

    trainer.train()

    assert model.jepa_module.target_encoder.weight.grad is None
    assert not torch.allclose(before, model.jepa_module.target_encoder.weight.detach())


def test_trainer_evaluation_loop_returns_finite_metrics_and_restores_train_mode(tmp_path):
    model = ValkModelForCausalLM(tiny_config()).cuda()
    args = TrainingArguments(num_training_steps=1, batch_size=2, checkpoint_dir=str(tmp_path), device="cuda")
    trainer = ValkTrainer(model=model, train_dataset=TinyTokenDataset(), eval_dataset=TinyTokenDataset(length=2), args=args)

    metrics = trainer.evaluate()

    assert model.training
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
    assert torch.isfinite(torch.tensor(metrics["eval_perplexity"]))


def test_trainer_log_step_records_health_metrics_and_prints_on_log_boundary(tmp_path, capsys):
    model = ValkModelForCausalLM(tiny_config(use_latent_state=True, latent_state_layers=[0], use_jepa=True, use_latent_branching=True, enable_unstable_latent_branching=True, latent_branching_layers=[0]))
    args = TrainingArguments(num_training_steps=1, batch_size=2, log_steps=1, checkpoint_dir=str(tmp_path), device="cuda", jepa_warmup_steps=0, branch_warmup_steps=0)
    trainer = ValkTrainer(model=model, train_dataset=TinyTokenDataset(), args=args)
    trainer.global_step = 1
    outputs = types.SimpleNamespace(
        logits=torch.zeros(2, 6, model.config.vocab_size),
        latent_state=torch.ones(2, 6, model.config.latent_state_dim),
        jepa_loss=torch.tensor(0.25),
        jepa_metrics={"prediction_variance": 0.3, "target_variance": 0.4, "cosine_mean": 0.5},
        branch_metrics=(
            {"branch_entropy": torch.tensor(0.6), "diversity_loss": torch.tensor(0.7), "branch_variance": torch.tensor(0.8)},
            {"branch_entropy": torch.tensor(1.0), "diversity_loss": torch.tensor(0.9), "branch_variance": torch.tensor(1.2)},
        ),
    )

    trainer._log_step(2.0, 0.5, 123.0, outputs)

    captured = capsys.readouterr().out
    assert "step=1" in captured
    assert "jepa_loss=0.2500" in captured
    assert trainer.last_metrics["learning_rate"] == trainer.scheduler.get_last_lr()[0]
    assert trainer.last_metrics["perplexity"] > 0
    assert trainer.last_metrics["jepa_prediction_variance"] == 0.3
    assert trainer.last_metrics["branch_entropy_weight"] == model.config.branch_entropy_weight
    assert "branch_entropy_w=" in captured
    assert trainer.last_metrics["branch_entropy_mean"] == pytest.approx(0.8)
    assert trainer.last_metrics["branch_diversity_loss_mean"] == pytest.approx(0.8)
    assert trainer.last_metrics["branch_variance_mean"] == pytest.approx(1.0)


def test_trainer_checkpoint_save_and_resume_restores_step(tmp_path):
    model = ValkModelForCausalLM(tiny_config()).cuda()
    args = TrainingArguments(num_training_steps=1, batch_size=2, save_steps=1, checkpoint_dir=str(tmp_path), device="cuda")
    trainer = ValkTrainer(model=model, train_dataset=TinyTokenDataset(), args=args)
    trainer.train()

    checkpoint = tmp_path / "step_1"
    resumed_model = ValkModelForCausalLM(tiny_config())
    resumed = ValkTrainer(model=resumed_model, train_dataset=TinyTokenDataset(), args=args)
    step = resumed.load_checkpoint(str(checkpoint))

    assert step == 1
    assert resumed.global_step == 1
