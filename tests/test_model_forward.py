import pytest
import torch

from valkmodel import ValkModelConfig, ValkModelForCausalLM


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


def test_causal_lm_forward_returns_finite_shifted_loss_and_logits():
    torch.manual_seed(0)
    model = ValkModelForCausalLM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))

    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs.logits.shape == (2, 7, model.config.vocab_size)
    assert outputs.loss.shape == ()
    assert torch.isfinite(outputs.logits).all()
    assert torch.isfinite(outputs.loss)


def test_causal_lm_backward_produces_finite_gradients():
    torch.manual_seed(0)
    model = ValkModelForCausalLM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))

    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()

    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_causal_lm_rejects_input_ids_and_inputs_embeds_together():
    model = ValkModelForCausalLM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
    inputs_embeds = model.model.embeddings(input_ids)

    try:
        model(input_ids=input_ids, inputs_embeds=inputs_embeds)
    except ValueError as exc:
        assert "both input_ids and inputs_embeds" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_causal_lm_loss_ignores_pad_tokens():
    torch.manual_seed(0)
    config = tiny_config(pad_token_id=0)
    model = ValkModelForCausalLM(config)
    input_ids = torch.tensor([[5, 6, 7, 8], [5, 6, 7, 8]])
    labels = input_ids.clone()
    labels[1, 2:] = config.pad_token_id

    outputs = model(input_ids=input_ids, labels=labels)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    expected = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, config.vocab_size),
        shift_labels.view(-1),
        ignore_index=config.pad_token_id,
    )

    assert torch.allclose(outputs.loss, expected)


def test_config_rejects_unimplemented_naive_gdn_backend():
    try:
        tiny_config(gdn_backend="naive")
    except ValueError as exc:
        assert "gdn_backend must be 'auto' or 'fla'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_config_attn_mode_reaches_gated_deltanet_layers_with_fla_backend():
    model = ValkModelForCausalLM(tiny_config(attn_mode="fused_recurrent", gdn_backend="fla", require_fla=True))

    assert all(layer.attn.mode == "fused_recurrent" for layer in model.model.layers)
    assert all(layer.attn.backend == "fla" for layer in model.model.layers)


def test_fla_backend_requires_installed_dependency_when_requested():
    config = tiny_config(gdn_backend="fla", require_fla=True)

    try:
        model = ValkModelForCausalLM(config)
    except ImportError as exc:
        assert "flash-linear-attention" in str(exc) or "fla" in str(exc)
    else:
        assert all(layer.attn.backend == "fla" for layer in model.model.layers)


def test_jepa_loss_is_added_only_in_training_mode():
    torch.manual_seed(0)
    config = tiny_config(
        use_latent_state=True,
        latent_state_dim=32,
        latent_state_layers=[0],
        use_jepa=True,
        jepa_hidden_dim=16,
        jepa_min_horizon=1,
        jepa_max_horizon=2,
        jepa_loss_weight=0.25,
    )
    model = ValkModelForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    model.train()
    train_outputs = model(input_ids=input_ids, labels=input_ids)
    train_outputs.loss.backward()

    assert train_outputs.jepa_loss is not None
    assert train_outputs.jepa_metrics is not None
    assert train_outputs.loss > train_outputs.jepa_loss * config.jepa_loss_weight
    assert model.jepa_module.context_encoder.weight.grad is not None
    assert all(child.weight.grad is not None for child in model.jepa_module.predictor if isinstance(child, torch.nn.Linear))
    assert model.jepa_module.target_encoder.weight.grad is None

    model.eval()
    with torch.no_grad():
        eval_outputs = model(input_ids=input_ids, labels=input_ids)

    assert eval_outputs.jepa_loss is None
    assert eval_outputs.jepa_metrics is None


def test_jepa_target_encoder_updates_only_when_explicitly_requested():
    torch.manual_seed(0)
    config = tiny_config(
        use_latent_state=True,
        latent_state_dim=32,
        latent_state_layers=[0],
        use_jepa=True,
        jepa_hidden_dim=16,
        jepa_ema_momentum=0.95,
    )
    model = ValkModelForCausalLM(config)
    original = model.jepa_module.target_encoder.weight.detach().clone()
    with torch.no_grad():
        model.jepa_module.context_encoder.weight.add_(1.0)

    assert torch.allclose(model.jepa_module.target_encoder.weight, original)
    model.jepa_module.update_target_encoder()

    expected = 0.95 * original + 0.05 * model.jepa_module.context_encoder.weight.detach()
    assert torch.allclose(model.jepa_module.target_encoder.weight, expected)


def test_branch_entropy_regularization_reduces_training_loss_only():
    torch.manual_seed(0)
    base_config = tiny_config(
        use_latent_state=True,
        latent_state_dim=32,
        latent_state_layers=[0],
        use_latent_branching=True,
        enable_unstable_latent_branching=True,
        latent_branching_layers=[0],
        num_branches=3,
        branch_diversity_weight=0.0,
        branch_entropy_weight=0.0,
    )
    entropy_config = tiny_config(
        use_latent_state=True,
        latent_state_dim=32,
        latent_state_layers=[0],
        use_latent_branching=True,
        enable_unstable_latent_branching=True,
        latent_branching_layers=[0],
        num_branches=3,
        branch_diversity_weight=0.0,
        branch_entropy_weight=0.25,
    )
    base_model = ValkModelForCausalLM(base_config)
    entropy_model = ValkModelForCausalLM(entropy_config)
    entropy_model.load_state_dict(base_model.state_dict())
    input_ids = torch.randint(0, base_config.vocab_size, (2, 6))

    base_model.train()
    entropy_model.train()
    base_outputs = base_model(input_ids=input_ids, labels=input_ids)
    entropy_outputs = entropy_model(input_ids=input_ids, labels=input_ids)
    branch_entropy = torch.stack([metrics["branch_entropy"] for metrics in entropy_outputs.branch_metrics]).mean()

    assert entropy_outputs.loss.item() == pytest.approx((base_outputs.loss - 0.25 * branch_entropy).item())

    base_model.eval()
    entropy_model.eval()
    with torch.no_grad():
        base_eval = base_model(input_ids=input_ids, labels=input_ids)
        entropy_eval = entropy_model(input_ids=input_ids, labels=input_ids)

    assert entropy_eval.loss.item() == pytest.approx(base_eval.loss.item())


def test_latent_branching_metrics_and_loss_are_training_only():
    torch.manual_seed(0)
    config = tiny_config(
        use_latent_state=True,
        latent_state_dim=32,
        latent_state_layers=[0],
        use_latent_branching=True,
        enable_unstable_latent_branching=True,
        latent_branching_layers=[0],
        num_branches=3,
        branch_diversity_weight=0.5,
    )
    model = ValkModelForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    model.train()
    train_outputs = model(input_ids=input_ids, labels=input_ids)
    train_outputs.loss.backward()

    assert train_outputs.branch_metrics is not None
    assert len(train_outputs.branch_metrics) == 1
    assert train_outputs.branch_metrics[0]["branch_probs"].shape == (2, 6, 3)
    assert train_outputs.branch_metrics[0]["diversity_loss"] >= 0
    assert model.model.layers[0].latent_branching.branch_projections[0].weight.grad is not None

    model.eval()
    with torch.no_grad():
        eval_outputs = model(input_ids=input_ids, labels=input_ids)

    assert eval_outputs.branch_metrics is not None
    assert eval_outputs.loss < train_outputs.loss
