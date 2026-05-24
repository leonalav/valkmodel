import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from valkmodel import ValkModelConfig, ValkModelForCausalLM
from valkmodel.layers.gated_deltanet_layer import GatedDeltaNetLayer, RMSNorm
from valkmodel.layers.latent_jepa import JEPAModule
from valkmodel.layers.latent_state import LatentStateModule
from valkmodel.utils.jepa_utils import compute_jepa_metrics


def tiny_gdn_layer(**overrides):
    values = {
        "hidden_size": 64,
        "expand_v": 2.0,
        "head_dim": 16,
        "num_heads": 3,
        "num_v_heads": 3,
        "use_gate": True,
        "use_short_conv": True,
        "conv_size": 4,
        "backend": "fla",
        "require_fla": True,
    }
    values.update(overrides)
    return GatedDeltaNetLayer(**values)


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


def test_gdn_decay_initialization_has_reference_timescale_diversity():
    torch.manual_seed(0)
    layer = tiny_gdn_layer()

    assert layer.A_log.dtype == torch.float32
    assert layer.dt_bias.dtype == torch.float32
    assert layer.A_log.detach().std() > 0
    assert layer.dt_bias.detach().std() > 0
    dt = F.softplus(layer.dt_bias.detach())
    assert torch.all(dt >= 1e-4)
    assert torch.all(dt <= 0.1 + 1e-6)


def test_gdn_gate_then_norm_is_not_old_norm_then_sigmoid_order():
    torch.manual_seed(0)
    norm = RMSNorm(4)
    o = torch.tensor([[[[100.0, 1.0, -3.0, 0.5]]]])
    g = torch.tensor([[[[-6.0, 3.0, 0.0, 2.0]]]])

    reference_order = norm(o * F.silu(g))
    old_order = norm(o) * torch.sigmoid(g)

    assert not torch.allclose(reference_order, old_order, atol=1e-4)


def test_gdn_rejects_padded_attention_mask_until_reference_unpadding_exists():
    layer = tiny_gdn_layer()
    hidden_states = torch.randn(2, 4, 64)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    with pytest.raises(ValueError, match="does not support padded attention_mask"):
        layer(hidden_states, attention_mask=attention_mask)


def test_latent_state_is_causal_sequence_recurrence_not_positionwise_update():
    torch.manual_seed(0)
    module = LatentStateModule(hidden_size=8, latent_state_dim=4, init_scale=0.2)
    hidden_a = torch.randn(1, 6, 8)
    hidden_b = hidden_a.clone()
    hidden_b[:, 0] = hidden_b[:, 0] + 3.0

    _, state_a = module(hidden_a)
    _, state_b = module(hidden_b)

    assert not torch.allclose(state_a[:, 1:], state_b[:, 1:])


def test_latent_state_accepts_boundary_state_for_chunk_continuation():
    torch.manual_seed(0)
    module = LatentStateModule(hidden_size=8, latent_state_dim=4, init_scale=0.2)
    first_chunk = torch.randn(2, 3, 8)
    second_chunk = torch.randn(2, 2, 8)
    _, first_state = module(first_chunk)

    _, continued_state = module(second_chunk, previous_state=first_state[:, -1])
    _, fresh_state = module(second_chunk)

    assert continued_state.shape == (2, 2, 4)
    assert not torch.allclose(continued_state, fresh_state)


def test_jepa_predictor_is_nonlinear_mlp():
    module = JEPAModule(latent_state_dim=4, jepa_hidden_dim=6, ema_momentum=0.95)

    assert isinstance(module.predictor, nn.Sequential)
    assert any(isinstance(child, nn.SiLU) for child in module.predictor)
    assert sum(isinstance(child, nn.Linear) for child in module.predictor) == 2


def test_jepa_target_encoder_remains_stop_gradient_with_mlp_predictor():
    torch.manual_seed(0)
    module = JEPAModule(latent_state_dim=4, jepa_hidden_dim=6, ema_momentum=0.95)
    current = torch.randn(2, 5, 4, requires_grad=True)
    future = torch.randn(2, 5, 4, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)

    loss, _ = module(current, future, mask)
    loss.backward()

    assert module.context_encoder.weight.grad is not None
    assert all(child.weight.grad is not None for child in module.predictor if isinstance(child, nn.Linear))
    assert module.target_encoder.weight.grad is None


def test_jepa_metrics_report_collapse_flags():
    predictions = torch.ones(2, 4, 3)
    targets = torch.ones(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)

    metrics = compute_jepa_metrics(predictions, targets, mask)

    assert metrics["prediction_collapsed"]
    assert metrics["target_collapsed"]


def test_latent_branching_requires_explicit_unstable_switch():
    with pytest.raises(ValueError, match="disabled for mathematical-fidelity stabilization"):
        tiny_config(
            use_latent_state=True,
            latent_state_dim=32,
            latent_state_layers=[0],
            use_latent_branching=True,
            latent_branching_layers=[0],
        )


def test_latent_branching_can_be_enabled_only_with_explicit_switch():
    config = tiny_config(
        use_latent_state=True,
        latent_state_dim=32,
        latent_state_layers=[0],
        use_latent_branching=True,
        enable_unstable_latent_branching=True,
        latent_branching_layers=[0],
    )
    model = ValkModelForCausalLM(config).cuda()

    assert model.model.layers[0].latent_branching is not None
