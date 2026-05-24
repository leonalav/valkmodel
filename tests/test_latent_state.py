import torch

from valkmodel import ValkModelConfig, ValkModelForCausalLM
from valkmodel.layers.latent_state import LatentStateModule


def tiny_latent_config(**overrides):
    values = {
        "vocab_size": 97,
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "num_heads": 3,
        "head_dim": 16,
        "num_v_heads": 3,
        "intermediate_size": 128,
        "max_position_embeddings": 128,
        "use_latent_state": True,
        "latent_state_dim": 32,
        "latent_state_layers": [1],
    }
    values.update(overrides)
    return ValkModelConfig(**values)


def test_latent_state_module_returns_residual_and_noncollapsed_state():
    torch.manual_seed(0)
    module = LatentStateModule(hidden_size=64, latent_state_dim=32, init_scale=0.02)
    hidden_states = torch.randn(2, 5, 64)

    residual, state = module(hidden_states)

    assert residual.shape == hidden_states.shape
    assert state.shape == (2, 5, 32)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(state).all()
    assert state.var() > 1e-8


def test_latent_state_module_uses_boundary_previous_state():
    torch.manual_seed(0)
    module = LatentStateModule(hidden_size=64, latent_state_dim=32, init_scale=0.02)
    hidden_states = torch.randn(2, 5, 64)
    _, first_state = module(hidden_states)

    _, continued_state = module(hidden_states, previous_state=first_state[:, -1])

    assert not torch.allclose(first_state, continued_state)


def test_model_threads_latent_state_and_backpropagates_through_it():
    torch.manual_seed(0)
    model = ValkModelForCausalLM(tiny_latent_config()).cuda()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6), device="cuda")

    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()

    assert outputs.latent_state is not None
    assert outputs.latent_state.shape == (2, 6, model.config.latent_state_dim)
    latent_grads = [p.grad for name, p in model.named_parameters() if "latent_state" in name and p.grad is not None]
    assert latent_grads
    assert all(torch.isfinite(grad).all() for grad in latent_grads)


def test_latent_state_rejects_invalid_previous_state_shape():
    module = LatentStateModule(hidden_size=64, latent_state_dim=32, init_scale=0.02)
    hidden_states = torch.randn(2, 1, 64)
    previous_state = torch.randn(2, 5, 32)

    try:
        module(hidden_states, previous_state=previous_state)
    except ValueError as exc:
        assert "previous_state must have shape" in str(exc)
    else:
        raise AssertionError("expected ValueError")
