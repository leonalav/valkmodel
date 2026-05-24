import torch

from valkmodel.layers.latent_jepa import JEPAModule
from valkmodel.utils.jepa_utils import compute_jepa_metrics, compute_normalized_mse, create_jepa_pairs


def predictor_linear_grads(module: JEPAModule):
    return [submodule.weight.grad for submodule in module.predictor if isinstance(submodule, torch.nn.Linear)]


def test_create_jepa_pairs_uses_per_position_horizons_and_masks_invalid_targets():
    latent = torch.arange(1 * 5 * 2, dtype=torch.float32).view(1, 5, 2)
    horizons = torch.tensor([[1, 2, 1, 2, 1]])

    current, future, mask = create_jepa_pairs(latent, horizons)

    assert torch.equal(current, latent)
    assert torch.equal(future[0, 0], latent[0, 1])
    assert torch.equal(future[0, 1], latent[0, 3])
    assert torch.equal(future[0, 2], latent[0, 3])
    assert torch.equal(mask, torch.tensor([[True, True, True, False, False]]))
    assert torch.equal(future[0, 3], torch.zeros(2))


def test_normalized_mse_ignores_masked_pairs_and_matches_cosine_distance():
    predictions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    targets = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-1.0, -1.0]]])
    mask = torch.tensor([[True, True, False]])

    loss = compute_normalized_mse(predictions, targets, mask)

    expected = torch.tensor((0.0 + 2.0) / 2.0)
    assert torch.allclose(loss, expected)


def test_jepa_metrics_report_variance_and_cosine_on_active_pairs():
    predictions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])
    targets = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [9.0, 9.0]]])
    mask = torch.tensor([[True, True, False]])

    metrics = compute_jepa_metrics(predictions, targets, mask)

    assert metrics["prediction_variance"] > 0
    assert metrics["target_variance"] == 0
    assert torch.allclose(metrics["cosine_mean"], torch.tensor(0.5))
    assert metrics["target_collapsed"]
    assert not metrics["prediction_collapsed"]


def test_jepa_module_returns_finite_loss_metrics_and_blocks_target_gradients():
    torch.manual_seed(0)
    module = JEPAModule(latent_state_dim=4, jepa_hidden_dim=6, ema_momentum=0.95)
    current = torch.randn(2, 5, 4, requires_grad=True)
    future = torch.randn(2, 5, 4, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)

    loss, metrics = module(current, future, mask)
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["prediction_variance"])
    assert torch.isfinite(metrics["target_variance"])
    assert module.context_encoder.weight.grad is not None
    assert all(grad is not None for grad in predictor_linear_grads(module))
    assert module.target_encoder.weight.grad is None
    assert current.grad is not None


def test_jepa_module_ema_update_matches_reference_momentum_rule():
    module = JEPAModule(latent_state_dim=3, jepa_hidden_dim=3, ema_momentum=0.8)
    with torch.no_grad():
        module.context_encoder.weight.fill_(2.0)
        module.target_encoder.weight.fill_(10.0)

    module.update_target_encoder()

    expected = torch.full_like(module.target_encoder.weight, 0.8 * 10.0 + 0.2 * 2.0)
    assert torch.allclose(module.target_encoder.weight, expected)
