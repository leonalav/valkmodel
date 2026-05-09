import torch

from valkmodel.layers.latent_branching import LatentBranchingModule
from valkmodel.losses import compute_branch_value_loss


def test_latent_branching_training_soft_merge_returns_probabilities_and_metrics():
    torch.manual_seed(0)
    module = LatentBranchingModule(latent_state_dim=4, num_branches=3, value_temperature=0.5)
    latent_state = torch.randn(2, 5, 4, requires_grad=True)

    merged_state, metrics = module(latent_state, training=True)
    loss = merged_state.pow(2).mean() + metrics["diversity_loss"]
    loss.backward()

    assert merged_state.shape == latent_state.shape
    assert metrics["branch_values"].shape == (2, 5, 3)
    assert metrics["branch_probs"].shape == (2, 5, 3)
    assert torch.allclose(metrics["branch_probs"].sum(dim=-1), torch.ones(2, 5))
    assert metrics["diversity_loss"].shape == ()
    assert metrics["diversity_loss"] >= 0
    assert metrics["branch_entropy"] > 0
    assert metrics["branch_variance"] > 0
    assert latent_state.grad is not None
    assert all(branch.weight.grad is not None for branch in module.branch_projections)
    assert all(head.weight.grad is not None for head in module.value_heads)


def test_latent_branching_eval_top1_selects_highest_value_branch():
    module = LatentBranchingModule(latent_state_dim=2, num_branches=2, selection_mode="top1")
    with torch.no_grad():
        module.branch_projections[0].weight.copy_(torch.eye(2))
        module.branch_projections[1].weight.copy_(2 * torch.eye(2))
        module.value_heads[0].weight.fill_(0.0)
        module.value_heads[0].bias.fill_(0.0)
        module.value_heads[1].weight.fill_(0.0)
        module.value_heads[1].bias.fill_(1.0)
    latent_state = torch.tensor([[[1.0, 2.0]]])

    merged_state, metrics = module(latent_state, training=False)

    assert torch.equal(metrics["selected_branch"], torch.ones(1, 1, dtype=torch.long))
    assert torch.allclose(merged_state, torch.tensor([[[2.0, 4.0]]]))


def test_latent_branching_diversity_penalty_is_higher_for_identical_than_orthogonal_branches():
    module = LatentBranchingModule(latent_state_dim=2, num_branches=2)
    latent_state = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    with torch.no_grad():
        module.branch_projections[0].weight.copy_(torch.eye(2))
        module.branch_projections[1].weight.copy_(torch.eye(2))

    _, identical_metrics = module(latent_state, training=True)
    with torch.no_grad():
        module.branch_projections[1].weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    _, orthogonal_metrics = module(latent_state, training=True)

    assert identical_metrics["diversity_loss"] > orthogonal_metrics["diversity_loss"]


def test_branch_value_loss_uses_expected_branch_value_and_mask():
    branch_values = torch.tensor([[[1.0, 3.0], [9.0, 9.0]]], requires_grad=True)
    target_values = torch.tensor([[2.5, 0.0]])
    mask = torch.tensor([[True, False]])

    loss = compute_branch_value_loss(branch_values, target_values, mask)
    loss.backward()

    probs = torch.softmax(branch_values.detach()[0, 0], dim=-1)
    expected_value = (probs * branch_values.detach()[0, 0]).sum()
    expected = (expected_value - target_values[0, 0]).pow(2)
    assert torch.allclose(loss, expected)
    assert branch_values.grad is not None
    assert torch.equal(branch_values.grad[0, 1], torch.zeros(2))
