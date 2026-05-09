from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentBranchingModule(nn.Module):
    def __init__(
        self,
        latent_state_dim: int,
        num_branches: int = 4,
        value_temperature: float = 1.0,
        selection_mode: str = "top1",
        init_scale: float = 0.02,
    ):
        super().__init__()
        if latent_state_dim <= 0:
            raise ValueError("latent_state_dim must be positive")
        if num_branches < 2:
            raise ValueError("num_branches must be at least 2")
        if value_temperature <= 0:
            raise ValueError("value_temperature must be positive")
        if selection_mode not in {"soft", "top1"}:
            raise ValueError("selection_mode must be 'soft' or 'top1'")
        self.latent_state_dim = latent_state_dim
        self.num_branches = num_branches
        self.value_temperature = value_temperature
        self.selection_mode = selection_mode
        self.branch_projections = nn.ModuleList(
            [nn.Linear(latent_state_dim, latent_state_dim, bias=False) for _ in range(num_branches)]
        )
        self.value_heads = nn.ModuleList([nn.Linear(latent_state_dim, 1, bias=True) for _ in range(num_branches)])
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float) -> None:
        for branch in self.branch_projections:
            nn.init.normal_(branch.weight, mean=0.0, std=init_scale)
        for head in self.value_heads:
            nn.init.normal_(head.weight, mean=0.0, std=init_scale)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        latent_state: torch.Tensor,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if latent_state.ndim != 3:
            raise ValueError("latent_state must have shape (batch, sequence, latent_dim)")
        if latent_state.shape[-1] != self.latent_state_dim:
            raise ValueError("latent_state last dimension must match latent_state_dim")
        training = self.training if training is None else training
        branch_states = torch.stack([branch(latent_state) for branch in self.branch_projections], dim=-2)
        branch_values = torch.cat([head(branch_states[:, :, idx]).squeeze(-1).unsqueeze(-1) for idx, head in enumerate(self.value_heads)], dim=-1)
        branch_probs = torch.softmax(branch_values / self.value_temperature, dim=-1)
        if training or self.selection_mode == "soft":
            merged_state = (branch_probs.unsqueeze(-1) * branch_states).sum(dim=-2)
            selected_branch = branch_values.argmax(dim=-1)
        else:
            selected_branch = branch_values.argmax(dim=-1)
            gather_index = selected_branch.unsqueeze(-1).unsqueeze(-1).expand(*selected_branch.shape, 1, self.latent_state_dim)
            merged_state = branch_states.gather(dim=-2, index=gather_index).squeeze(-2)
        metrics = {
            "branch_values": branch_values,
            "branch_probs": branch_probs,
            "diversity_loss": self._compute_diversity_loss(branch_states),
            "branch_entropy": self._compute_entropy(branch_probs),
            "branch_variance": branch_states.var(dim=-2, unbiased=False).mean(),
            "selected_branch": selected_branch,
        }
        return merged_state, metrics

    def _compute_diversity_loss(self, branch_states: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(branch_states, dim=-1, eps=1e-6)
        similarities = torch.matmul(normalized, normalized.transpose(-1, -2))
        pair_mask = torch.triu(
            torch.ones(self.num_branches, self.num_branches, dtype=torch.bool, device=branch_states.device),
            diagonal=1,
        )
        pairwise = similarities[..., pair_mask]
        return pairwise.pow(2).mean()

    def _compute_entropy(self, branch_probs: torch.Tensor) -> torch.Tensor:
        return -(branch_probs * branch_probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
