from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentStateModule(nn.Module):
    def __init__(self, hidden_size: int, latent_state_dim: int, init_scale: float = 0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_state_dim = latent_state_dim
        self.input_proj = nn.Linear(hidden_size, latent_state_dim, bias=False)
        self.update_proj = nn.Linear(latent_state_dim, latent_state_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size + latent_state_dim, latent_state_dim, bias=True)
        self.output_proj = nn.Linear(latent_state_dim, hidden_size, bias=False)
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float) -> None:
        for module in (self.input_proj, self.update_proj, self.gate_proj, self.output_proj):
            nn.init.normal_(module.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_proj(hidden_states)
        if previous_state is None:
            previous_state = projected.new_zeros(projected.shape)
        if previous_state.shape != projected.shape:
            raise ValueError(
                "previous_state must have the same batch and sequence shape as hidden_states; "
                "incremental latent-state cache is not implemented in Phase 0-1"
            )

        candidate = torch.tanh(projected + self.update_proj(previous_state))
        gate = torch.sigmoid(self.gate_proj(torch.cat([hidden_states, previous_state], dim=-1)))
        state = gate * candidate + (1.0 - gate) * previous_state
        residual = self.output_proj(state)
        return residual, state
