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
        self.reset_proj = nn.Linear(hidden_size + latent_state_dim, latent_state_dim, bias=True)
        self.gate_proj = nn.Linear(hidden_size + latent_state_dim, latent_state_dim, bias=True)
        self.output_proj = nn.Linear(latent_state_dim, hidden_size, bias=False)
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float) -> None:
        for module in (self.input_proj, self.update_proj, self.reset_proj, self.gate_proj, self.output_proj):
            nn.init.normal_(module.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.reset_proj.bias)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, hidden_size)")
        batch_size, seq_len, _ = hidden_states.shape
        if previous_state is None:
            state_t = hidden_states.new_zeros(batch_size, self.latent_state_dim)
        elif previous_state.shape == (batch_size, self.latent_state_dim):
            state_t = previous_state
        elif previous_state.shape == (batch_size, seq_len, self.latent_state_dim):
            state_t = previous_state[:, -1]
        else:
            raise ValueError(
                "previous_state must have shape (batch, latent_state_dim) or "
                "(batch, sequence, latent_state_dim)"
            )

        states = []
        for timestep in range(seq_len):
            hidden_t = hidden_states[:, timestep]
            projected_t = self.input_proj(hidden_t)
            gate_input = torch.cat([hidden_t, state_t], dim=-1)
            reset_t = torch.sigmoid(self.reset_proj(gate_input))
            candidate_t = torch.tanh(projected_t + self.update_proj(reset_t * state_t))
            gate_t = torch.sigmoid(self.gate_proj(gate_input))
            state_t = gate_t * candidate_t + (1.0 - gate_t) * state_t
            states.append(state_t)

        state_sequence = torch.stack(states, dim=1)
        residual = self.output_proj(state_sequence)
        return residual, state_sequence
