from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.jepa_utils import compute_jepa_metrics, compute_normalized_mse


class JEPAModule(nn.Module):
    def __init__(
        self,
        latent_state_dim: int,
        jepa_hidden_dim: int,
        ema_momentum: float = 0.996,
        init_scale: float = 0.02,
    ):
        super().__init__()
        if latent_state_dim <= 0 or jepa_hidden_dim <= 0:
            raise ValueError("latent_state_dim and jepa_hidden_dim must be positive")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in [0, 1)")
        self.latent_state_dim = latent_state_dim
        self.jepa_hidden_dim = jepa_hidden_dim
        self.ema_momentum = ema_momentum
        self.context_encoder = nn.Linear(latent_state_dim, jepa_hidden_dim, bias=False)
        self.predictor = nn.Sequential(
            nn.Linear(jepa_hidden_dim, 4 * jepa_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(4 * jepa_hidden_dim, jepa_hidden_dim, bias=False),
        )
        self.target_encoder = nn.Linear(latent_state_dim, jepa_hidden_dim, bias=False)
        self.reset_parameters(init_scale)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    def reset_parameters(self, init_scale: float) -> None:
        nn.init.normal_(self.context_encoder.weight, mean=0.0, std=init_scale)
        for module in self.predictor:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.target_encoder.weight.data.copy_(self.context_encoder.weight.data)

    def forward(
        self,
        current_latent_state: torch.Tensor,
        future_latent_state: torch.Tensor,
        horizon_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        predictions = self.predictor(self.context_encoder(current_latent_state))
        with torch.no_grad():
            targets = self.target_encoder(future_latent_state)
        loss = compute_normalized_mse(predictions, targets, horizon_mask)
        metrics = compute_jepa_metrics(predictions.detach(), targets.detach(), horizon_mask)
        return loss, metrics

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        momentum = self.ema_momentum if momentum is None else momentum
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_((1.0 - momentum) * online.detach().data)
