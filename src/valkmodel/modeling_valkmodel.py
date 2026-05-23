from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from .configuration_valkmodel import ValkModelConfig
from .layers.gated_deltanet_layer import GatedDeltaNetLayer, GatedMLP, RMSNorm
from .layers.latent_branching import LatentBranchingModule
from .layers.latent_jepa import JEPAModule
from .layers.latent_state import LatentStateModule
from .losses import compute_weighted_lm_loss
from .utils.jepa_utils import create_jepa_pairs
from .utils.tool_masks import create_tool_mask

try:
    from transformers import PreTrainedModel
except ImportError:  # pragma: no cover
    class PreTrainedModel(nn.Module):
        config_class = None
        base_model_prefix = ""

        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            self.apply(self._init_weights)


@dataclass
class ValkModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: object | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    latent_state: torch.Tensor | None = None
    branch_metrics: tuple[dict[str, torch.Tensor], ...] | None = None


@dataclass
class ValkCausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: object | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    latent_state: torch.Tensor | None = None
    jepa_loss: torch.Tensor | None = None
    jepa_metrics: dict[str, torch.Tensor] | None = None
    branch_metrics: tuple[dict[str, torch.Tensor], ...] | None = None


class ValkModelBlock(nn.Module):
    def __init__(self, config: ValkModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedDeltaNetLayer(
            hidden_size=config.hidden_size,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_v_heads=config.num_v_heads,
            use_gate=config.use_gate,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            mode=config.attn_mode,
            norm_eps=config.norm_eps,
            backend=config.gdn_backend,
            require_fla=config.require_fla,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)
        latent_layers = config.latent_state_layers
        if latent_layers is None and config.use_latent_state:
            latent_layers = [config.num_hidden_layers // 2]
        self.latent_state = (
            LatentStateModule(config.hidden_size, config.latent_state_dim, config.latent_state_init_scale)
            if config.use_latent_state and layer_idx in latent_layers
            else None
        )
        branching_layers = config.latent_branching_layers
        if branching_layers is None and config.use_latent_branching:
            branching_layers = [config.num_hidden_layers // 2]
        self.latent_branching = (
            LatentBranchingModule(
                config.latent_state_dim,
                config.num_branches,
                config.branch_value_temperature,
                config.branch_selection_mode,
                config.latent_branching_init_scale,
            )
            if config.use_latent_branching and layer_idx in branching_layers
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        latent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object | None, torch.Tensor | None, dict[str, torch.Tensor] | None]:
        attn_output, new_cache = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        branch_metrics = None
        if self.latent_state is not None:
            latent_residual, latent_state = self.latent_state(hidden_states, latent_state)
            hidden_states = hidden_states + latent_residual
        if self.latent_branching is not None and latent_state is not None:
            latent_state, branch_metrics = self.latent_branching(latent_state, training=self.training)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, new_cache, latent_state, branch_metrics


class ValkModelPreTrainedModel(PreTrainedModel):
    config_class = ValkModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ValkModel(ValkModelPreTrainedModel):
    def __init__(self, config: ValkModelConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([ValkModelBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[object | None, ...] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> ValkModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("must specify input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one cache entry per layer")

        hidden_states = inputs_embeds
        latent_state = kwargs.get("latent_state")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        next_cache = [] if use_cache else None
        all_hidden_states = [] if output_hidden_states else None
        all_branch_metrics = []
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_cache = None if past_key_values is None else past_key_values[layer_idx]
            hidden_states, new_layer_cache, latent_state, branch_metrics = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=layer_cache,
                use_cache=use_cache,
                latent_state=latent_state,
            )
            if branch_metrics is not None:
                all_branch_metrics.append(branch_metrics)
            if use_cache:
                next_cache.append(new_layer_cache)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return ValkModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=tuple(next_cache) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            latent_state=latent_state,
            branch_metrics=tuple(all_branch_metrics) if all_branch_metrics else None,
        )


class ValkModelForCausalLM(ValkModelPreTrainedModel):
    def __init__(self, config: ValkModelConfig):
        super().__init__(config)
        self.model = ValkModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.jepa_module = (
            JEPAModule(config.latent_state_dim, config.jepa_hidden_dim, config.jepa_ema_momentum, config.jepa_init_scale)
            if config.use_jepa
            else None
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        tool_mask: torch.Tensor | None = None,
        training_lambdas: dict[str, float] | None = None,
        **kwargs,
    ) -> ValkCausalLMOutput:
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        jepa_loss = None
        jepa_metrics = None
        tool_weight = self.config.tool_loss_weight if training_lambdas is None else training_lambdas.get("tool", self.config.tool_loss_weight)
        jepa_weight = self.config.jepa_loss_weight if training_lambdas is None else training_lambdas.get("jepa", self.config.jepa_loss_weight)
        branch_weight = self.config.branch_diversity_weight if training_lambdas is None else training_lambdas.get("branch", self.config.branch_diversity_weight)
        branch_entropy_weight = self.config.branch_entropy_weight if training_lambdas is None else training_lambdas.get("branch_entropy", self.config.branch_entropy_weight)
        if labels is not None:
            ignore_index = self.config.pad_token_id if self.config.pad_token_id is not None else -100
            if tool_mask is None and input_ids is not None and (
                self.config.tool_call_token_id is not None
                or self.config.tool_result_token_id is not None
                or self.config.reasoning_start_token_id is not None
            ):
                tool_mask = create_tool_mask(
                    input_ids,
                    tool_call_token_id=self.config.tool_call_token_id,
                    tool_result_token_id=self.config.tool_result_token_id,
                    reasoning_start_token_id=self.config.reasoning_start_token_id,
                    reasoning_end_token_id=self.config.reasoning_end_token_id,
                    branch_marker_token_id=self.config.branch_marker_token_id,
                    tool_call_span=self.config.tool_call_span,
                    tool_result_span=self.config.tool_result_span,
                )
            loss = compute_weighted_lm_loss(
                logits,
                labels,
                tool_mask=tool_mask,
                tool_weight=tool_weight,
                ignore_index=ignore_index,
            )
            if self.training and self.jepa_module is not None and outputs.latent_state is not None:
                batch_size, seq_len, _ = outputs.latent_state.shape
                horizons = torch.randint(
                    self.config.jepa_min_horizon,
                    self.config.jepa_max_horizon + 1,
                    (batch_size, seq_len),
                    device=outputs.latent_state.device,
                )
                current, future, horizon_mask = create_jepa_pairs(outputs.latent_state, horizons)
                jepa_loss, jepa_metrics = self.jepa_module(current, future, horizon_mask)
                loss = loss + jepa_weight * jepa_loss
            if self.training and outputs.branch_metrics is not None:
                if branch_weight > 0:
                    diversity_loss = torch.stack([metrics["diversity_loss"] for metrics in outputs.branch_metrics]).mean()
                    loss = loss + branch_weight * diversity_loss
                if branch_entropy_weight > 0:
                    branch_entropy = torch.stack([metrics["branch_entropy"] for metrics in outputs.branch_metrics]).mean()
                    loss = loss - branch_entropy_weight * branch_entropy
        return ValkCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            latent_state=outputs.latent_state,
            jepa_loss=jepa_loss,
            jepa_metrics=jepa_metrics,
            branch_metrics=outputs.branch_metrics,
        )
