from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
except ImportError as _fla_import_error:
    raise ImportError(
        "flash-linear-attention (fla) is required. "
        "Install it with: pip install flash-linear-attention"
    ) from _fla_import_error


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, bias: bool):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        padded = F.pad(channels_first, (self.kernel_size - 1, 0))
        return F.silu(self.conv(padded).transpose(1, 2))


class GatedDeltaNetLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expand_v: float,
        head_dim: int,
        num_heads: int,
        num_v_heads: int,
        use_gate: bool,
        use_short_conv: bool,
        conv_size: int,
        mode: str = "chunk",
        conv_bias: bool = False,
        norm_eps: float = 1e-6,
        backend: str = "fla",
        require_fla: bool = True,
    ):
        super().__init__()
        if mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("mode must be 'chunk' or 'fused_recurrent'")
        self.mode = mode
        self.backend = backend
        if self.backend != "fla":
            raise NotImplementedError(
                "Only gdn_backend='fla' is supported. CPU fallback backends are intentionally unsupported."
            )
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.key_dim = num_heads * head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.value_dim = num_v_heads * self.head_v_dim

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_v_heads, bias=False)
        A = torch.empty(num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_floor = 1e-4
        dt = torch.exp(torch.rand(num_v_heads, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, conv_bias)

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _project_inputs(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        if self.use_short_conv:
            q = self.q_conv1d(q)
            k = self.k_conv1d(k)
            v = self.v_conv1d(v)
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, object | None]:
        if attention_mask is not None and not attention_mask.to(dtype=torch.bool).all():
            raise ValueError(
                "GatedDeltaNetLayer does not support padded attention_mask values yet. "
                "Padding would update the recurrent GDN state as real tokens; pass dense sequences "
                "or implement reference-style unpadding/cu_seqlens first."
            )
        if hidden_states.device.type != "cuda":
            raise ValueError("FLA GatedDeltaNetLayer requires CUDA tensors")
        q, k, v = self._project_inputs(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        q = F.normalize(q.view(batch_size, seq_len, self.num_heads, self.head_dim), dim=-1)
        k = F.normalize(k.view(batch_size, seq_len, self.num_heads, self.head_dim), dim=-1)
        v = v.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        beta = self.b_proj(hidden_states).sigmoid()
        rule = fused_recurrent_gated_delta_rule if self.mode == "fused_recurrent" else chunk_gated_delta_rule
        o, recurrent_state = rule(
            q=q,
            k=k,
            v=v,
            g=self.a_proj(hidden_states),
            beta=beta,
            initial_state=past_key_values,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )
        if self.use_gate:
            g = self.g_proj(hidden_states).view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
            o = self.o_norm(o * F.silu(g))
        else:
            o = self.o_norm(o)
        return self.o_proj(o.reshape(batch_size, seq_len, self.value_dim)), recurrent_state if use_cache else None
