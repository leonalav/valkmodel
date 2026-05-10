from __future__ import annotations

try:
    from transformers import PretrainedConfig
except ImportError:  # pragma: no cover
    class PretrainedConfig:
        model_type = ""

        def __init__(self, pad_token_id=None, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False, **kwargs):
            self.pad_token_id = pad_token_id
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.tie_word_embeddings = tie_word_embeddings
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self):
            data = dict(self.__dict__)
            data["model_type"] = self.model_type
            return data

from .presets import VALKMODEL_PRESETS


class ValkModelConfig(PretrainedConfig):
    model_type = "valkmodel"

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 768,
        expand_v: float = 2.0,
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        head_dim: int = 96,
        num_heads: int = 6,
        num_v_heads: int | None = None,
        max_position_embeddings: int = 272_000,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 12,
        norm_eps: float = 1e-6,
        vocab_size: int = 32_000,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        gdn_backend: str = "auto",
        require_fla: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        use_latent_state: bool = False,
        tool_call_token_id: int | None = None,
        tool_result_token_id: int | None = None,
        reasoning_start_token_id: int | None = None,
        reasoning_end_token_id: int | None = None,
        branch_marker_token_id: int | None = None,
        tool_loss_weight: float = 2.0,
        tool_call_span: int = 32,
        tool_result_span: int = 64,
        use_jepa: bool = False,
        jepa_hidden_dim: int | None = None,
        jepa_ema_momentum: float = 0.996,
        jepa_min_horizon: int = 1,
        jepa_max_horizon: int = 16,
        jepa_loss_weight: float = 0.1,
        jepa_init_scale: float = 0.02,
        max_training_seq_len: int | None = None,
        chunk_size: int = 2048,
        use_packed_sequences: bool = False,
        document_separator_token_id: int | None = None,
        latent_state_dim: int | None = None,
        latent_state_layers: list[int] | None = None,
        latent_state_init_scale: float = 0.02,
        use_latent_branching: bool = False,
        latent_branching_layers: list[int] | None = None,
        num_branches: int = 4,
        branch_value_temperature: float = 1.0,
        branch_selection_mode: str = "top1",
        branch_diversity_weight: float = 0.01,
        branch_entropy_weight: float = 0.0,
        branch_value_loss_weight: float = 0.0,
        latent_branching_init_scale: float = 0.02,
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size if intermediate_size is not None else hidden_size * (hidden_ratio or 4)
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.gdn_backend = gdn_backend
        self.require_fla = require_fla
        self.use_latent_state = use_latent_state
        self.tool_call_token_id = tool_call_token_id
        self.tool_result_token_id = tool_result_token_id
        self.reasoning_start_token_id = reasoning_start_token_id
        self.reasoning_end_token_id = reasoning_end_token_id
        self.branch_marker_token_id = branch_marker_token_id
        self.tool_loss_weight = tool_loss_weight
        self.tool_call_span = tool_call_span
        self.tool_result_span = tool_result_span
        self.use_jepa = use_jepa
        self.jepa_ema_momentum = jepa_ema_momentum
        self.jepa_min_horizon = jepa_min_horizon
        self.jepa_max_horizon = jepa_max_horizon
        self.jepa_loss_weight = jepa_loss_weight
        self.jepa_init_scale = jepa_init_scale
        self.max_training_seq_len = max_training_seq_len if max_training_seq_len is not None else max_position_embeddings
        self.chunk_size = chunk_size
        self.use_packed_sequences = use_packed_sequences
        self.document_separator_token_id = document_separator_token_id
        self.latent_state_dim = latent_state_dim if latent_state_dim is not None else max(1, hidden_size // 2)
        self.jepa_hidden_dim = jepa_hidden_dim if jepa_hidden_dim is not None else self.latent_state_dim
        self.latent_state_layers = latent_state_layers
        self.latent_state_init_scale = latent_state_init_scale
        self.use_latent_branching = use_latent_branching
        self.latent_branching_layers = latent_branching_layers
        self.num_branches = num_branches
        self.branch_value_temperature = branch_value_temperature
        self.branch_selection_mode = branch_selection_mode
        self.branch_diversity_weight = branch_diversity_weight
        self.branch_entropy_weight = branch_entropy_weight
        self.branch_value_loss_weight = branch_value_loss_weight
        self.latent_branching_init_scale = latent_branching_init_scale

        self.key_dim = self.num_heads * self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.value_dim = int(self.num_v_heads * self.head_dim * self.expand_v)
        self._validate_geometry()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ValkModelConfig":
        if name not in VALKMODEL_PRESETS:
            raise ValueError(f"Unknown ValkModel preset: {name}")
        values = {**VALKMODEL_PRESETS[name], **overrides}
        return cls(**values)

    def estimate_parameters(self) -> int:
        embedding = self.vocab_size * self.hidden_size
        gdn = self._estimate_gdn_layer_parameters()
        mlp = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size
        layers = self.num_hidden_layers * (gdn + mlp + norms)
        output = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        latent = 0
        if self.use_latent_state:
            latent_layers = self.latent_state_layers if self.latent_state_layers is not None else [self.num_hidden_layers // 2]
            latent = len(latent_layers) * self._estimate_latent_state_parameters()
        jepa = self._estimate_jepa_parameters() if self.use_jepa else 0
        branching = 0
        if self.use_latent_branching:
            branching_layers = self.latent_branching_layers if self.latent_branching_layers is not None else [self.num_hidden_layers // 2]
            branching = len(branching_layers) * self._estimate_latent_branching_parameters()
        return int(embedding + layers + output + latent + jepa + branching)

    def _estimate_gdn_layer_parameters(self) -> int:
        g_proj = self.hidden_size * self.value_dim if self.use_gate else 0
        return int(
            self.hidden_size * self.key_dim
            + self.hidden_size * self.key_dim
            + self.hidden_size * self.value_dim
            + self.hidden_size * self.num_v_heads
            + self.hidden_size * self.num_v_heads
            + g_proj
            + self.value_dim * self.hidden_size
            + self.num_v_heads
            + self.num_v_heads
        )

    def _estimate_latent_state_parameters(self) -> int:
        return int(
            self.hidden_size * self.latent_state_dim
            + self.latent_state_dim * self.latent_state_dim
            + (self.hidden_size + self.latent_state_dim) * self.latent_state_dim
            + self.latent_state_dim
            + self.latent_state_dim * self.hidden_size
        )

    def _estimate_jepa_parameters(self) -> int:
        return int(
            self.latent_state_dim * self.jepa_hidden_dim
            + self.jepa_hidden_dim * self.jepa_hidden_dim
            + self.latent_state_dim * self.jepa_hidden_dim
        )

    def _estimate_latent_branching_parameters(self) -> int:
        return int(
            self.num_branches * self.latent_state_dim * self.latent_state_dim
            + self.num_branches * self.latent_state_dim
            + self.num_branches
        )

    def _validate_geometry(self) -> None:
        if self.attn_mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("attn_mode must be 'chunk' or 'fused_recurrent'")
        if self.gdn_backend not in {"auto", "naive", "fla"}:
            raise ValueError("gdn_backend must be 'auto', 'naive', or 'fla'")
        if self.use_gate and self.num_heads * self.head_dim != int(0.75 * self.hidden_size):
            raise ValueError("num_heads * head_dim must equal 0.75 * hidden_size when use_gate=True")
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_heads when using grouped value heads")
        if self.head_dim * self.expand_v != self.head_v_dim:
            raise ValueError("expand_v must produce an integer head value dimension")
        if self.num_v_heads * self.head_dim * self.expand_v != self.value_dim:
            raise ValueError("expand_v must produce an integer value dimension")
        for field_name in (
            "tool_call_token_id",
            "tool_result_token_id",
            "reasoning_start_token_id",
            "reasoning_end_token_id",
            "branch_marker_token_id",
        ):
            token_id = getattr(self, field_name)
            if token_id is not None and not 0 <= token_id < self.vocab_size:
                raise ValueError(f"{field_name} must be None or in [0, vocab_size)")
        if self.tool_loss_weight <= 0:
            raise ValueError("tool_loss_weight must be positive")
        if self.tool_call_span < 0 or self.tool_result_span < 0:
            raise ValueError("tool spans must be nonnegative")
        if self.use_jepa and not self.use_latent_state:
            raise ValueError("use_jepa requires use_latent_state=True")
        if self.jepa_hidden_dim <= 0:
            raise ValueError("jepa_hidden_dim must be positive")
        if not 0.9 <= self.jepa_ema_momentum < 1.0:
            raise ValueError("jepa_ema_momentum must be in [0.9, 1.0)")
        if self.jepa_min_horizon < 1 or self.jepa_max_horizon < self.jepa_min_horizon:
            raise ValueError("jepa horizon range must be positive and ordered")
        if self.jepa_loss_weight < 0:
            raise ValueError("jepa_loss_weight must be nonnegative")
        if self.max_training_seq_len <= 0 or self.max_training_seq_len > self.max_position_embeddings:
            raise ValueError("max_training_seq_len must be positive and no larger than max_position_embeddings")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.document_separator_token_id is not None and not 0 <= self.document_separator_token_id < self.vocab_size:
            raise ValueError("document_separator_token_id must be None or in [0, vocab_size)")
        if self.latent_state_dim <= 0:
            raise ValueError("latent_state_dim must be positive")
        if self.use_latent_branching and not self.use_latent_state:
            raise ValueError("use_latent_branching requires use_latent_state=True")
        if self.num_branches < 2:
            raise ValueError("num_branches must be at least 2")
        if self.branch_value_temperature <= 0:
            raise ValueError("branch_value_temperature must be positive")
        if self.branch_selection_mode not in {"soft", "top1"}:
            raise ValueError("branch_selection_mode must be 'soft' or 'top1'")
        if self.branch_diversity_weight < 0:
            raise ValueError("branch_diversity_weight must be nonnegative")
        if self.branch_entropy_weight < 0:
            raise ValueError("branch_entropy_weight must be nonnegative")
        if self.branch_value_loss_weight < 0:
            raise ValueError("branch_value_loss_weight must be nonnegative")
        if self.latent_branching_layers is not None and any(
            layer < 0 or layer >= self.num_hidden_layers for layer in self.latent_branching_layers
        ):
            raise ValueError("latent_branching_layers entries must be valid layer indices")
