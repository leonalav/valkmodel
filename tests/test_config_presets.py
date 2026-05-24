import pytest

from valkmodel import VALKMODEL_PRESETS, ValkModelConfig


EXPECTED_PRESETS = ("130m", "260m", "520m", "780m", "1.2b", "2.8b", "5b", "8b")


def test_all_expected_presets_instantiate_valid_configs():
    assert tuple(VALKMODEL_PRESETS) == EXPECTED_PRESETS

    for name in EXPECTED_PRESETS:
        config = ValkModelConfig.from_preset(name)
        assert config.model_type == "valkmodel"
        assert config.num_heads * config.head_dim == int(0.75 * config.hidden_size)
        assert config.value_dim == int(config.num_v_heads * config.head_dim * config.expand_v)
        assert config.max_position_embeddings in {272_000, 512_000, 768_000, 1_000_000}


def test_config_rejects_invalid_gated_head_geometry():
    with pytest.raises(ValueError, match="num_heads \* head_dim"):
        ValkModelConfig(hidden_size=768, num_heads=5, head_dim=96)


def test_config_rejects_invalid_value_head_grouping():
    with pytest.raises(ValueError, match="num_v_heads"):
        ValkModelConfig(hidden_size=768, num_heads=6, head_dim=96, num_v_heads=7)


def test_config_rejects_invalid_gdn_backend():
    with pytest.raises(ValueError, match="gdn_backend"):
        ValkModelConfig(gdn_backend="unknown")


def test_config_rejects_invalid_tool_objective_fields():
    with pytest.raises(ValueError, match="tool_call_token_id"):
        ValkModelConfig(tool_call_token_id=32_000)
    with pytest.raises(ValueError, match="tool_loss_weight"):
        ValkModelConfig(tool_loss_weight=0.0)
    with pytest.raises(ValueError, match="tool spans"):
        ValkModelConfig(tool_call_span=-1)


def test_config_rejects_invalid_jepa_fields():
    with pytest.raises(ValueError, match="use_jepa"):
        ValkModelConfig(use_jepa=True, use_latent_state=False)
    with pytest.raises(ValueError, match="jepa_ema_momentum"):
        ValkModelConfig(use_latent_state=True, use_jepa=True, jepa_ema_momentum=1.0)
    with pytest.raises(ValueError, match="jepa horizon"):
        ValkModelConfig(use_latent_state=True, use_jepa=True, jepa_min_horizon=4, jepa_max_horizon=2)


def test_config_rejects_invalid_latent_branching_fields():
    with pytest.raises(ValueError, match="use_latent_branching"):
        ValkModelConfig(use_latent_branching=True, use_latent_state=False)
    with pytest.raises(ValueError, match="num_branches"):
        ValkModelConfig(use_latent_state=True, use_latent_branching=True, enable_unstable_latent_branching=True, num_branches=1)
    with pytest.raises(ValueError, match="branch_value_temperature"):
        ValkModelConfig(use_latent_state=True, use_latent_branching=True, enable_unstable_latent_branching=True, branch_value_temperature=0.0)
    with pytest.raises(ValueError, match="branch_selection_mode"):
        ValkModelConfig(use_latent_state=True, use_latent_branching=True, enable_unstable_latent_branching=True, branch_selection_mode="sample")
    with pytest.raises(ValueError, match="latent_branching_layers"):
        ValkModelConfig(use_latent_state=True, use_latent_branching=True, enable_unstable_latent_branching=True, latent_branching_layers=[99])


def test_config_rejects_invalid_long_context_fields():
    with pytest.raises(ValueError, match="max_training_seq_len"):
        ValkModelConfig(max_training_seq_len=300_000, max_position_embeddings=272_000)
    with pytest.raises(ValueError, match="chunk_size"):
        ValkModelConfig(chunk_size=0)
    with pytest.raises(ValueError, match="document_separator_token_id"):
        ValkModelConfig(document_separator_token_id=32_000)


def test_config_round_trips_through_dict():
    original = ValkModelConfig.from_preset("130m")
    restored = ValkModelConfig(**original.to_dict())

    assert restored.to_dict() == original.to_dict()


def test_parameter_estimate_is_positive_and_ordered_by_preset_size():
    estimates = [ValkModelConfig.from_preset(name).estimate_parameters() for name in EXPECTED_PRESETS]

    assert all(estimate > 0 for estimate in estimates)
    assert estimates == sorted(estimates)


def test_parameter_estimate_matches_baseline_projection_geometry():
    config = ValkModelConfig(
        vocab_size=97,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=3,
        head_dim=16,
        num_v_heads=3,
        intermediate_size=128,
        tie_word_embeddings=False,
    )
    key_dim = config.num_heads * config.head_dim
    value_dim = config.num_v_heads * int(config.head_dim * config.expand_v)
    gdn = (
        config.hidden_size * key_dim
        + config.hidden_size * key_dim
        + config.hidden_size * value_dim
        + config.hidden_size * config.num_v_heads
        + config.hidden_size * config.num_v_heads
        + config.hidden_size * value_dim
        + value_dim * config.hidden_size
        + config.num_v_heads
        + config.num_v_heads
    )
    mlp = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    expected = (
        config.vocab_size * config.hidden_size
        + config.num_hidden_layers * (gdn + mlp + norms)
        + config.vocab_size * config.hidden_size
    )

    assert config.estimate_parameters() == expected


def test_parameter_estimate_counts_each_latent_state_layer_exactly():
    config = ValkModelConfig(
        vocab_size=97,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=3,
        head_dim=16,
        num_v_heads=3,
        intermediate_size=128,
        use_latent_state=True,
        latent_state_dim=32,
        latent_state_layers=[0, 1],
    )
    without_latent = ValkModelConfig(
        vocab_size=97,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=3,
        head_dim=16,
        num_v_heads=3,
        intermediate_size=128,
    )
    recurrent_gate = (config.hidden_size + config.latent_state_dim) * config.latent_state_dim + config.latent_state_dim
    latent_per_layer = (
        config.hidden_size * config.latent_state_dim
        + config.latent_state_dim * config.latent_state_dim
        + recurrent_gate
        + recurrent_gate
        + config.latent_state_dim * config.hidden_size
    )

    assert config.estimate_parameters() - without_latent.estimate_parameters() == 2 * latent_per_layer


def test_parameter_estimate_counts_jepa_module_once_when_enabled():
    config = ValkModelConfig(
        vocab_size=97,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=3,
        head_dim=16,
        num_v_heads=3,
        intermediate_size=128,
        use_latent_state=True,
        latent_state_dim=32,
        use_jepa=True,
        jepa_hidden_dim=24,
    )
    without_jepa = ValkModelConfig(
        vocab_size=97,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=3,
        head_dim=16,
        num_v_heads=3,
        intermediate_size=128,
        use_latent_state=True,
        latent_state_dim=32,
    )
    predictor = 24 * (4 * 24) + (4 * 24) + (4 * 24) * 24
    expected = 32 * 24 + predictor + 32 * 24

    assert config.estimate_parameters() - without_jepa.estimate_parameters() == expected


def test_parameter_estimate_counts_each_latent_branching_layer_exactly():
    config = ValkModelConfig(
        vocab_size=97,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=3,
        head_dim=16,
        num_v_heads=3,
        intermediate_size=128,
        use_latent_state=True,
        latent_state_dim=32,
        use_latent_branching=True,
        enable_unstable_latent_branching=True,
        latent_branching_layers=[0, 1],
        num_branches=3,
    )
    without_branching = ValkModelConfig(
        vocab_size=97,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=3,
        head_dim=16,
        num_v_heads=3,
        intermediate_size=128,
        use_latent_state=True,
        latent_state_dim=32,
    )
    branch_per_layer = 3 * 32 * 32 + 3 * 32 + 3

    assert config.estimate_parameters() - without_branching.estimate_parameters() == 2 * branch_per_layer
