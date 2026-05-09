import torch

from valkmodel import ValkModelConfig, ValkModelForCausalLM
from valkmodel.losses import compute_weighted_lm_loss
from valkmodel.utils.tool_masks import create_tool_mask


def tiny_config(**overrides):
    values = {
        "vocab_size": 97,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_heads": 3,
        "head_dim": 16,
        "num_v_heads": 3,
        "intermediate_size": 128,
        "max_position_embeddings": 128,
        "use_short_conv": True,
        "gdn_backend": "naive",
    }
    values.update(overrides)
    return ValkModelConfig(**values)


def test_create_tool_mask_marks_call_result_and_reasoning_regions():
    input_ids = torch.tensor([[1, 10, 2, 3, 20, 4, 30, 5, 31, 6]])

    mask = create_tool_mask(
        input_ids,
        tool_call_token_id=10,
        tool_result_token_id=20,
        reasoning_start_token_id=30,
        reasoning_end_token_id=31,
        tool_call_span=2,
        tool_result_span=1,
    )

    expected = torch.tensor([[False, True, True, True, True, True, True, True, True, False]])
    assert torch.equal(mask, expected)


def test_create_tool_mask_truncates_spans_at_sequence_end_and_ignores_missing_ids():
    input_ids = torch.tensor([[1, 2, 10]])

    mask = create_tool_mask(
        input_ids,
        tool_call_token_id=10,
        tool_result_token_id=None,
        reasoning_start_token_id=None,
        reasoning_end_token_id=None,
        tool_call_span=8,
        tool_result_span=8,
    )

    expected = torch.tensor([[False, False, True]])
    assert torch.equal(mask, expected)


def test_compute_weighted_lm_loss_matches_cross_entropy_without_tool_mask():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 11)
    labels = torch.randint(0, 11, (2, 4))

    loss = compute_weighted_lm_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, 11), labels[:, 1:].reshape(-1))

    assert torch.allclose(loss, expected)


def test_compute_weighted_lm_loss_applies_shifted_tool_weights_and_ignore_index():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, -1.0],
                [0.0, -1.0, 4.0],
                [1.0, 0.0, 2.0],
            ]
        ]
    )
    labels = torch.tensor([[0, 1, 2, -100]])
    tool_mask = torch.tensor([[False, True, True, True]])

    loss = compute_weighted_lm_loss(logits, labels, tool_mask=tool_mask, tool_weight=3.0, ignore_index=-100)
    per_token = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 3),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(1, 3)
    expected = (per_token[:, :2] * 3.0).sum() / 6.0

    assert torch.allclose(loss, expected)


def test_model_uses_explicit_tool_mask_for_weighted_loss():
    torch.manual_seed(0)
    config = tiny_config(tool_loss_weight=4.0)
    model = ValkModelForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    tool_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    tool_mask[:, 2:4] = True

    outputs = model(input_ids=input_ids, labels=input_ids, tool_mask=tool_mask)
    expected = compute_weighted_lm_loss(
        outputs.logits,
        input_ids,
        tool_mask=tool_mask,
        tool_weight=config.tool_loss_weight,
        ignore_index=-100,
    )

    assert torch.allclose(outputs.loss, expected)


def test_model_auto_generates_tool_mask_from_configured_token_ids():
    torch.manual_seed(0)
    config = tiny_config(tool_call_token_id=10, tool_call_span=2, tool_loss_weight=5.0)
    model = ValkModelForCausalLM(config)
    input_ids = torch.tensor([[4, 10, 11, 12, 13, 14]])

    outputs = model(input_ids=input_ids, labels=input_ids)
    expected_mask = create_tool_mask(input_ids, tool_call_token_id=10, tool_call_span=2)
    expected = compute_weighted_lm_loss(
        outputs.logits,
        input_ids,
        tool_mask=expected_mask,
        tool_weight=config.tool_loss_weight,
        ignore_index=-100,
    )

    assert torch.equal(expected_mask, torch.tensor([[False, True, True, True, False, False]]))
    assert torch.allclose(outputs.loss, expected)
