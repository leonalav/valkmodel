from valkmodel import ValkModelConfig
from valkmodel.training import HardwareProfile, ScalingValidator


def test_scaling_validator_memory_estimate_increases_with_context_and_model_size():
    validator = ScalingValidator()
    small = ValkModelConfig.from_preset("130m")
    large = ValkModelConfig.from_preset("260m")

    small_8k = validator.estimate_training_memory(small, batch_size=1, seq_len=8192)
    small_32k = validator.estimate_training_memory(small, batch_size=1, seq_len=32768)
    large_8k = validator.estimate_training_memory(large, batch_size=1, seq_len=8192)

    assert small_8k["total_gb"] > 0
    assert small_32k["total_gb"] > small_8k["total_gb"]
    assert large_8k["total_gb"] > small_8k["total_gb"]


def test_scaling_validator_estimates_gdn_cache_separately_from_activations():
    validator = ScalingValidator()
    config = ValkModelConfig.from_preset("130m")

    cache_gb = validator.estimate_gdn_cache_memory(config, batch_size=2, dtype_bytes=2)

    expected_bytes = config.num_hidden_layers * 2 * config.num_v_heads * config.head_v_dim * 2
    assert cache_gb == expected_bytes / 1024**3


def test_scaling_validator_computes_mfu_from_tokens_per_second_and_parameters():
    validator = ScalingValidator(gpu_peak_tflops={"test_gpu": 100.0})

    mfu = validator.compute_mfu(num_parameters=1_000_000_000, tokens_per_sec=10_000, gpu_type="test_gpu")

    assert mfu == 6 * 1_000_000_000 * 10_000 / (100.0 * 1_000_000_000_000)


def test_scaling_validator_detects_non_monotonic_scaling_curve():
    validator = ScalingValidator()
    good = [
        HardwareProfile("130m", 130_000_000, 8192, 1, 10.0, 1000.0, 0.1, "h100", eval_loss=3.5),
        HardwareProfile("260m", 260_000_000, 8192, 1, 20.0, 800.0, 0.12, "h100", eval_loss=3.2),
    ]
    bad = [
        HardwareProfile("130m", 130_000_000, 8192, 1, 10.0, 1000.0, 0.1, "h100", eval_loss=3.1),
        HardwareProfile("260m", 260_000_000, 8192, 1, 20.0, 800.0, 0.12, "h100", eval_loss=3.2),
    ]

    assert validator.validate_scaling_curve(good)["eval_loss_monotonic"]
    assert not validator.validate_scaling_curve(bad)["eval_loss_monotonic"]


def test_scaling_validator_recommends_fit_status_for_target_hardware():
    validator = ScalingValidator(gpu_memory_gb={"tiny_gpu": 1.0, "large_gpu": 10_000.0})
    config = ValkModelConfig.from_preset("130m")

    tiny = validator.recommend_training_config(config, context_length=8192, gpu_type="tiny_gpu")
    large = validator.recommend_training_config(config, context_length=8192, gpu_type="large_gpu")

    assert not tiny["fits"]
    assert large["fits"]
    assert large["max_batch_size"] >= 1
