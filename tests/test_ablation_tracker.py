from valkmodel.training import AblationResult, AblationTracker


def test_ablation_tracker_compares_variants_against_baseline():
    tracker = AblationTracker()
    tracker.log_result(AblationResult("baseline_gdn", "130m", eval_loss=3.0, tool_benchmark_score=0.5, general_benchmark_score=0.7, training_tokens=1_000))
    tracker.log_result(AblationResult("full", "130m", eval_loss=2.7, tool_benchmark_score=0.65, general_benchmark_score=0.69, training_tokens=1_000))

    comparison = tracker.compare_variants("130m", baseline_variant="baseline_gdn")

    assert comparison["full"]["eval_loss_delta"] == -0.3
    assert comparison["full"]["tool_benchmark_score_delta"] == 0.15
    assert comparison["full"]["general_benchmark_score_delta"] == -0.01


def test_ablation_tracker_detects_general_metric_and_loss_regressions():
    tracker = AblationTracker()
    tracker.log_result(AblationResult("baseline_gdn", "130m", eval_loss=3.0, tool_benchmark_score=0.5, general_benchmark_score=0.7, training_tokens=1_000))
    tracker.log_result(AblationResult("branch", "130m", eval_loss=3.3, tool_benchmark_score=0.8, general_benchmark_score=0.5, training_tokens=1_000))

    regressions = tracker.detect_regressions(baseline_variant="baseline_gdn", threshold=0.05)

    assert any("eval_loss" in regression for regression in regressions)
    assert any("general_benchmark_score" in regression for regression in regressions)
    assert not any("tool_benchmark_score" in regression for regression in regressions)
