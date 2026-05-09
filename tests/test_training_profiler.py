import math

from valkmodel.training import TrainingProfiler


def test_training_profiler_aggregates_metrics_by_context_length():
    profiler = TrainingProfiler()
    profiler.log_step(step=0, loss=2.0, grad_norm=1.0, latent_state_norm=0.5, memory_mb=100.0, tokens_per_sec=1000.0, context_length=8)
    profiler.log_step(step=1, loss=1.0, grad_norm=3.0, latent_state_norm=1.5, memory_mb=200.0, tokens_per_sec=500.0, context_length=8)
    profiler.log_step(step=2, loss=4.0, grad_norm=2.0, latent_state_norm=2.0, memory_mb=300.0, tokens_per_sec=250.0, context_length=32)

    summary = profiler.summarize_by_context_length()

    assert summary[8]["steps"] == 2
    assert summary[8]["loss_mean"] == 1.5
    assert summary[8]["grad_norm_mean"] == 2.0
    assert summary[8]["memory_mb_peak"] == 200.0
    assert summary[8]["tokens_per_sec_mean"] == 750.0
    assert summary[32]["steps"] == 1


def test_training_profiler_stores_optional_health_metrics_and_summarizes_them():
    profiler = TrainingProfiler()
    profiler.log_step(
        step=0,
        loss=2.0,
        grad_norm=1.0,
        latent_state_norm=0.5,
        memory_mb=100.0,
        tokens_per_sec=1000.0,
        context_length=8,
        learning_rate=1e-4,
        perplexity=7.0,
        jepa_loss=0.2,
        jepa_prediction_variance=0.3,
        jepa_target_variance=0.4,
        jepa_cosine_mean=0.5,
        branch_entropy_mean=0.6,
        branch_diversity_loss_mean=0.7,
        branch_variance_mean=0.8,
    )

    record = profiler.records[0]
    summary = profiler.summarize_by_context_length()

    assert record.learning_rate == 1e-4
    assert record.jepa_loss == 0.2
    assert summary[8]["jepa_loss_mean"] == 0.2
    assert summary[8]["branch_entropy_mean"] == 0.6


def test_training_profiler_reports_anomalies_for_unstable_metrics():
    profiler = TrainingProfiler(max_grad_norm=10.0, min_latent_variance=1e-4, min_tokens_per_sec=100.0)
    profiler.log_step(
        step=0,
        loss=math.nan,
        grad_norm=11.0,
        latent_state_norm=0.0,
        memory_mb=100.0,
        tokens_per_sec=50.0,
        context_length=8,
        latent_state_variance=0.0,
        perplexity=math.inf,
        jepa_prediction_variance=0.0,
        jepa_target_variance=0.0,
        branch_entropy_mean=0.0,
    )

    anomalies = profiler.detect_anomalies()

    assert any("loss is not finite" in anomaly for anomaly in anomalies)
    assert any("grad_norm" in anomaly for anomaly in anomalies)
    assert any("latent variance" in anomaly for anomaly in anomalies)
    assert any("throughput" in anomaly for anomaly in anomalies)
    assert any("perplexity" in anomaly for anomaly in anomalies)
    assert any("JEPA prediction variance" in anomaly for anomaly in anomalies)
    assert any("JEPA target variance" in anomaly for anomaly in anomalies)
    assert any("branch entropy" in anomaly for anomaly in anomalies)
