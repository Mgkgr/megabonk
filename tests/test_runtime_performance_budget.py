from megabonk_bot.runtime.perf_budget import (
    build_performance_sample,
    format_over_budget,
    normalize_performance_budget_ms,
    summarize_performance_samples,
)


def test_performance_sample_flags_stage_and_tick_over_budget():
    budget = normalize_performance_budget_ms(
        {
            "capture": 5.0,
            "hud": 1.0,
            "scene_analysis": 20.0,
            "overlay": 10.0,
        },
        tick_budget_ms=33.0,
    )
    sample = build_performance_sample(
        {
            "capture": 6.25,
            "hud": 0.2,
            "scene_analysis": 19.0,
            "overlay": 10.0,
        },
        budget_ms=budget,
        tick_ms=34.5,
    )

    assert sample["within_budget"] is False
    assert sample["over_budget_ms"] == {
        "capture": 1.25,
        "tick": 1.5,
    }
    assert format_over_budget(sample) == "capture=+1.2ms, tick=+1.5ms"


def test_performance_summary_reports_baseline_stats():
    budget = normalize_performance_budget_ms(None, tick_budget_ms=83.333)
    samples = [
        build_performance_sample(
            {
                "capture": 1.0,
                "hud": 0.1,
                "scene_analysis": 10.0,
                "overlay": 3.0,
            },
            budget_ms=budget,
            tick_ms=20.0,
        ),
        build_performance_sample(
            {
                "capture": 3.0,
                "hud": 0.3,
                "scene_analysis": 20.0,
                "overlay": 5.0,
            },
            budget_ms=budget,
            tick_ms=40.0,
        ),
    ]

    summary = summarize_performance_samples(samples)

    assert summary["sample_count"] == 2
    assert summary["stages"]["capture"]["avg_ms"] == 2.0
    assert summary["stages"]["capture"]["max_ms"] == 3.0
    assert summary["stages"]["tick"]["budget_ms"] == 83.333
