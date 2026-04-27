from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence

PERFORMANCE_STAGE_NAMES = ("capture", "hud", "scene_analysis", "overlay")
DEFAULT_PERFORMANCE_BUDGET_MS = {
    "capture": 8.0,
    "hud": 2.0,
    "scene_analysis": 45.0,
    "overlay": 18.0,
}


def normalize_performance_budget_ms(
    raw_budget: Mapping[str, Any] | None,
    *,
    tick_budget_ms: float,
) -> dict[str, float]:
    budget = dict(DEFAULT_PERFORMANCE_BUDGET_MS)
    if raw_budget is not None:
        unknown = set(raw_budget) - set(PERFORMANCE_STAGE_NAMES)
        if unknown:
            raise ValueError(f"Unknown performance budget stages: {sorted(unknown)}")
        for name, value in raw_budget.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"performance budget {name} must be a number")
            if float(value) < 0.0:
                raise ValueError(f"performance budget {name} must be >= 0")
            budget[str(name)] = float(value)
    if float(tick_budget_ms) <= 0.0:
        raise ValueError("tick performance budget must be > 0")
    budget["tick"] = float(tick_budget_ms)
    return budget


def _round_ms(value: float) -> float:
    return round(float(value), 3)


def _round_ms_map(values: Mapping[str, float]) -> dict[str, float]:
    return {str(key): _round_ms(value) for key, value in values.items()}


def build_performance_sample(
    stage_timings_ms: Mapping[str, float],
    *,
    budget_ms: Mapping[str, float],
    tick_ms: float,
) -> dict[str, Any]:
    stages = {
        name: max(0.0, float(stage_timings_ms.get(name, 0.0)))
        for name in PERFORMANCE_STAGE_NAMES
    }
    stages["tick"] = max(0.0, float(tick_ms))
    budget = {
        name: float(budget_ms[name])
        for name in (*PERFORMANCE_STAGE_NAMES, "tick")
        if name in budget_ms
    }
    over_budget = {
        name: stages[name] - limit
        for name, limit in budget.items()
        if stages.get(name, 0.0) > limit
    }
    return {
        "stages_ms": _round_ms_map(stages),
        "budget_ms": _round_ms_map(budget),
        "over_budget_ms": _round_ms_map(over_budget),
        "within_budget": not over_budget,
    }


def format_over_budget(sample: Mapping[str, Any]) -> str:
    over_budget = sample.get("over_budget_ms")
    if not isinstance(over_budget, Mapping) or not over_budget:
        return "within budget"
    return ", ".join(
        f"{name}=+{float(delta):.1f}ms"
        for name, delta in sorted(over_budget.items())
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_performance_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0, "stages": {}}
    stage_names = (*PERFORMANCE_STAGE_NAMES, "tick")
    summary: dict[str, Any] = {"sample_count": len(samples), "stages": {}}
    for name in stage_names:
        values = []
        budget_value = None
        over_budget_count = 0
        for sample in samples:
            stages = sample.get("stages_ms")
            if isinstance(stages, Mapping) and name in stages:
                values.append(float(stages[name]))
            budgets = sample.get("budget_ms")
            if isinstance(budgets, Mapping) and name in budgets:
                budget_value = float(budgets[name])
            over_budget = sample.get("over_budget_ms")
            if isinstance(over_budget, Mapping) and name in over_budget:
                over_budget_count += 1
        if not values:
            continue
        summary["stages"][name] = {
            "avg_ms": _round_ms(mean(values)),
            "p95_ms": _round_ms(_percentile(values, 95.0)),
            "max_ms": _round_ms(max(values)),
            "budget_ms": _round_ms(budget_value) if budget_value is not None else None,
            "over_budget_count": over_budget_count,
        }
    return summary
