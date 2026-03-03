from __future__ import annotations

import random


def current_safety_strength(
    *,
    safety_enabled: bool,
    safety_strength: float,
    safety_anneal_steps: int,
    safety_min_strength: float,
    safety_step_count: int,
) -> float:
    if not safety_enabled:
        return 0.0
    base = max(0.0, float(safety_strength))
    if int(safety_anneal_steps) <= 0:
        return base
    progress = min(1.0, float(safety_step_count) / float(safety_anneal_steps))
    strength = base * (1.0 - progress)
    return max(float(safety_min_strength), strength)


def apply_safety_override(
    *,
    dir_id: int,
    yaw: int,
    pitch: int,
    jump: int,
    slide: int,
    danger_now: bool,
    stuck_now: bool,
    strength: float,
) -> tuple[int, int, int, int, int, str | None]:
    if strength <= 0.0:
        return dir_id, yaw, pitch, jump, slide, None
    if not (danger_now or stuck_now):
        return dir_id, yaw, pitch, jump, slide, None
    if random.random() > float(strength):
        return dir_id, yaw, pitch, jump, slide, None
    if danger_now:
        return 2, 1, 1, 1, 0, "danger"
    return 0, 1, 1, 1, 0, "stuck"
