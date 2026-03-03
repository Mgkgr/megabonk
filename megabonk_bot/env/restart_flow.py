from __future__ import annotations


def should_skip_restart(*, now: float, last_restart_ts: float, cooldown_s: float) -> bool:
    return (float(now) - float(last_restart_ts)) < max(0.0, float(cooldown_s))


def is_restart_timeout(*, now: float, started_ts: float, timeout_s: float) -> bool:
    if float(started_ts) <= 0.0:
        return False
    return (float(now) - float(started_ts)) >= max(0.0, float(timeout_s))
