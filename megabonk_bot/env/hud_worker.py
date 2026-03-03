from __future__ import annotations

from dataclasses import dataclass

VALID_HUD_DEBUG_POLICIES = {"startup", "on_fail_change", "interval", "off"}


@dataclass
class HudDumpPolicyState:
    startup_dump_done: bool = False
    last_dump_reason: str | None = None
    last_dump_ts: float = 0.0
    fail_streak: int = 0


def normalize_hud_debug_policy(value: str | None) -> str:
    normalized = str(value or "on_fail_change").strip().lower()
    if normalized not in VALID_HUD_DEBUG_POLICIES:
        return "on_fail_change"
    return normalized


def should_dump_hud_debug(
    *,
    state: HudDumpPolicyState,
    policy: str,
    now: float,
    fail_reason: str | None,
    min_interval_s: float,
    startup: bool = False,
) -> bool:
    mode = normalize_hud_debug_policy(policy)
    if startup:
        if state.startup_dump_done:
            return False
        state.startup_dump_done = True
        if mode == "off":
            return False
        state.last_dump_reason = "startup"
        state.last_dump_ts = now
        return mode in {"startup", "on_fail_change", "interval"}

    if fail_reason is None:
        state.fail_streak = 0
        return False

    state.fail_streak += 1
    if mode in {"off", "startup"}:
        return False

    min_interval = max(0.0, float(min_interval_s))
    if mode == "interval":
        if (now - state.last_dump_ts) >= min_interval:
            state.last_dump_reason = str(fail_reason)
            state.last_dump_ts = now
            return True
        return False

    # on_fail_change:
    if state.fail_streak == 1 or state.last_dump_reason != str(fail_reason):
        state.last_dump_reason = str(fail_reason)
        state.last_dump_ts = now
        return True
    if (now - state.last_dump_ts) >= min_interval:
        state.last_dump_reason = str(fail_reason)
        state.last_dump_ts = now
        return True
    return False
