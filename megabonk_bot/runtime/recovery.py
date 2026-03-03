from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecoveryState:
    attempts: int = 0
    started_ts: float = 0.0
    last_restart_ts: float = 0.0


@dataclass
class RecoveryDecision:
    running_restored: bool
    hold_restart: bool
    try_fallback_click: bool
    restart_event: str | None


def decide_recovery_action(
    *,
    now: float,
    screen: str,
    is_dead: bool,
    press_restart: bool,
    state: RecoveryState,
    restart_cooldown_s: float,
    restart_max_attempts: int,
    restart_wait_timeout_s: float,
) -> RecoveryDecision:
    if screen == "RUNNING" and not is_dead:
        state.attempts = 0
        state.started_ts = 0.0
        return RecoveryDecision(
            running_restored=True,
            hold_restart=False,
            try_fallback_click=False,
            restart_event="running_restored",
        )

    hold_restart = False
    restart_event = None
    if (
        press_restart
        and (now - state.last_restart_ts) >= restart_cooldown_s
        and state.attempts < max(1, int(restart_max_attempts))
    ):
        state.last_restart_ts = now
        state.attempts += 1
        hold_restart = True
        restart_event = f"hold_r_attempt_{state.attempts}"

    try_fallback_click = (
        state.started_ts > 0.0
        and (now - state.started_ts) >= max(0.0, float(restart_wait_timeout_s))
    )

    return RecoveryDecision(
        running_restored=False,
        hold_restart=hold_restart,
        try_fallback_click=try_fallback_click,
        restart_event=restart_event,
    )
