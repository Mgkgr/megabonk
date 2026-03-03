from megabonk_bot.runtime.recovery import RecoveryState, decide_recovery_action


def test_recovery_running_restored_resets_state():
    state = RecoveryState(attempts=2, started_ts=100.0, last_restart_ts=90.0)
    decision = decide_recovery_action(
        now=120.0,
        screen="RUNNING",
        is_dead=False,
        press_restart=False,
        state=state,
        restart_cooldown_s=3.5,
        restart_max_attempts=2,
        restart_wait_timeout_s=8.0,
    )
    assert decision.running_restored is True
    assert decision.restart_event == "running_restored"
    assert state.attempts == 0
    assert state.started_ts == 0.0


def test_recovery_holds_restart_when_allowed():
    state = RecoveryState(attempts=0, started_ts=100.0, last_restart_ts=90.0)
    decision = decide_recovery_action(
        now=100.0,
        screen="DEAD",
        is_dead=True,
        press_restart=True,
        state=state,
        restart_cooldown_s=3.5,
        restart_max_attempts=2,
        restart_wait_timeout_s=8.0,
    )
    assert decision.running_restored is False
    assert decision.hold_restart is True
    assert decision.restart_event == "hold_r_attempt_1"
    assert state.attempts == 1
    assert state.last_restart_ts == 100.0


def test_recovery_requests_fallback_after_timeout():
    state = RecoveryState(attempts=2, started_ts=10.0, last_restart_ts=19.0)
    decision = decide_recovery_action(
        now=20.0,
        screen="DEAD",
        is_dead=True,
        press_restart=True,
        state=state,
        restart_cooldown_s=3.5,
        restart_max_attempts=2,
        restart_wait_timeout_s=8.0,
    )
    assert decision.hold_restart is False
    assert decision.try_fallback_click is True
