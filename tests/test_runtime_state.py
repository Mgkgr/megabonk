from megabonk_bot.runtime_logic import BotMode
from megabonk_bot.runtime_state import RuntimeStateMachine


def test_toggle_transitions_off_active_off():
    sm = RuntimeStateMachine(mode=BotMode.OFF)
    assert sm.on_events(toggle=True) == BotMode.ACTIVE
    assert sm.on_events(toggle=True) == BotMode.OFF


def test_panic_and_recover_via_toggle():
    sm = RuntimeStateMachine(mode=BotMode.ACTIVE)
    assert sm.on_events(panic=True) == BotMode.PANIC
    assert sm.on_events(toggle=True) == BotMode.ACTIVE


def test_recovery_flow():
    sm = RuntimeStateMachine(mode=BotMode.ACTIVE)
    assert sm.on_events(dead_detected=True) == BotMode.RECOVERY
    assert sm.on_events(running_restored=True) == BotMode.ACTIVE
