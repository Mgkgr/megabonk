from dataclasses import dataclass
from types import SimpleNamespace

from run_runtime_bot import build_runtime_event


@dataclass
class _FakeAction:
    dir_id: int = 1
    yaw: int = 1
    jump: int = 0
    slide: int = 0
    press_space: bool = False
    press_r: bool = False
    press_tab: bool = False
    reason: str = "test"


def _make_snapshot():
    return SimpleNamespace(
        time_s=10,
        hp_ratio=0.5,
        lvl=2,
        kills=3,
        enemies=[],
        obstacles=[],
        projectiles=[],
        interactables=[],
        is_dead=False,
        is_upgrade=False,
    )


def test_runtime_event_v2_contains_capture_and_hud_blocks():
    event = build_runtime_event(
        ts=1.0,
        mode=SimpleNamespace(value="ACTIVE"),
        frame_id=1,
        screen="RUNNING",
        snapshot=_make_snapshot(),
        hud_values={"gold": 77},
        action=_FakeAction(),
        action_reason="test",
        restart_event=None,
        safe_sector="center",
        boss_prep=False,
        boss_name=None,
        preferred_direction="center",
        threats=[],
        loop_start=0.0,
        step_hz=12,
        dt=1 / 12,
        window_title="Megabonk",
        frame_width=1280,
        frame_height=720,
        capture_bad_grab_count=4,
        capture_last_error="RuntimeError: fail",
        hud_debug_dumped=True,
        hud_fail_streak=9,
    )
    assert event["schema_version"] == "runtime_events_v2"
    assert event["capture"]["bad_grab_count"] == 4
    assert event["capture"]["last_error"] == "RuntimeError: fail"
    assert event["hud"]["debug_dumped"] is True
    assert event["hud"]["fail_streak"] == 9


def test_runtime_event_v1_does_not_emit_v2_blocks():
    event = build_runtime_event(
        ts=1.0,
        mode=SimpleNamespace(value="ACTIVE"),
        frame_id=1,
        screen="RUNNING",
        snapshot=_make_snapshot(),
        hud_values={"gold": 77},
        action=_FakeAction(),
        action_reason="test",
        restart_event=None,
        safe_sector="center",
        boss_prep=False,
        boss_name=None,
        preferred_direction="center",
        threats=[],
        loop_start=0.0,
        step_hz=12,
        dt=1 / 12,
        window_title="Megabonk",
        frame_width=1280,
        frame_height=720,
        schema_version="runtime_events_v1",
    )
    assert event["schema_version"] == "runtime_events_v1"
    assert "capture" not in event
    assert "hud" not in event
