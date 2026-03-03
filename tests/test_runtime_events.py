import time
from dataclasses import dataclass
from types import SimpleNamespace

from run_runtime_bot import build_runtime_event


@dataclass
class _FakeAction:
    dir_id: int = 1
    yaw: int = 0
    jump: int = 0
    slide: int = 0
    press_space: bool = False
    press_r: bool = False
    press_tab: bool = False
    reason: str = "test"


def test_build_runtime_event_has_schema_and_telemetry_fields():
    snapshot = SimpleNamespace(
        time_s=123.4,
        hp_ratio=0.85,
        lvl=7,
        kills=42,
        enemies=[],
        obstacles=[],
        projectiles=[],
        interactables=[],
        is_dead=False,
        is_upgrade=False,
    )
    action = _FakeAction()
    event = build_runtime_event(
        ts=1_700_000_000.0,
        mode=SimpleNamespace(value="ACTIVE"),
        frame_id=10,
        screen="RUNNING",
        snapshot=snapshot,
        hud_values={
            "gold": 12345,
            "time_fail_reason": "ocr_empty",
            "time_ocr_ms": 12.3,
            "time_rect": (10, 20, 30, 40),
            "gold_fail_reason": None,
            "gold_ocr_ms": 3.2,
            "gold_rect": (11, 21, 31, 41),
            "tesseract_cmd": "/usr/bin/tesseract",
        },
        action=action,
        action_reason=action.reason,
        restart_event=None,
        safe_sector="center",
        boss_prep=False,
        boss_name=None,
        preferred_direction="center",
        threats=[],
        loop_start=time.perf_counter(),
        step_hz=12,
        dt=1.0 / 12.0,
        window_title="Megabonk",
        frame_width=1280,
        frame_height=720,
        schema_version="runtime_events_v1",
    )

    assert event["schema_version"] == "runtime_events_v1"
    assert event["telemetry"]["time"] == 123.4
    assert event["telemetry"]["gold"] == 12345
    assert event["telemetry_raw"]["time_fail_reason"] == "ocr_empty"
    assert event["telemetry_raw"]["time_ocr_ms"] == 12.3
    assert event["telemetry_raw"]["time_rect"] == (10, 20, 30, 40)
    assert event["telemetry_raw"]["gold_ocr_ms"] == 3.2
    assert event["telemetry_raw"]["tesseract_cmd"] == "/usr/bin/tesseract"
