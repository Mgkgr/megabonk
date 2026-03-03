import csv
import json
import subprocess
import sys
from pathlib import Path


def test_export_jsonl_supports_v1_and_v2(tmp_path):
    input_path = tmp_path / "events.jsonl"
    out_path = tmp_path / "events.csv"
    events = [
        {
            "schema_version": "runtime_events_v1",
            "ts": 1.0,
            "mode": "ACTIVE",
            "frame_id": 1,
            "screen": "RUNNING",
            "step_hz": 12,
            "dt_ms": 83.3,
            "window_title": "Megabonk",
            "frame_size": {"width": 1280, "height": 720},
            "telemetry": {"time": 10, "gold": 100, "lvl": 2, "kills": 1, "hp_ratio": 0.5},
            "latency_ms": 7.0,
            "is_dead": False,
            "is_upgrade": False,
            "safe_sector": "center",
            "reason": "test_v1",
            "restart": None,
            "boss_prep": False,
            "boss_name": None,
            "preferred_direction": "center",
        },
        {
            "schema_version": "runtime_events_v2",
            "ts": 2.0,
            "mode": "ACTIVE",
            "frame_id": 2,
            "screen": "RUNNING",
            "step_hz": 12,
            "dt_ms": 83.3,
            "window_title": "Megabonk",
            "frame_size": {"width": 1280, "height": 720},
            "telemetry": {"time": 11, "gold": 120, "lvl": 3, "kills": 2, "hp_ratio": 0.6},
            "latency_ms": 8.0,
            "is_dead": False,
            "is_upgrade": False,
            "safe_sector": "left",
            "reason": "test_v2",
            "restart": None,
            "boss_prep": False,
            "boss_name": None,
            "preferred_direction": "left",
            "capture": {"bad_grab_count": 4, "last_error": "RuntimeError: fail"},
            "hud": {"debug_dumped": True, "fail_streak": 9},
        },
    ]
    input_path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_jsonl.py"
    subprocess.run(
        [sys.executable, str(script_path), str(input_path), "-o", str(out_path)],
        check=True,
    )

    with out_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))

    assert len(rows) == 2
    assert rows[0]["schema_version"] == "runtime_events_v1"
    assert rows[0]["capture_bad_grab_count"] == ""
    assert rows[1]["schema_version"] == "runtime_events_v2"
    assert rows[1]["capture_bad_grab_count"] == "4"
    assert rows[1]["capture_last_error"] == "RuntimeError: fail"
    assert rows[1]["hud_debug_dumped"] == "True"
    assert rows[1]["hud_fail_streak"] == "9"
