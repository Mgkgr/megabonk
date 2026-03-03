from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

RUNTIME_EVENT_SCHEMA_VERSION = "runtime_events_v2"


class JsonlEventLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        self._fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def _serialize_detection_list(items) -> list[dict[str, Any]]:
    output = []
    for item in items:
        if hasattr(item, "label") and hasattr(item, "rect") and hasattr(item, "score"):
            output.append(
                {
                    "label": getattr(item, "label"),
                    "rect": list(getattr(item, "rect")),
                    "score": float(getattr(item, "score")),
                }
            )
    return output


def _extract_hud_debug(hud_values: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(hud_values, dict):
        return {}
    debug = {}
    for key, value in hud_values.items():
        if (
            key == "tesseract_cmd"
            or key.endswith("_fail_reason")
            or key.endswith("_ocr_ms")
            or key.endswith("_rect")
        ):
            debug[key] = value
    return debug


def build_runtime_event(
    *,
    ts: float,
    mode,
    frame_id: int,
    screen: str,
    snapshot,
    hud_values: dict[str, Any],
    action,
    action_reason: str,
    restart_event: Optional[str],
    safe_sector: str,
    boss_prep: bool,
    boss_name: Optional[str],
    preferred_direction: str,
    threats,
    loop_start: float,
    step_hz: int,
    dt: float,
    window_title: str,
    frame_width: int,
    frame_height: int,
    capture_bad_grab_count: int = 0,
    capture_last_error: Optional[str] = None,
    hud_debug_dumped: bool = False,
    hud_fail_streak: int = 0,
    schema_version: str = RUNTIME_EVENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    event = {
        "schema_version": schema_version,
        "ts": ts,
        "mode": mode.value,
        "frame_id": int(frame_id),
        "step_hz": int(step_hz),
        "dt_ms": float(dt) * 1000.0,
        "window_title": str(window_title),
        "frame_size": {"width": int(frame_width), "height": int(frame_height)},
        "screen": screen,
        "telemetry": {
            "time": getattr(snapshot, "time_s", None),
            "hp_ratio": getattr(snapshot, "hp_ratio", None),
            "lvl": getattr(snapshot, "lvl", None),
            "kills": getattr(snapshot, "kills", None),
            "gold": hud_values.get("gold"),
        },
        "telemetry_raw": _extract_hud_debug(hud_values),
        "detections": {
            "enemies": _serialize_detection_list(getattr(snapshot, "enemies", [])),
            "obstacles": _serialize_detection_list(getattr(snapshot, "obstacles", [])),
            "projectiles": _serialize_detection_list(getattr(snapshot, "projectiles", [])),
            "interactables": _serialize_detection_list(getattr(snapshot, "interactables", [])),
        },
        "action": asdict(action),
        "reason": action_reason,
        "restart": restart_event,
        "safe_sector": safe_sector,
        "is_dead": bool(getattr(snapshot, "is_dead", False)),
        "is_upgrade": bool(getattr(snapshot, "is_upgrade", False)),
        "boss_prep": bool(boss_prep),
        "boss_name": boss_name,
        "preferred_direction": preferred_direction,
        "top_threats": [
            {
                "label": threat.label,
                "rect": list(threat.rect),
                "priority": threat.priority,
                "distance_norm": threat.distance_norm,
            }
            for threat in threats[:3]
        ],
        "latency_ms": (time.perf_counter() - loop_start) * 1000.0,
    }
    if schema_version != "runtime_events_v1":
        event["capture"] = {
            "bad_grab_count": int(capture_bad_grab_count),
            "last_error": capture_last_error,
        }
        event["hud"] = {
            "debug_dumped": bool(hud_debug_dumped),
            "fail_streak": int(hud_fail_streak),
        }
    return event
