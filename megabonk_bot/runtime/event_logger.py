from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

RUNTIME_EVENT_SCHEMA_VERSION = "runtime_events_v4"
LOGGER = logging.getLogger(__name__)


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
            LOGGER.warning("Failed to close runtime event log %s", self.path, exc_info=True)


def _serialize_detection_list(items) -> list[dict[str, Any]]:
    output = []
    for item in items:
        if hasattr(item, "label") and hasattr(item, "rect") and hasattr(item, "score"):
            payload = {
                "label": getattr(item, "label"),
                "rect": list(getattr(item, "rect")),
                "score": float(getattr(item, "score")),
            }
            if hasattr(item, "source"):
                payload["source"] = getattr(item, "source")
            if hasattr(item, "entity_id") and getattr(item, "entity_id") is not None:
                payload["entity_id"] = getattr(item, "entity_id")
            if hasattr(item, "threat_tier"):
                payload["threat_tier"] = float(getattr(item, "threat_tier") or 0.0)
            if hasattr(item, "poi_type") and getattr(item, "poi_type") is not None:
                payload["poi_type"] = getattr(item, "poi_type")
            if hasattr(item, "family") and getattr(item, "family") is not None:
                payload["family"] = getattr(item, "family")
            if hasattr(item, "variant") and getattr(item, "variant") is not None:
                payload["variant"] = getattr(item, "variant")
            if hasattr(item, "hazard_kind") and getattr(item, "hazard_kind") is not None:
                payload["hazard_kind"] = getattr(item, "hazard_kind")
            if hasattr(item, "icon_id") and getattr(item, "icon_id") is not None:
                payload["icon_id"] = getattr(item, "icon_id")
            metadata = getattr(item, "metadata", None)
            if isinstance(metadata, dict) and metadata:
                payload["metadata"] = metadata
            output.append(payload)
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


def _serialize_player_pose(snapshot) -> dict[str, Any] | None:
    pose = getattr(snapshot, "player_pose", None)
    if pose is None:
        return None
    return {
        "map_norm": list(pose.map_norm) if pose.map_norm is not None else None,
        "world_pos": list(pose.world_pos) if pose.world_pos is not None else None,
        "heading_deg": pose.heading_deg,
        "source": pose.source,
        "confidence": pose.confidence,
    }


def _serialize_map_state(snapshot) -> dict[str, Any] | None:
    state = getattr(snapshot, "map_state", None)
    if state is None:
        return None
    return {
        "map_open": bool(state.map_open),
        "minimap_visible": bool(state.minimap_visible),
        "player_norm": list(state.player_norm) if state.player_norm is not None else None,
        "biome": state.biome,
        "scene_id": state.scene_id,
        "active_room_id": state.active_room_id,
        "room_start": list(state.room_start) if state.room_start is not None else None,
        "room_end": list(state.room_end) if state.room_end is not None else None,
        "is_crypt": state.is_crypt,
        "objective": state.objective,
        "objective_confidence": state.objective_confidence,
        "boss_spotted": state.boss_spotted,
        "charged_shrines": state.charged_shrines,
        "graveyard_crypt_keys": state.graveyard_crypt_keys,
        "pois": [
            {
                "label": poi.label,
                "pos_norm": list(poi.pos_norm),
                "score": float(poi.score),
                "source": poi.source,
                "poi_type": poi.poi_type,
                "icon_id": poi.icon_id,
            }
            for poi in state.pois
        ],
        "source": state.source,
        "confidence": state.confidence,
    }


def _serialize_navigation_context(navigation_context) -> dict[str, Any] | None:
    if navigation_context is None:
        return None
    lanes = []
    for lane in getattr(navigation_context, "lanes", ()) or ():
        lanes.append(
            {
                "index": int(getattr(lane, "index", 0)),
                "label": getattr(lane, "label", "lane"),
                "x0": int(getattr(lane, "x0", 0)),
                "x1": int(getattr(lane, "x1", 0)),
                "threat_score": float(getattr(lane, "threat_score", 0.0)),
                "obstacle_cost": float(getattr(lane, "obstacle_cost", 0.0)),
                "drop_risk": float(getattr(lane, "drop_risk", 0.0)),
                "clearance": float(getattr(lane, "clearance", 0.0)),
                "landing_clearance": float(getattr(lane, "landing_clearance", 0.0)),
                "terrain_kind": getattr(lane, "terrain_kind", "unknown"),
                "total_cost": float(getattr(lane, "total_cost", 0.0)),
            }
        )
    return {
        "terrain_kind": getattr(navigation_context, "terrain_kind", "unknown"),
        "drop_risk": float(getattr(navigation_context, "drop_risk", 0.0)),
        "obstacle_cost": float(getattr(navigation_context, "obstacle_cost", 0.0)),
        "clearance": float(getattr(navigation_context, "clearance", 0.0)),
        "nav_confidence": float(getattr(navigation_context, "nav_confidence", 0.0)),
        "source": getattr(navigation_context, "source", "cv"),
        "slope_source": getattr(navigation_context, "slope_source", "none"),
        "slope_delta_z": getattr(navigation_context, "slope_delta_z", None),
        "escape_lane": getattr(navigation_context, "escape_lane", None),
        "jump_gate": getattr(navigation_context, "jump_gate", "not_evaluated"),
        "slide_gate": getattr(navigation_context, "slide_gate", "not_evaluated"),
        "lanes": lanes,
    }


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
    navigation_context=None,
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
    if schema_version in {"runtime_events_v3", "runtime_events_v4"}:
        event["detections"]["enemy_classes"] = _serialize_detection_list(getattr(snapshot, "enemy_classes", []))
        event["detections"]["projectile_classes"] = _serialize_detection_list(getattr(snapshot, "projectile_classes", []))
        event["detections"]["world_objects"] = _serialize_detection_list(getattr(snapshot, "world_objects", []))
        event["detections"]["hazards"] = _serialize_detection_list(getattr(snapshot, "hazards", []))
        event["player_pose"] = _serialize_player_pose(snapshot)
        event["map_state"] = _serialize_map_state(snapshot)
        event["memory_probe_status"] = getattr(snapshot, "memory_probe_status", "disabled")
        event["detection_sources"] = dict(getattr(snapshot, "detection_sources", {}) or {})
        event["source_confidence"] = {
            str(key): float(value)
            for key, value in dict(getattr(snapshot, "source_confidence", {}) or {}).items()
        }
    if schema_version == "runtime_events_v4":
        event["navigation"] = _serialize_navigation_context(navigation_context)
    return event
