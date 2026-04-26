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
        {
            "schema_version": "runtime_events_v4",
            "ts": 3.0,
            "mode": "ACTIVE",
            "frame_id": 3,
            "screen": "RUNNING",
            "step_hz": 12,
            "dt_ms": 83.3,
            "window_title": "Megabonk",
            "frame_size": {"width": 1280, "height": 720},
            "telemetry": {"time": 12, "gold": 140, "lvl": 4, "kills": 3, "hp_ratio": 0.7},
            "latency_ms": 9.0,
            "is_dead": False,
            "is_upgrade": False,
            "safe_sector": "right",
            "reason": "test_v3",
            "restart": None,
            "boss_prep": False,
            "boss_name": None,
            "preferred_direction": "right",
            "capture": {"bad_grab_count": 0, "last_error": None},
            "hud": {"debug_dumped": False, "fail_streak": 0},
            "memory_probe_status": "ready",
            "player_pose": {
                "map_norm": [0.1, 0.9],
                "world_pos": [1.0, 2.0, 3.0],
                "heading_deg": 87.0,
                "source": "external_memory",
                "confidence": 0.95,
            },
            "map_state": {
                "map_open": True,
                "minimap_visible": True,
                "player_norm": [0.1, 0.9],
                "biome": "forest",
                "scene_id": "GeneratedMap",
                "active_room_id": "room_9",
                "is_crypt": False,
                "objective": "challenge_shrine",
                "objective_confidence": 0.91,
                "boss_spotted": True,
                "charged_shrines": 2,
                "graveyard_crypt_keys": 1,
                "pois": [{"label": "portal", "pos_norm": [0.7, 0.2], "score": 0.9, "source": "ui_cv", "poi_type": "exit", "icon_id": "portal"}],
                "source": "external_memory",
                "confidence": 0.95,
            },
            "navigation": {
                "terrain_kind": "downhill",
                "drop_risk": 0.12,
                "obstacle_cost": 0.21,
                "clearance": 0.84,
                "nav_confidence": 0.93,
                "source": "hybrid",
                "slope_source": "memory",
                "slope_delta_z": -0.24,
                "escape_lane": "left",
                "jump_gate": "blocked_by_no_obstacle",
                "slide_gate": "allowed_slide_downhill",
                "lanes": [{"label": "center", "drop_risk": 0.12}],
            },
            "detections": {
                "enemy_classes": [{"label": "Bandit", "rect": [1, 2, 3, 4], "score": 0.9, "source": "asset_catalog", "entity_id": "bandit", "family": "bandit"}],
                "projectile_classes": [{"label": "ProjectileBloodMagic", "rect": [2, 3, 4, 5], "score": 0.88, "source": "asset_catalog", "entity_id": "projectile_bloodmagic", "family": "projectile_bloodmagic"}],
                "world_objects": [{"label": "Portal", "rect": [4, 5, 6, 7], "score": 0.8, "source": "runtime_template", "entity_id": "portal"}],
                "hazards": [{"label": "GhostBossSpike", "rect": [5, 6, 7, 8], "score": 0.87, "source": "asset_catalog", "entity_id": "ghostboss_spike", "hazard_kind": "spike"}],
            },
            "detection_sources": {"map_state": "external_memory"},
            "source_confidence": {"map_state": 0.95},
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

    assert len(rows) == 3
    assert rows[0]["schema_version"] == "runtime_events_v1"
    assert rows[0]["capture_bad_grab_count"] == ""
    assert rows[1]["schema_version"] == "runtime_events_v2"
    assert rows[1]["capture_bad_grab_count"] == "4"
    assert rows[1]["capture_last_error"] == "RuntimeError: fail"
    assert rows[1]["hud_debug_dumped"] == "True"
    assert rows[1]["hud_fail_streak"] == "9"
    assert rows[2]["schema_version"] == "runtime_events_v4"
    assert rows[2]["memory_probe_status"] == "ready"
    assert rows[2]["player_pose_world_x"] == "1.0"
    assert rows[2]["player_pose_heading_deg"] == "87.0"
    assert rows[2]["map_open"] == "True"
    assert rows[2]["map_scene_id"] == "GeneratedMap"
    assert rows[2]["map_objective"] == "challenge_shrine"
    assert rows[2]["map_poi_count"] == "1"
    assert rows[2]["map_poi_icons_json"] == "[\"portal\"]"
    assert rows[2]["navigation_terrain_kind"] == "downhill"
    assert rows[2]["navigation_slide_gate"] == "allowed_slide_downhill"
    assert "center" in rows[2]["navigation_lanes_json"]
    assert "projectile_bloodmagic" in rows[2]["projectile_classes_json"]
    assert "ghostboss_spike" in rows[2]["hazards_json"]
