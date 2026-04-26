from dataclasses import dataclass
from types import SimpleNamespace

from run_runtime_bot import build_runtime_event
from megabonk_bot.navigation import NavigationContext
from megabonk_bot.world_state import MapState, PlayerPose


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
        schema_version="runtime_events_v2",
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


def test_runtime_event_v3_contains_world_state_blocks():
    snapshot = _make_snapshot()
    snapshot.player_pose = PlayerPose(map_norm=(0.25, 0.75), world_pos=(1.0, 2.0, 3.0), source="external_memory", confidence=0.95)
    snapshot.map_state = MapState(
        map_open=True,
        minimap_visible=True,
        player_norm=(0.25, 0.75),
        biome="forest",
        scene_id="GeneratedMap",
        active_room_id="room_7",
        is_crypt=False,
        objective="challenge_shrine",
        objective_confidence=0.91,
        boss_spotted=True,
        charged_shrines=2,
        graveyard_crypt_keys=1,
        source="external_memory",
        confidence=0.95,
    )
    snapshot.enemy_classes = [SimpleNamespace(label="Bandit", rect=(10, 20, 30, 40), score=0.9, source="asset_catalog", entity_id="bandit", threat_tier=2.0, family="bandit", variant="default")]
    snapshot.projectile_classes = [SimpleNamespace(label="ProjectileBloodMagic", rect=(12, 24, 20, 20), score=0.88, source="asset_catalog", entity_id="projectile_bloodmagic", threat_tier=2.6, family="projectile_bloodmagic", variant="default")]
    snapshot.world_objects = [SimpleNamespace(label="Portal", rect=(40, 50, 20, 20), score=0.8, source="runtime_template", entity_id="portal", poi_type="exit")]
    snapshot.hazards = [SimpleNamespace(label="GhostBossSpike", rect=(60, 55, 20, 25), score=0.87, source="asset_catalog", entity_id="ghostboss_spike", family="ghostboss_spike", variant="default", hazard_kind="spike", icon_id=None, metadata={"hazard_kind": "spike"})]
    snapshot.detection_sources = {"map_state": "external_memory", "enemies": "asset_catalog"}
    snapshot.source_confidence = {"map_state": 0.95, "enemies": 0.8}
    snapshot.memory_probe_status = "ready"

    event = build_runtime_event(
        ts=1.0,
        mode=SimpleNamespace(value="ACTIVE"),
        frame_id=1,
        screen="RUNNING",
        snapshot=snapshot,
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
        schema_version="runtime_events_v3",
    )
    assert event["schema_version"] == "runtime_events_v3"
    assert event["player_pose"]["world_pos"] == [1.0, 2.0, 3.0]
    assert event["map_state"]["map_open"] is True
    assert event["map_state"]["scene_id"] == "GeneratedMap"
    assert event["map_state"]["objective"] == "challenge_shrine"
    assert event["memory_probe_status"] == "ready"
    assert event["detections"]["enemy_classes"][0]["entity_id"] == "bandit"
    assert event["detections"]["projectile_classes"][0]["entity_id"] == "projectile_bloodmagic"
    assert event["detections"]["world_objects"][0]["entity_id"] == "portal"
    assert event["detections"]["hazards"][0]["hazard_kind"] == "spike"
    assert event["detection_sources"]["map_state"] == "external_memory"


def test_runtime_event_v4_contains_navigation_block():
    snapshot = _make_snapshot()
    snapshot.player_pose = PlayerPose(
        map_norm=(0.25, 0.75),
        world_pos=(1.0, 2.0, 3.0),
        heading_deg=90.0,
        source="external_memory",
        confidence=0.95,
    )
    navigation = NavigationContext(
        terrain_kind="downhill",
        drop_risk=0.12,
        obstacle_cost=0.2,
        clearance=0.85,
        nav_confidence=0.93,
        source="hybrid",
        slope_source="memory",
        escape_lane="left",
        jump_gate="blocked_by_no_obstacle",
        slide_gate="allowed_slide_downhill",
        slope_delta_z=-0.22,
    )
    event = build_runtime_event(
        ts=1.0,
        mode=SimpleNamespace(value="ACTIVE"),
        frame_id=1,
        screen="RUNNING",
        snapshot=snapshot,
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
        navigation_context=navigation,
        schema_version="runtime_events_v4",
    )
    assert event["schema_version"] == "runtime_events_v4"
    assert event["player_pose"]["heading_deg"] == 90.0
    assert event["navigation"]["terrain_kind"] == "downhill"
    assert event["navigation"]["source"] == "hybrid"
    assert event["navigation"]["slide_gate"] == "allowed_slide_downhill"
