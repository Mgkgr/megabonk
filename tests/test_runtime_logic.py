from types import SimpleNamespace

from megabonk_bot.runtime_logic import (
    BotMode,
    build_scene_snapshot,
    choose_mvp_action,
)


def _box(label, rect, score=0.9):
    return SimpleNamespace(label=label, rect=rect, score=score)


def test_build_scene_snapshot_extracts_obstacles_and_sector():
    analysis = {
        "enemies": [_box("enemy", (300, 20, 40, 40), 0.9)],
        "grid": [
            _box("surface", (0, 0, 20, 20), 0.2),
            _box("obstacle", (40, 40, 20, 20), 0.8),
        ],
        "interactables": [_box("chest", (100, 100, 40, 40), 0.7)],
        "projectiles": [],
    }
    hud = {"hp_ratio": 0.75, "lvl": 6, "kills": 12, "time": 130}
    snapshot = build_scene_snapshot(
        frame_id=1,
        ts=123.0,
        frame_width=360,
        analysis=analysis,
        hud_values=hud,
        is_dead=False,
        is_upgrade=False,
    )
    assert len(snapshot.obstacles) == 1
    assert snapshot.safe_sector == "left"
    assert snapshot.lvl == 6
    assert snapshot.kills == 12


def test_choose_mvp_action_picks_upgrade_with_space():
    analysis = {"enemies": [], "grid": [], "interactables": [], "projectiles": []}
    snapshot = build_scene_snapshot(
        frame_id=1,
        ts=1.0,
        frame_width=200,
        analysis=analysis,
        hud_values={},
        is_dead=False,
        is_upgrade=True,
    )
    action = choose_mvp_action(snapshot, mode=BotMode.ACTIVE, heuristic_action=None)
    assert action.press_space is True
    assert action.open_chest is False
    assert action.reason == "upgrade_random_space"


def test_choose_mvp_action_recovery_requests_restart():
    analysis = {"enemies": [], "grid": [], "interactables": [], "projectiles": []}
    snapshot = build_scene_snapshot(
        frame_id=2,
        ts=2.0,
        frame_width=200,
        analysis=analysis,
        hud_values={},
        is_dead=True,
        is_upgrade=False,
    )
    action = choose_mvp_action(snapshot, mode=BotMode.RECOVERY, heuristic_action=None)
    assert action.press_r is True
    assert action.dir_id == 0


def test_choose_mvp_action_never_opens_chest():
    analysis = {"enemies": [], "grid": [], "interactables": [_box("chest", (0, 0, 10, 10))], "projectiles": []}
    snapshot = build_scene_snapshot(
        frame_id=3,
        ts=3.0,
        frame_width=200,
        analysis=analysis,
        hud_values={},
        is_dead=False,
        is_upgrade=False,
    )
    action = choose_mvp_action(
        snapshot,
        mode=BotMode.ACTIVE,
        heuristic_action=(1, 2, 0, 0, "EVADE_ENEMY"),
    )
    assert action.open_chest is False
    assert action.press_space is False
    assert action.reason == "EVADE_ENEMY"


def test_choose_mvp_action_prefers_projectile_evasion():
    analysis = {
        "enemies": [],
        "grid": [],
        "interactables": [],
        "projectiles": [_box("projectile", (80, 40, 60, 60), 0.95)],
        "enemy_classes": [],
        "world_objects": [],
    }
    snapshot = build_scene_snapshot(
        frame_id=4,
        ts=4.0,
        frame_width=200,
        frame_height=120,
        analysis=analysis,
        hud_values={},
        is_dead=False,
        is_upgrade=False,
    )
    action = choose_mvp_action(snapshot, mode=BotMode.ACTIVE, heuristic_action=None)
    assert action.reason == "evade_center_danger"
    assert action.jump == 0
    assert action.press_space is False


def test_build_scene_snapshot_keeps_projectile_classes_and_hazards():
    analysis = {
        "enemies": [],
        "grid": [],
        "interactables": [],
        "projectiles": [],
        "projectile_classes": [
            SimpleNamespace(
                label="ProjectileBloodMagic",
                rect=(70, 30, 40, 40),
                score=0.92,
                source="asset_catalog",
                entity_id="projectile_bloodmagic",
                threat_tier=2.6,
                family="projectile_bloodmagic",
                variant="default",
                metadata={"damage_type": "blood"},
            )
        ],
        "hazards": [
            SimpleNamespace(
                label="GhostBossSpike",
                rect=(80, 55, 40, 50),
                score=0.95,
                source="asset_catalog",
                entity_id="ghostboss_spike",
                poi_type=None,
                family="ghostboss_spike",
                variant="default",
                hazard_kind="spike",
                icon_id=None,
                metadata={"hazard_kind": "spike"},
            )
        ],
    }
    snapshot = build_scene_snapshot(
        frame_id=5,
        ts=5.0,
        frame_width=200,
        frame_height=120,
        analysis=analysis,
        hud_values={},
        is_dead=False,
        is_upgrade=False,
    )

    assert snapshot.projectile_classes[0].entity_id == "projectile_bloodmagic"
    assert snapshot.hazards[0].hazard_kind == "spike"

    action = choose_mvp_action(snapshot, mode=BotMode.ACTIVE, heuristic_action=None)
    assert action.reason == "evade_center_danger"
