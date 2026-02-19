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
