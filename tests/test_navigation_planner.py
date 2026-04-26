from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from megabonk_bot.navigation import StatefulNavigationPlanner
from megabonk_bot.runtime_logic import BotMode, build_scene_snapshot
from megabonk_bot.world_state import MapState, PlayerPose


def _grid_cell(label, rect, score=0.8):
    return SimpleNamespace(label=label, rect=rect, score=score)


def _enemy(rect, score=0.95, threat_tier=2.0):
    return SimpleNamespace(
        label="Bandit",
        rect=rect,
        score=score,
        source="asset_catalog",
        entity_id="bandit",
        threat_tier=threat_tier,
        family="bandit",
        variant="default",
        metadata={},
    )


def _frame(frame_w=250, frame_h=200, value=140):
    return np.full((frame_h, frame_w, 3), value, dtype=np.uint8)


def _lane_grid(lane_profiles, *, frame_w=250, frame_h=200):
    lane_count = len(lane_profiles)
    nav_y0 = int(frame_h * 0.55)
    nav_h = frame_h - nav_y0
    row_h = max(1, nav_h // 4)
    lane_w = frame_w // lane_count
    cells = []
    for lane_index, labels in enumerate(lane_profiles):
        x0 = lane_index * lane_w
        width = lane_w if lane_index < (lane_count - 1) else frame_w - x0
        for row_index, label in enumerate(labels):
            y0 = nav_y0 + (row_index * row_h)
            height = row_h if row_index < 3 else frame_h - y0
            cells.append(_grid_cell(label, (x0, y0, width, height)))
    return cells


def _snapshot(
    lane_profiles,
    *,
    frame_w=250,
    frame_h=200,
    enemy_classes=None,
    player_pose=None,
    map_state=None,
):
    analysis = {
        "enemies": [],
        "grid": _lane_grid(lane_profiles, frame_w=frame_w, frame_h=frame_h),
        "interactables": [],
        "projectiles": [],
        "enemy_classes": enemy_classes or [],
    }
    if player_pose is not None:
        analysis["player_pose"] = player_pose
    if map_state is not None:
        analysis["map_state"] = map_state
    return build_scene_snapshot(
        frame_id=1,
        ts=1.0,
        frame_width=frame_w,
        frame_height=frame_h,
        analysis=analysis,
        hud_values={},
        is_dead=False,
        is_upgrade=False,
    )


def test_navigation_planner_escapes_left_without_random_jump():
    planner = StatefulNavigationPlanner(config={"profile": "cautious", "lane_count": 5})
    frame = _frame()
    snapshot = _snapshot(
        [
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
        ],
        enemy_classes=[_enemy((100, 70, 50, 90))],
    )

    action, navigation = planner.evaluate(frame, snapshot, mode=BotMode.ACTIVE)

    assert action.reason == "escape_lane_left"
    assert action.jump == 0
    assert navigation.escape_lane == "left_far"


def test_navigation_planner_jumps_only_over_confirmed_obstacle():
    planner = StatefulNavigationPlanner(config={"profile": "cautious", "lane_count": 5})
    frame = _frame()
    snapshot = _snapshot(
        [
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "obstacle", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
        ]
    )

    action, navigation = planner.evaluate(frame, snapshot, mode=BotMode.ACTIVE)

    assert action.reason == "jump_obstacle"
    assert action.jump == 1
    assert navigation.jump_gate == "allowed_jump_obstacle"


def test_navigation_planner_uses_bunny_hop_only_on_safe_flat_ground():
    planner = StatefulNavigationPlanner(config={"profile": "cautious", "lane_count": 5})
    frame = _frame()
    player_pose = PlayerPose(world_pos=(1.0, 1.0, 1.0), heading_deg=90.0, source="external_memory", confidence=0.99)
    map_state = MapState(active_room_id="room_a")
    snapshot = _snapshot(
        [
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
        ],
        player_pose=player_pose,
        map_state=map_state,
    )

    action = None
    for _ in range(3):
        action, navigation = planner.evaluate(frame, snapshot, mode=BotMode.ACTIVE)

    assert action.reason == "bunny_hop_safe"
    assert action.jump == 1
    assert navigation.terrain_kind == "flat"


def test_navigation_planner_slides_only_on_downhill():
    planner = StatefulNavigationPlanner(config={"profile": "cautious", "lane_count": 5})
    frame = _frame()
    map_state = MapState(active_room_id="room_a")
    action = None
    navigation = None
    for z in (1.2, 0.9, 0.6):
        snapshot = _snapshot(
            [
                ("surface", "surface", "surface", "surface"),
                ("surface", "surface", "surface", "surface"),
                ("surface", "surface", "surface", "surface"),
                ("surface", "surface", "surface", "surface"),
                ("surface", "surface", "surface", "surface"),
            ],
            player_pose=PlayerPose(
                world_pos=(1.0, 1.0, z),
                heading_deg=90.0,
                source="external_memory",
                confidence=0.99,
            ),
            map_state=map_state,
        )
        action, navigation = planner.evaluate(frame, snapshot, mode=BotMode.ACTIVE)

    assert action.reason == "slide_downhill"
    assert action.slide == 1
    assert navigation.slide_gate == "allowed_slide_downhill"


def test_navigation_planner_blocks_slide_without_memory_slope():
    planner = StatefulNavigationPlanner(
        config={
            "profile": "cautious",
            "lane_count": 5,
            "memory_required_for_slide": True,
        }
    )
    frame = _frame()
    snapshot = _snapshot(
        [
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
        ]
    )

    action, navigation = planner.evaluate(frame, snapshot, mode=BotMode.ACTIVE)

    assert action.reason == "advance_safe"
    assert action.slide == 0
    assert navigation.slide_gate == "blocked_by_memory_required"


def test_navigation_planner_holds_on_high_drop_risk():
    planner = StatefulNavigationPlanner(config={"profile": "cautious", "lane_count": 5})
    frame = _frame()
    frame[150:200, 100:150] = 0
    snapshot = _snapshot(
        [
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
            ("unknown", "unknown", "unknown", "unknown"),
            ("surface", "surface", "surface", "surface"),
            ("surface", "surface", "surface", "surface"),
        ]
    )

    action, navigation = planner.evaluate(frame, snapshot, mode=BotMode.ACTIVE)

    assert action.reason == "hold_drop_risk"
    assert action.jump == 0
    assert action.slide == 0
    assert navigation.drop_risk >= planner.drop_risk_threshold
