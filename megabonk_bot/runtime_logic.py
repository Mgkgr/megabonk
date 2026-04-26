from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from megabonk_bot.world_state import MapState, PlayerPose, TrackedEntity, WorldObject


class BotMode(str, Enum):
    OFF = "OFF"
    ACTIVE = "ACTIVE"
    RECOVERY = "RECOVERY"
    PANIC = "PANIC"


@dataclass(frozen=True)
class DetectionBox:
    label: str
    rect: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class SceneSnapshot:
    frame_id: int
    ts: float
    frame_width: int = 0
    frame_height: int = 0
    enemies: list[DetectionBox] = field(default_factory=list)
    obstacles: list[DetectionBox] = field(default_factory=list)
    projectiles: list[DetectionBox] = field(default_factory=list)
    projectile_classes: list[TrackedEntity] = field(default_factory=list)
    interactables: list[DetectionBox] = field(default_factory=list)
    enemy_classes: list[TrackedEntity] = field(default_factory=list)
    world_objects: list[WorldObject] = field(default_factory=list)
    hazards: list[WorldObject] = field(default_factory=list)
    player_pose: PlayerPose | None = None
    map_state: MapState | None = None
    detection_sources: dict[str, str] = field(default_factory=dict)
    source_confidence: dict[str, float] = field(default_factory=dict)
    memory_probe_status: str = "disabled"
    hp_ratio: float | None = None
    lvl: int | None = None
    kills: int | None = None
    time_s: int | None = None
    is_dead: bool = False
    is_upgrade: bool = False
    safe_sector: str = "center"
    analysis: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BotAction:
    dir_id: int = 0
    yaw: int = 1
    jump: int = 0
    slide: int = 0
    press_space: bool = False
    press_tab: bool = False
    press_r: bool = False
    open_chest: bool = False
    reason: str = "idle"


def _safe_sector_from_enemies(
    enemies: list[TrackedEntity | DetectionBox],
    frame_width: int,
) -> str:
    if frame_width <= 0 or not enemies:
        return "center"
    left = 0.0
    center = 0.0
    right = 0.0
    for box in enemies:
        x, _, w, _ = box.rect
        cx = x + (w / 2.0)
        weight = max(0.05, float(getattr(box, "score", 0.0) or 0.0))
        weight *= max(1.0, float(getattr(box, "threat_tier", 1.0) or 1.0))
        if cx < frame_width / 3.0:
            left += weight
        elif cx > frame_width * (2.0 / 3.0):
            right += weight
        else:
            center += weight
    sector_scores = {
        "left": left,
        "center": center,
        "right": right,
    }
    return min(sector_scores, key=sector_scores.get)


def _entity_from_item(item) -> TrackedEntity:
    return TrackedEntity(
        label=str(getattr(item, "label", "enemy")),
        rect=tuple(getattr(item, "rect", (0, 0, 0, 0))),
        score=float(getattr(item, "score", 0.0)),
        source=str(getattr(item, "source", "screen_cv")),
        entity_id=getattr(item, "entity_id", None),
        threat_tier=float(getattr(item, "threat_tier", 1.0) or 1.0),
        family=getattr(item, "family", None),
        variant=getattr(item, "variant", None),
        metadata=dict(getattr(item, "metadata", {}) or {}),
    )


def _world_object_from_item(item) -> WorldObject:
    return WorldObject(
        label=str(getattr(item, "label", "object")),
        rect=tuple(getattr(item, "rect", (0, 0, 0, 0))),
        score=float(getattr(item, "score", 0.0)),
        source=str(getattr(item, "source", "screen_cv")),
        entity_id=getattr(item, "entity_id", None),
        poi_type=getattr(item, "poi_type", None),
        family=getattr(item, "family", None),
        variant=getattr(item, "variant", None),
        hazard_kind=getattr(item, "hazard_kind", None),
        icon_id=getattr(item, "icon_id", None),
        metadata=dict(getattr(item, "metadata", {}) or {}),
    )


def build_scene_snapshot(
    *,
    frame_id: int,
    ts: float,
    frame_width: int,
    analysis: dict[str, Any],
    hud_values: dict[str, Any],
    is_dead: bool,
    is_upgrade: bool,
    frame_height: int = 0,
) -> SceneSnapshot:
    enemies = [
        DetectionBox(label=item.label, rect=item.rect, score=float(item.score))
        for item in analysis.get("enemies", [])
    ]
    obstacles = [
        DetectionBox(label=cell.label, rect=cell.rect, score=float(cell.score))
        for cell in analysis.get("grid", [])
        if getattr(cell, "label", "") == "obstacle"
    ]
    interactables = [
        DetectionBox(label=item.label, rect=item.rect, score=float(item.score))
        for item in analysis.get("interactables", [])
    ]
    projectiles = [
        DetectionBox(label=item.label, rect=item.rect, score=float(item.score))
        for item in analysis.get("projectiles", [])
    ]
    projectile_classes = [_entity_from_item(item) for item in analysis.get("projectile_classes", [])]
    enemy_classes = [_entity_from_item(item) for item in analysis.get("enemy_classes", [])]
    world_objects = [_world_object_from_item(item) for item in analysis.get("world_objects", [])]
    hazards = [_world_object_from_item(item) for item in analysis.get("hazards", [])]

    player_pose = analysis.get("player_pose")
    if not isinstance(player_pose, PlayerPose):
        player_pose = PlayerPose()
    map_state = analysis.get("map_state")
    if not isinstance(map_state, MapState):
        map_state = MapState()

    safe_sector_source = enemy_classes if enemy_classes else enemies
    safe_sector = _safe_sector_from_enemies(safe_sector_source, frame_width=frame_width)
    return SceneSnapshot(
        frame_id=int(frame_id),
        ts=float(ts),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        enemies=enemies,
        obstacles=obstacles,
        projectiles=projectiles,
        projectile_classes=projectile_classes,
        interactables=interactables,
        enemy_classes=enemy_classes,
        world_objects=world_objects,
        hazards=hazards,
        player_pose=player_pose,
        map_state=map_state,
        detection_sources=dict(analysis.get("detection_sources", {}) or {}),
        source_confidence={
            str(key): float(value)
            for key, value in dict(analysis.get("source_confidence", {}) or {}).items()
        },
        memory_probe_status=str(analysis.get("memory_probe_status", "disabled")),
        hp_ratio=hud_values.get("hp_ratio"),
        lvl=hud_values.get("lvl"),
        kills=hud_values.get("kills"),
        time_s=hud_values.get("time"),
        is_dead=bool(is_dead),
        is_upgrade=bool(is_upgrade),
        safe_sector=safe_sector,
        analysis=analysis,
    )


def _detect_emergency_danger(snapshot: SceneSnapshot) -> str | None:
    if snapshot.frame_width <= 0:
        return None
    combined = list(snapshot.enemy_classes) + [
        TrackedEntity(
            label=item.label,
            rect=item.rect,
            score=item.score,
            source="screen_cv",
            entity_id=item.label,
            threat_tier=1.6,
        )
        for item in (snapshot.projectile_classes or snapshot.projectiles)
    ]
    combined.extend(
        TrackedEntity(
            label=item.label,
            rect=item.rect,
            score=item.score,
            source=item.source,
            entity_id=item.entity_id,
            threat_tier=max(1.7, float(item.metadata.get("threat_tier", 1.7)) if isinstance(item.metadata, dict) else 1.7),
            family=item.family,
            variant=item.variant,
            metadata=dict(item.metadata),
        )
        for item in snapshot.hazards
    )
    if not combined:
        return None
    sector_scores = {
        "left": 0.0,
        "center": 0.0,
        "right": 0.0,
    }
    for item in combined:
        x, y, w, h = item.rect
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        center_dist = abs((snapshot.frame_width / 2.0) - cx) / max(1.0, snapshot.frame_width / 2.0)
        vertical_weight = 1.0
        if snapshot.frame_height > 0:
            vertical_weight = 1.2 - min(1.0, cy / max(1.0, snapshot.frame_height))
        weight = max(0.1, float(item.score)) * max(1.0, float(item.threat_tier)) * max(0.2, vertical_weight)
        weight *= 1.3 - min(1.0, center_dist)
        if cx < snapshot.frame_width / 3.0:
            sector_scores["left"] += weight
        elif cx > snapshot.frame_width * (2.0 / 3.0):
            sector_scores["right"] += weight
        else:
            sector_scores["center"] += weight
    sector, value = max(sector_scores.items(), key=lambda kv: kv[1])
    return sector if value >= 1.0 else None


def _evasive_action(
    danger_sector: str,
    *,
    allow_map_scan: bool,
    map_scan_now: bool,
) -> BotAction:
    press_tab = bool(allow_map_scan and map_scan_now)
    if danger_sector == "left":
        return BotAction(dir_id=4, yaw=2, jump=0, slide=1, press_tab=press_tab, reason="evade_left_danger")
    if danger_sector == "right":
        return BotAction(dir_id=3, yaw=0, jump=0, slide=1, press_tab=press_tab, reason="evade_right_danger")
    return BotAction(dir_id=2, yaw=1, jump=0, slide=0, press_tab=press_tab, reason="evade_center_danger")


def choose_mvp_action(
    snapshot: SceneSnapshot,
    mode: BotMode,
    heuristic_action: tuple[int, int, int, int, str] | None = None,
    *,
    allow_map_scan: bool = False,
    map_scan_now: bool = False,
) -> BotAction:
    if mode == BotMode.PANIC:
        return BotAction(reason="panic")
    if mode == BotMode.OFF:
        return BotAction(reason="off")
    if snapshot.is_dead or mode == BotMode.RECOVERY:
        return BotAction(press_r=True, reason="recovery_restart")
    if snapshot.is_upgrade:
        return BotAction(press_space=True, reason="upgrade_random_space")

    emergency_sector = _detect_emergency_danger(snapshot)
    if emergency_sector is not None:
        return _evasive_action(
            emergency_sector,
            allow_map_scan=allow_map_scan,
            map_scan_now=map_scan_now,
        )

    if heuristic_action is not None:
        dir_id, yaw, jump, slide, reason = heuristic_action
        press_tab = bool(allow_map_scan and map_scan_now)
        return BotAction(
            dir_id=int(dir_id),
            yaw=int(yaw),
            jump=int(jump),
            slide=int(slide),
            press_tab=press_tab,
            open_chest=False,
            reason=str(reason),
        )

    yaw = 1
    if snapshot.safe_sector == "left":
        yaw = 0
    elif snapshot.safe_sector == "right":
        yaw = 2
    press_tab = bool(allow_map_scan and map_scan_now)
    return BotAction(
        dir_id=1,
        yaw=yaw,
        jump=0,
        slide=0,
        press_tab=press_tab,
        open_chest=False,
        reason="fallback_cruise",
    )
