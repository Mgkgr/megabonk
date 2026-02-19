from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


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
    enemies: list[DetectionBox]
    obstacles: list[DetectionBox]
    projectiles: list[DetectionBox]
    interactables: list[DetectionBox]
    hp_ratio: float | None
    lvl: int | None
    kills: int | None
    time_s: int | None
    is_dead: bool
    is_upgrade: bool
    safe_sector: str
    analysis: dict[str, Any]


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
    enemies: list[DetectionBox],
    frame_width: int,
) -> str:
    if frame_width <= 0 or not enemies:
        return "center"
    left = 0
    center = 0
    right = 0
    for box in enemies:
        x, _, w, _ = box.rect
        cx = x + (w / 2.0)
        if cx < frame_width / 3.0:
            left += 1
        elif cx > frame_width * (2.0 / 3.0):
            right += 1
        else:
            center += 1
    sector_scores = {
        "left": left,
        "center": center,
        "right": right,
    }
    return min(sector_scores, key=sector_scores.get)


def build_scene_snapshot(
    *,
    frame_id: int,
    ts: float,
    frame_width: int,
    analysis: dict[str, Any],
    hud_values: dict[str, Any],
    is_dead: bool,
    is_upgrade: bool,
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
    safe_sector = _safe_sector_from_enemies(enemies, frame_width=frame_width)
    return SceneSnapshot(
        frame_id=int(frame_id),
        ts=float(ts),
        enemies=enemies,
        obstacles=obstacles,
        projectiles=projectiles,
        interactables=interactables,
        hp_ratio=hud_values.get("hp_ratio"),
        lvl=hud_values.get("lvl"),
        kills=hud_values.get("kills"),
        time_s=hud_values.get("time"),
        is_dead=bool(is_dead),
        is_upgrade=bool(is_upgrade),
        safe_sector=safe_sector,
        analysis=analysis,
    )


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

    # Fallback если эвристика не отдала действие.
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
