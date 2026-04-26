from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrackedEntity:
    label: str
    rect: tuple[int, int, int, int]
    score: float
    source: str = "screen_cv"
    entity_id: str | None = None
    threat_tier: float = 1.0
    family: str | None = None
    variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldObject:
    label: str
    rect: tuple[int, int, int, int]
    score: float
    source: str = "screen_cv"
    entity_id: str | None = None
    poi_type: str | None = None
    family: str | None = None
    variant: str | None = None
    hazard_kind: str | None = None
    icon_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MapPoi:
    label: str
    pos_norm: tuple[float, float]
    score: float
    source: str = "ui_cv"
    poi_type: str | None = None
    icon_id: str | None = None


@dataclass(frozen=True)
class PlayerPose:
    map_norm: tuple[float, float] | None = None
    world_pos: tuple[float, float, float] | None = None
    heading_deg: float | None = None
    source: str = "none"
    confidence: float = 0.0


@dataclass(frozen=True)
class MapState:
    map_open: bool = False
    minimap_visible: bool = False
    player_norm: tuple[float, float] | None = None
    biome: str | None = None
    scene_id: str | None = None
    active_room_id: str | None = None
    room_start: tuple[float, float, float] | None = None
    room_end: tuple[float, float, float] | None = None
    is_crypt: bool | None = None
    objective: str | None = None
    objective_confidence: float = 0.0
    boss_spotted: bool | None = None
    charged_shrines: int | None = None
    graveyard_crypt_keys: int | None = None
    pois: tuple[MapPoi, ...] = ()
    source: str = "none"
    confidence: float = 0.0
