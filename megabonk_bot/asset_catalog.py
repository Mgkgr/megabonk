from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CatalogEntry:
    entity_id: str
    kind: str
    display_name: str
    aliases: tuple[str, ...] = ()
    preview_relpath: str | None = None
    template_names: tuple[str, ...] = ()
    threat_tier: float = 1.0
    poi_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OcrLexicon:
    tokens: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    normalize: dict[str, str] = field(default_factory=dict)

    def all_terms(self) -> tuple[str, ...]:
        ordered = []
        seen = set()
        for value in self.phrases + self.tokens + tuple(self.normalize.keys()):
            item = str(value).strip()
            if not item:
                continue
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(item)
        return tuple(ordered)


@dataclass
class LoadedCatalogEntry:
    entry: CatalogEntry
    preview_bgr: Any | None = None
    preview_size: tuple[int, int] | None = None
    preview_path: Path | None = None
    minimap_icon_bgr: Any | None = None
    minimap_icon_size: tuple[int, int] | None = None
    minimap_icon_path: Path | None = None
    silhouette_bgr: Any | None = None
    silhouette_size: tuple[int, int] | None = None
    silhouette_path: Path | None = None

    @property
    def entity_id(self) -> str:
        return self.entry.entity_id

    @property
    def kind(self) -> str:
        return self.entry.kind

    @property
    def display_name(self) -> str:
        return self.entry.display_name

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.entry.aliases

    @property
    def template_names(self) -> tuple[str, ...]:
        return self.entry.template_names

    @property
    def threat_tier(self) -> float:
        return float(self.entry.threat_tier)

    @property
    def poi_type(self) -> str | None:
        return self.entry.poi_type

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.entry.metadata)

    @property
    def family(self) -> str | None:
        value = self.entry.metadata.get("family")
        return str(value) if value is not None else None

    @property
    def variant(self) -> str | None:
        value = self.entry.metadata.get("variant")
        return str(value) if value is not None else None

    @property
    def scene_tags(self) -> tuple[str, ...]:
        raw = self.entry.metadata.get("scene_tags", ())
        if isinstance(raw, (list, tuple, set)):
            return tuple(str(item) for item in raw if item)
        if raw:
            return (str(raw),)
        return ()

    @property
    def damage_type(self) -> str | None:
        value = self.entry.metadata.get("damage_type")
        return str(value) if value is not None else None

    @property
    def hazard_kind(self) -> str | None:
        value = self.entry.metadata.get("hazard_kind")
        return str(value) if value is not None else None

    @property
    def icon_id(self) -> str | None:
        value = self.entry.metadata.get("icon_id")
        return str(value) if value is not None else None

    @property
    def mesh_relpath(self) -> str | None:
        value = self.entry.metadata.get("mesh_relpath")
        return str(value) if value is not None else None

    @property
    def minimap_icon_relpath(self) -> str | None:
        value = self.entry.metadata.get("minimap_icon_relpath")
        return str(value) if value is not None else None

    @property
    def silhouette_relpath(self) -> str | None:
        value = self.entry.metadata.get("silhouette_relpath")
        return str(value) if value is not None else None


@dataclass(frozen=True)
class CuratedCatalogs:
    enemies: tuple[LoadedCatalogEntry, ...] = ()
    world: tuple[LoadedCatalogEntry, ...] = ()
    projectiles: tuple[LoadedCatalogEntry, ...] = ()
    ocr_lexicon: OcrLexicon = field(default_factory=OcrLexicon)
    asset_refs_dir: Path | None = None
    enemy_catalog_path: Path | None = None
    world_catalog_path: Path | None = None
    projectile_catalog_path: Path | None = None
    ocr_lexicon_path: Path | None = None

    def has_enemy_previews(self) -> bool:
        return any(item.preview_bgr is not None for item in self.enemies)

    def has_world_previews(self) -> bool:
        return any(item.preview_bgr is not None for item in self.world)

    def has_projectile_previews(self) -> bool:
        return any(item.preview_bgr is not None for item in self.projectiles)

    def minimap_icon_entries(self) -> tuple[LoadedCatalogEntry, ...]:
        combined = list(self.world) + list(self.projectiles) + list(self.enemies)
        return tuple(item for item in combined if item.minimap_icon_bgr is not None)


def _coerce_entry(raw: dict[str, Any]) -> CatalogEntry:
    aliases = tuple(str(item) for item in raw.get("aliases", []) if item)
    template_names = tuple(str(item) for item in raw.get("template_names", []) if item)
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return CatalogEntry(
        entity_id=str(raw["entity_id"]),
        kind=str(raw.get("kind", "unknown")),
        display_name=str(raw.get("display_name", raw["entity_id"])),
        aliases=aliases,
        preview_relpath=str(raw["preview_relpath"]) if raw.get("preview_relpath") else None,
        template_names=template_names,
        threat_tier=float(raw.get("threat_tier", 1.0)),
        poi_type=str(raw["poi_type"]) if raw.get("poi_type") else None,
        metadata=metadata,
    )


def _read_manifest(path: Path | None) -> tuple[CatalogEntry, ...]:
    if path is None or not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Catalog manifest must be a list: {path}")
    entries = []
    for item in raw:
        if not isinstance(item, dict) or "entity_id" not in item:
            continue
        entries.append(_coerce_entry(item))
    return tuple(entries)


def _read_ocr_lexicon(path: Path | None) -> OcrLexicon:
    if path is None or not path.exists():
        return OcrLexicon()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return OcrLexicon()
    tokens = tuple(str(item) for item in raw.get("tokens", []) if item)
    phrases = tuple(str(item) for item in raw.get("phrases", []) if item)
    normalize = raw.get("normalize", {})
    if not isinstance(normalize, dict):
        normalize = {}
    return OcrLexicon(
        tokens=tokens,
        phrases=phrases,
        normalize={str(key): str(value) for key, value in normalize.items() if key and value},
    )


def _load_image(path: Path | None):
    if path is None or not path.exists():
        return None, None, None
    try:
        import cv2
    except Exception:
        return None, None, None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None, None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    h, w = image.shape[:2]
    return image, (w, h), path


def _resolve_asset_path(asset_refs_dir: Path | None, relpath: str | None) -> Path | None:
    if asset_refs_dir is None or not relpath:
        return None
    path = asset_refs_dir / str(relpath)
    return path if path.exists() else None


def _load_entries(asset_refs_dir: Path | None, entries: tuple[CatalogEntry, ...]) -> tuple[LoadedCatalogEntry, ...]:
    loaded: list[LoadedCatalogEntry] = []
    for entry in entries:
        preview_path = _resolve_asset_path(asset_refs_dir, entry.preview_relpath)
        preview_bgr, preview_size, preview_path = _load_image(preview_path)

        minimap_icon_relpath = entry.metadata.get("minimap_icon_relpath")
        minimap_icon_path = _resolve_asset_path(asset_refs_dir, str(minimap_icon_relpath) if minimap_icon_relpath else None)
        minimap_icon_bgr, minimap_icon_size, minimap_icon_path = _load_image(minimap_icon_path)

        silhouette_relpath = entry.metadata.get("silhouette_relpath")
        silhouette_path = _resolve_asset_path(asset_refs_dir, str(silhouette_relpath) if silhouette_relpath else None)
        silhouette_bgr, silhouette_size, silhouette_path = _load_image(silhouette_path)

        loaded.append(
            LoadedCatalogEntry(
                entry=entry,
                preview_bgr=preview_bgr,
                preview_size=preview_size,
                preview_path=preview_path,
                minimap_icon_bgr=minimap_icon_bgr,
                minimap_icon_size=minimap_icon_size,
                minimap_icon_path=minimap_icon_path,
                silhouette_bgr=silhouette_bgr,
                silhouette_size=silhouette_size,
                silhouette_path=silhouette_path,
            )
        )
    return tuple(loaded)


def load_curated_catalogs(
    *,
    asset_refs_dir: str | Path | None,
    enemy_catalog_path: str | Path | None,
    world_catalog_path: str | Path | None,
    projectile_catalog_path: str | Path | None = None,
    ocr_lexicon_path: str | Path | None = None,
) -> CuratedCatalogs:
    asset_refs = Path(asset_refs_dir).resolve() if asset_refs_dir else None
    enemy_path = Path(enemy_catalog_path).resolve() if enemy_catalog_path else None
    world_path = Path(world_catalog_path).resolve() if world_catalog_path else None
    projectile_path = Path(projectile_catalog_path).resolve() if projectile_catalog_path else None
    lexicon_path = Path(ocr_lexicon_path).resolve() if ocr_lexicon_path else None

    enemy_entries = _read_manifest(enemy_path)
    world_entries = _read_manifest(world_path)
    projectile_entries = _read_manifest(projectile_path)

    return CuratedCatalogs(
        enemies=_load_entries(asset_refs, enemy_entries),
        world=_load_entries(asset_refs, world_entries),
        projectiles=_load_entries(asset_refs, projectile_entries),
        ocr_lexicon=_read_ocr_lexicon(lexicon_path),
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_path,
        world_catalog_path=world_path,
        projectile_catalog_path=projectile_path,
        ocr_lexicon_path=lexicon_path,
    )
