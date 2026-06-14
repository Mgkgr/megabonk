from __future__ import annotations

import json
import re
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
    preview_relpaths: tuple[str, ...] = ()
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


@dataclass(frozen=True)
class PreviewDescriptor:
    gray32: Any
    hist32: Any
    edges32: Any
    mask32: Any


@dataclass
class LoadedCatalogEntry:
    entry: CatalogEntry
    preview_bgr: Any | None = None
    preview_size: tuple[int, int] | None = None
    preview_path: Path | None = None
    preview_descriptor: PreviewDescriptor | None = None
    extra_preview_bgrs: tuple[Any, ...] = ()
    extra_preview_sizes: tuple[tuple[int, int], ...] = ()
    extra_preview_paths: tuple[Path, ...] = ()
    extra_preview_descriptors: tuple[PreviewDescriptor, ...] = ()
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

    @property
    def preview_samples(self) -> tuple[Any, ...]:
        samples = []
        if self.preview_bgr is not None:
            samples.append(self.preview_bgr)
        samples.extend(item for item in self.extra_preview_bgrs if item is not None)
        return tuple(samples)

    @property
    def preview_descriptors(self) -> tuple[PreviewDescriptor, ...]:
        descriptors = []
        if self.preview_descriptor is not None:
            descriptors.append(self.preview_descriptor)
        descriptors.extend(item for item in self.extra_preview_descriptors if item is not None)
        return tuple(descriptors)


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
    preview_relpaths = tuple(str(item) for item in raw.get("preview_relpaths", []) if item)
    template_names = tuple(str(item) for item in raw.get("template_names", []) if item)
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    preview_relpath = str(raw["preview_relpath"]) if raw.get("preview_relpath") else None
    if not preview_relpath and preview_relpaths:
        preview_relpath = preview_relpaths[0]
    return CatalogEntry(
        entity_id=str(raw["entity_id"]),
        kind=str(raw.get("kind", "unknown")),
        display_name=str(raw.get("display_name", raw["entity_id"])),
        aliases=aliases,
        preview_relpath=preview_relpath,
        preview_relpaths=preview_relpaths,
        template_names=template_names,
        threat_tier=float(raw.get("threat_tier", 1.0)),
        poi_type=str(raw["poi_type"]) if raw.get("poi_type") else None,
        metadata=metadata,
    )


def _read_structured_file(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Для YAML-каталога нужен пакет pyyaml (`pip install pyyaml`).") from exc
        return yaml.safe_load(raw)
    return json.loads(raw)


def _read_manifest(path: Path | None) -> tuple[CatalogEntry, ...]:
    if path is None or not path.exists():
        return ()
    raw = _read_structured_file(path)
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
    raw = _read_structured_file(path)
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


def _build_preview_descriptor(image):
    if image is None:
        return None
    try:
        import cv2
    except Exception:
        return None
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    edges = cv2.Canny(gray, 60, 180)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return PreviewDescriptor(gray32=gray, hist32=hist, edges32=edges, mask32=mask)


def _resolve_asset_path(asset_refs_dir: Path | None, relpath: str | None) -> Path | None:
    if asset_refs_dir is None or not relpath:
        return None
    path = asset_refs_dir / str(relpath)
    return path if path.exists() else None


def _compact_match_token(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z]+", "", str(value).strip().lower())


def _entry_match_tokens(entry: CatalogEntry) -> tuple[str, ...]:
    raw_tokens = [entry.entity_id, entry.display_name, *entry.aliases]
    family = entry.metadata.get("family")
    if family:
        raw_tokens.append(str(family))
    seen = set()
    tokens = []
    for raw in raw_tokens:
        token = _compact_match_token(str(raw))
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


def _discover_debug_samples_dir(*paths: Path | None) -> Path | None:
    checked = set()
    for path in paths:
        if path is None:
            continue
        for base in (path.parent, path.parent.parent):
            if base in checked:
                continue
            checked.add(base)
            candidate = base / "dbg_hud"
            if candidate.exists() and candidate.is_dir():
                return candidate
    return None


def _discover_extra_preview_paths(debug_samples_dir: Path | None, entry: CatalogEntry) -> tuple[Path, ...]:
    if debug_samples_dir is None or not debug_samples_dir.exists():
        return ()
    tokens = _entry_match_tokens(entry)
    if not tokens:
        return ()
    matches: list[Path] = []
    for path in sorted(debug_samples_dir.rglob("*.png")):
        stem = _compact_match_token(path.stem)
        if not stem:
            continue
        if any(token == stem or token in stem for token in tokens):
            matches.append(path)
    return tuple(matches)


def _load_extra_previews(paths: tuple[Path, ...], primary_path: Path | None) -> tuple[tuple[Any, ...], tuple[tuple[int, int], ...], tuple[Path, ...]]:
    images: list[Any] = []
    sizes: list[tuple[int, int]] = []
    loaded_paths: list[Path] = []
    primary_resolved = primary_path.resolve() if primary_path is not None and primary_path.exists() else None
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if primary_resolved is not None and resolved == primary_resolved:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        image, size, loaded_path = _load_image(path)
        if image is None or size is None or loaded_path is None:
            continue
        images.append(image)
        sizes.append(size)
        loaded_paths.append(loaded_path)
    return tuple(images), tuple(sizes), tuple(loaded_paths)


def _load_entries(
    asset_refs_dir: Path | None,
    entries: tuple[CatalogEntry, ...],
    *,
    debug_samples_dir: Path | None = None,
) -> tuple[LoadedCatalogEntry, ...]:
    loaded: list[LoadedCatalogEntry] = []
    for entry in entries:
        preview_path = _resolve_asset_path(asset_refs_dir, entry.preview_relpath)
        preview_bgr, preview_size, preview_path = _load_image(preview_path)
        preview_descriptor = _build_preview_descriptor(preview_bgr)
        manifest_extra_paths = tuple(
            _resolve_asset_path(asset_refs_dir, relpath)
            for relpath in entry.preview_relpaths
            if relpath and relpath != entry.preview_relpath
        )
        extra_preview_paths = tuple(path for path in manifest_extra_paths if path is not None)
        extra_preview_paths = extra_preview_paths + _discover_extra_preview_paths(debug_samples_dir, entry)
        extra_preview_bgrs, extra_preview_sizes, extra_preview_paths = _load_extra_previews(
            extra_preview_paths,
            preview_path,
        )
        extra_preview_descriptors = tuple(_build_preview_descriptor(item) for item in extra_preview_bgrs if item is not None)

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
                preview_descriptor=preview_descriptor,
                extra_preview_bgrs=extra_preview_bgrs,
                extra_preview_sizes=extra_preview_sizes,
                extra_preview_paths=extra_preview_paths,
                extra_preview_descriptors=extra_preview_descriptors,
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
    debug_samples_dir = _discover_debug_samples_dir(enemy_path, world_path, projectile_path)

    return CuratedCatalogs(
        enemies=_load_entries(asset_refs, enemy_entries, debug_samples_dir=debug_samples_dir),
        world=_load_entries(asset_refs, world_entries, debug_samples_dir=debug_samples_dir),
        projectiles=_load_entries(asset_refs, projectile_entries, debug_samples_dir=debug_samples_dir),
        ocr_lexicon=_read_ocr_lexicon(lexicon_path),
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_path,
        world_catalog_path=world_path,
        projectile_catalog_path=projectile_path,
        ocr_lexicon_path=lexicon_path,
    )
