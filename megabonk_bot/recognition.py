from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from megabonk_bot.asset_catalog import CuratedCatalogs, LoadedCatalogEntry
from megabonk_bot.hud import DEFAULT_HUD_REGIONS
from megabonk_bot.memory_probe import ProbeResult
from megabonk_bot.ui_ocr import UiTextDetection
from megabonk_bot.vision import find_in_region
from megabonk_bot.world_state import MapPoi, MapState, PlayerPose, TrackedEntity, WorldObject


@dataclass(frozen=True)
class LabeledBox:
    label: str
    rect: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class GridCell:
    label: str
    rect: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class RegionOverlay:
    label: str
    rect: tuple[int, int, int, int]
    color: tuple[int, int, int] = (255, 0, 255)


@dataclass(frozen=True)
class DetectorProfile:
    scene_id: str = "GeneratedMap"
    tags: tuple[str, ...] = ("generatedmap",)
    allow_loot: bool = True
    allow_shrines: bool = True
    allow_chests: bool = True
    boss_emphasis: float = 1.0
    hazard_emphasis: float = 1.0


def _normalize_tag(value: str | None) -> str:
    if not value:
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _grid_edges(gray: np.ndarray, rows: int, cols: int) -> list[GridCell]:
    h, w = gray.shape[:2]
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag_norm = mag / (mag.max() + 1e-6)

    cell_h = max(1, h // rows)
    cell_w = max(1, w // cols)

    scores = []
    cells = []
    for row in range(rows):
        for col in range(cols):
            x0 = col * cell_w
            y0 = row * cell_h
            x1 = w if col == cols - 1 else (x0 + cell_w)
            y1 = h if row == rows - 1 else (y0 + cell_h)
            patch = mag_norm[y0:y1, x0:x1]
            score = float(patch.mean()) if patch.size else 0.0
            scores.append(score)
            cells.append((x0, y0, x1 - x0, y1 - y0, score))

    low_thr = float(np.percentile(scores, 35)) if scores else 0.0
    high_thr = float(np.percentile(scores, 70)) if scores else 0.0
    if high_thr - low_thr < 0.05:
        high_thr = low_thr + 0.05

    labeled = []
    for x0, y0, cw, ch, score in cells:
        if score <= low_thr:
            label = "surface"
        elif score >= high_thr:
            label = "obstacle"
        else:
            label = "unknown"
        labeled.append(GridCell(label=label, rect=(x0, y0, cw, ch), score=score))
    return labeled


def _mask_to_boxes(mask: np.ndarray, min_area: float, label: str) -> list[LabeledBox]:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    boxes = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        score = min(1.0, area / (min_area * 4))
        boxes.append(LabeledBox(label=label, rect=(x, y, w, h), score=score))
    return boxes


def _color_mask_boxes(
    frame_bgr: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    min_area: float,
    label: str,
) -> list[LabeledBox]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    return _mask_to_boxes(mask, min_area=min_area, label=label)


def _multi_color_mask_boxes(
    frame_bgr: np.ndarray,
    ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    *,
    min_area: float,
    label: str,
) -> list[LabeledBox]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
    return _mask_to_boxes(mask, min_area=min_area, label=label)


def _template_boxes(
    frame_bgr: np.ndarray,
    templates: dict[str, np.ndarray] | None,
    name_hints: tuple[str, ...],
    threshold: float,
) -> list[LabeledBox]:
    if not templates:
        return []
    h, w = frame_bgr.shape[:2]
    region = (0, 0, w, h)
    found = []
    hints = tuple(item.lower() for item in name_hints)
    for name, tpl in templates.items():
        lowered = name.lower()
        if not any(hint in lowered for hint in hints):
            continue
        ok, (cx, cy), score = find_in_region(
            frame_bgr,
            tpl,
            region,
            threshold=threshold,
            scales=(0.8, 0.9, 1.0, 1.1, 1.25),
        )
        if not ok:
            continue
        th, tw = tpl.shape[:2]
        x0 = max(0, int(cx - tw / 2))
        y0 = max(0, int(cy - th / 2))
        found.append(LabeledBox(label=name, rect=(x0, y0, tw, th), score=score))
    return found


def _friendly_name(raw: str) -> str:
    lowered = raw.lower()
    if "chest" in lowered:
        return "Chest"
    if "coin" in lowered or "gold" in lowered:
        return "Coin"
    if "projectile" in lowered or "orb" in lowered or "arrow" in lowered or "spike" in lowered:
        return "Projectile"
    if "foliant" in lowered or "tome" in lowered or "book" in lowered:
        return "Tome"
    if "shrine" in lowered:
        return "Shrine"
    if "portal" in lowered:
        return "Portal"
    if "door" in lowered:
        return "Door"
    if "key" in lowered:
        return "Key"
    return raw.replace("tpl_", "").replace("_", " ").title()


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = float((x1 - x0) * (y1 - y0))
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _dedupe_boxes(items: list, iou_thr: float = 0.45) -> list:
    output = []
    for item in sorted(items, key=lambda current: float(getattr(current, "score", 0.0)), reverse=True):
        rect = getattr(item, "rect", None)
        if rect is not None and any(_iou(rect, prev.rect) >= iou_thr for prev in output if getattr(prev, "rect", None) is not None):
            continue
        output.append(item)
    return output


def _dedupe_pois(items: list[MapPoi], distance_thr: float = 0.08) -> tuple[MapPoi, ...]:
    output: list[MapPoi] = []
    for item in sorted(items, key=lambda current: float(getattr(current, "score", 0.0)), reverse=True):
        pos = getattr(item, "pos_norm", None)
        if pos is None:
            output.append(item)
            continue
        duplicate = False
        for prev in output:
            prev_pos = getattr(prev, "pos_norm", None)
            if prev_pos is None:
                continue
            dx = float(pos[0]) - float(prev_pos[0])
            dy = float(pos[1]) - float(prev_pos[1])
            if (dx * dx + dy * dy) ** 0.5 <= distance_thr and (
                getattr(item, "icon_id", None) == getattr(prev, "icon_id", None)
                or getattr(item, "label", None) == getattr(prev, "label", None)
            ):
                duplicate = True
                break
        if not duplicate:
            output.append(item)
    return tuple(output)


def _extract_patch(frame_bgr: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(frame_bgr.shape[1], int(x + w))
    y1 = min(frame_bgr.shape[0], int(y + h))
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return patch


def _patch_similarity(patch_bgr: np.ndarray, preview_bgr: np.ndarray) -> float:
    if patch_bgr is None or preview_bgr is None or patch_bgr.size == 0 or preview_bgr.size == 0:
        return 0.0
    target_size = (32, 32)
    patch = cv2.resize(patch_bgr, target_size, interpolation=cv2.INTER_AREA)
    preview = cv2.resize(preview_bgr, target_size, interpolation=cv2.INTER_AREA)

    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    preview_gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    corr = float(cv2.matchTemplate(patch_gray, preview_gray, cv2.TM_CCOEFF_NORMED)[0][0])
    corr = max(0.0, min(1.0, (corr + 1.0) / 2.0))

    patch_hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    prev_hist = cv2.calcHist([preview], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(patch_hist, patch_hist)
    cv2.normalize(prev_hist, prev_hist)
    hist_score = float(cv2.compareHist(patch_hist, prev_hist, cv2.HISTCMP_CORREL))
    hist_score = max(0.0, min(1.0, (hist_score + 1.0) / 2.0))

    patch_edges = cv2.Canny(patch_gray, 60, 180)
    preview_edges = cv2.Canny(preview_gray, 60, 180)
    edge_score = float(cv2.matchTemplate(patch_edges, preview_edges, cv2.TM_CCOEFF_NORMED)[0][0])
    edge_score = max(0.0, min(1.0, (edge_score + 1.0) / 2.0))
    return (corr * 0.45) + (hist_score * 0.4) + (edge_score * 0.15)


def _silhouette_similarity(patch_bgr: np.ndarray | None, silhouette_bgr: np.ndarray | None) -> float:
    if patch_bgr is None or silhouette_bgr is None or patch_bgr.size == 0 or silhouette_bgr.size == 0:
        return 0.0
    patch_gray = cv2.cvtColor(cv2.resize(patch_bgr, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    silhouette_gray = cv2.cvtColor(cv2.resize(silhouette_bgr, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    patch_mask = cv2.Canny(patch_gray, 60, 180)
    silhouette_mask = cv2.Canny(silhouette_gray, 60, 180)
    score = float(cv2.matchTemplate(patch_mask, silhouette_mask, cv2.TM_CCOEFF_NORMED)[0][0])
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _entry_alias_tokens(entry: LoadedCatalogEntry) -> tuple[str, ...]:
    tokens = {entry.entity_id.lower(), entry.display_name.lower()}
    tokens.update(alias.lower() for alias in entry.aliases)
    tokens.update(name.lower() for name in entry.template_names)
    return tuple(sorted(token for token in tokens if token))


def _catalog_match_score(entry: LoadedCatalogEntry, *, patch: np.ndarray | None, profile: DetectorProfile) -> float:
    texture_score = _patch_similarity(patch, entry.preview_bgr)
    silhouette_score = _silhouette_similarity(patch, entry.silhouette_bgr)
    score = texture_score
    if silhouette_score > 0.0:
        score = max(score, texture_score * 0.7 + silhouette_score * 0.3)
    scene_tags = {_normalize_tag(tag) for tag in entry.scene_tags}
    if scene_tags and not any(tag in scene_tags for tag in profile.tags):
        score *= 0.6 if profile.scene_id != "FinalBossMap" else 0.35
    return score


def _classify_box_against_entries(
    frame_bgr: np.ndarray,
    proposal: LabeledBox,
    *,
    entries: tuple[LoadedCatalogEntry, ...],
    profile: DetectorProfile,
    threshold: float,
) -> tuple[LoadedCatalogEntry | None, float, str]:
    if not entries:
        return None, 0.0, "screen_cv"
    patch = _extract_patch(frame_bgr, proposal.rect)
    best_entry = None
    best_score = 0.0
    second_best = 0.0
    for entry in entries:
        current_score = _catalog_match_score(entry, patch=patch, profile=profile)
        if current_score > best_score:
            second_best = best_score
            best_score = current_score
            best_entry = entry
        elif current_score > second_best:
            second_best = current_score
    if best_entry is None or best_score < threshold:
        return None, 0.0, "screen_cv"
    source = "asset_catalog"
    if best_entry.silhouette_bgr is not None and abs(best_score - second_best) <= 0.08:
        source = "asset_catalog+silhouette"
    return best_entry, best_score, source


def _match_catalog_entry(name: str, entries: tuple[LoadedCatalogEntry, ...]) -> LoadedCatalogEntry | None:
    lowered = name.lower()
    for item in entries:
        aliases = _entry_alias_tokens(item)
        if lowered in aliases:
            return item
        if any(alias in lowered for alias in aliases):
            return item
    return None


def _profile_from_probe(probe_result: ProbeResult | None) -> DetectorProfile:
    scene_value = None
    if probe_result is not None:
        scene_value = probe_result.scene_id or probe_result.biome_or_scene_id
    normalized = _normalize_tag(scene_value)
    tags = {"generatedmap"}
    scene_id = "GeneratedMap"
    allow_loot = True
    allow_shrines = True
    allow_chests = True
    boss_emphasis = 1.0
    hazard_emphasis = 1.0

    if normalized in {"finalbossmap", "finalboss"}:
        scene_id = "FinalBossMap"
        tags = {"finalbossmap", "finalboss", "boss"}
        allow_loot = False
        allow_shrines = False
        allow_chests = False
        boss_emphasis = 1.35
        hazard_emphasis = 1.35
    elif normalized:
        scene_id = str(scene_value)
        tags.add(normalized)
    if probe_result is not None and probe_result.is_in_crypt:
        tags.add("crypt")

    return DetectorProfile(
        scene_id=scene_id,
        tags=tuple(sorted(tags)),
        allow_loot=allow_loot,
        allow_shrines=allow_shrines,
        allow_chests=allow_chests,
        boss_emphasis=boss_emphasis,
        hazard_emphasis=hazard_emphasis,
    )


def enemy_proposals(
    frame_bgr: np.ndarray,
    *,
    enemy_hsv_lower: tuple[int, int, int],
    enemy_hsv_upper: tuple[int, int, int],
    enemy_min_area: float,
) -> list[LabeledBox]:
    return _dedupe_boxes(
        _color_mask_boxes(
            frame_bgr,
            enemy_hsv_lower,
            enemy_hsv_upper,
            enemy_min_area,
            label="enemy",
        )
    )


def enemy_classification(
    frame_bgr: np.ndarray,
    *,
    proposals: list[LabeledBox],
    catalogs: CuratedCatalogs | None,
    mode: str,
    profile: DetectorProfile,
) -> list[TrackedEntity]:
    if not proposals:
        return []
    mode = str(mode or "hybrid").lower()
    entries = tuple(item for item in (catalogs.enemies if catalogs else ()) if item.preview_bgr is not None)
    output: list[TrackedEntity] = []
    for proposal in proposals:
        label = proposal.label
        entity_id = None
        source = "screen_cv"
        score = float(proposal.score)
        threat_tier = 1.0
        family = None
        variant = None
        metadata: dict[str, object] = {}
        if mode != "off" and entries:
            threshold = 0.34 if mode == "hybrid" else 0.26
            best_entry, best_score, best_source = _classify_box_against_entries(
                frame_bgr,
                proposal,
                entries=entries,
                profile=profile,
                threshold=threshold,
            )
            if best_entry is not None:
                label = best_entry.display_name
                entity_id = best_entry.entity_id
                source = best_source
                threat_tier = best_entry.threat_tier * (profile.boss_emphasis if best_entry.kind == "boss" else 1.0)
                score = max(score, float(best_score))
                family = best_entry.family
                variant = best_entry.variant
                metadata = best_entry.metadata
        output.append(
            TrackedEntity(
                label=label,
                rect=proposal.rect,
                score=score,
                source=source,
                entity_id=entity_id,
                threat_tier=threat_tier,
                family=family,
                variant=variant,
                metadata=metadata,
            )
        )
    return _dedupe_boxes(output)


def projectile_detection(
    frame_bgr: np.ndarray,
    *,
    templates: dict[str, np.ndarray] | None = None,
) -> list[LabeledBox]:
    boxes: list[LabeledBox] = []
    boxes.extend(
        _multi_color_mask_boxes(
            frame_bgr,
            [
                ((0, 120, 120), (12, 255, 255)),
                ((165, 120, 120), (179, 255, 255)),
                ((12, 100, 120), (28, 255, 255)),
                ((90, 120, 120), (130, 255, 255)),
            ],
            min_area=90.0,
            label="projectile",
        )
    )
    boxes.extend(
        _template_boxes(
            frame_bgr,
            templates,
            name_hints=("projectile", "orb", "arrow", "spike", "blood", "rocket", "flask"),
            threshold=0.54,
        )
    )
    return _dedupe_boxes(
        [
            LabeledBox(label="projectile", rect=item.rect, score=float(getattr(item, "score", 0.0)))
            for item in boxes
        ]
    )


def projectile_classification(
    frame_bgr: np.ndarray,
    *,
    proposals: list[LabeledBox],
    catalogs: CuratedCatalogs | None,
    profile: DetectorProfile,
    mode: str,
) -> tuple[list[TrackedEntity], list[WorldObject]]:
    if not proposals:
        return [], []
    mode = str(mode or "hybrid").lower()
    entries = tuple(item for item in (catalogs.projectiles if catalogs else ()) if item.preview_bgr is not None)
    projectile_classes: list[TrackedEntity] = []
    hazards: list[WorldObject] = []
    for proposal in proposals:
        label = "Projectile"
        entity_id = "projectile"
        source = "screen_cv"
        score = float(proposal.score)
        threat_tier = 1.6 * profile.hazard_emphasis
        family = "projectile"
        variant = None
        metadata: dict[str, object] = {}
        matched = None
        if mode != "off" and entries:
            threshold = 0.28 if mode == "hybrid" else 0.22
            matched, best_score, best_source = _classify_box_against_entries(
                frame_bgr,
                proposal,
                entries=entries,
                profile=profile,
                threshold=threshold,
            )
            if matched is not None:
                label = matched.display_name
                entity_id = matched.entity_id
                source = best_source
                score = max(score, float(best_score))
                threat_tier = max(threat_tier, matched.threat_tier * profile.hazard_emphasis)
                family = matched.family or matched.entity_id
                variant = matched.variant
                metadata = matched.metadata
        if matched is not None and (matched.hazard_kind or matched.kind in {"hazard", "world_hazard"}):
            hazards.append(
                WorldObject(
                    label=label,
                    rect=proposal.rect,
                    score=score,
                    source=source,
                    entity_id=entity_id,
                    poi_type=matched.poi_type,
                    family=family,
                    variant=variant,
                    hazard_kind=matched.hazard_kind or "hazard",
                    icon_id=matched.icon_id,
                    metadata=metadata,
                )
            )
        else:
            projectile_classes.append(
                TrackedEntity(
                    label=label,
                    rect=proposal.rect,
                    score=score,
                    source=source,
                    entity_id=entity_id,
                    threat_tier=threat_tier,
                    family=family,
                    variant=variant,
                    metadata=metadata,
                )
            )
    return _dedupe_boxes(projectile_classes), _dedupe_boxes(hazards)


def world_object_detection(
    frame_bgr: np.ndarray,
    *,
    templates: dict[str, np.ndarray] | None,
    catalogs: CuratedCatalogs | None,
    interact_threshold: float,
    profile: DetectorProfile,
) -> tuple[list[LabeledBox], list[WorldObject]]:
    hints = {
        "chest",
        "coin",
        "loot",
        "foliant",
        "tome",
        "shrine",
        "portal",
        "altar",
        "door",
        "boss",
        "key",
        "book",
        "crypt",
        "grave",
        "cage",
        "objective",
    }
    for entry in catalogs.world if catalogs else ():
        hints.update(alias.lower() for alias in entry.aliases)
        hints.update(template.lower() for template in entry.template_names)
    hits = _template_boxes(
        frame_bgr,
        templates,
        name_hints=tuple(sorted(hints)),
        threshold=interact_threshold,
    )
    world_objects: list[WorldObject] = []
    for hit in hits:
        matched = _match_catalog_entry(hit.label, catalogs.world if catalogs else ())
        if matched is not None:
            if not profile.allow_loot and matched.poi_type == "loot":
                continue
            if not profile.allow_shrines and (matched.poi_type == "buff" or "shrine" in _entry_alias_tokens(matched)):
                continue
            if not profile.allow_chests and "chest" in _entry_alias_tokens(matched):
                continue
            label = matched.display_name
            entity_id = matched.entity_id
            poi_type = matched.poi_type
            source = "runtime_template+catalog"
            family = matched.family
            variant = matched.variant
            hazard_kind = matched.hazard_kind
            icon_id = matched.icon_id
            metadata = matched.metadata
        else:
            label = _friendly_name(hit.label)
            entity_id = None
            poi_type = None
            source = "runtime_template"
            family = None
            variant = None
            hazard_kind = None
            icon_id = None
            metadata = {}
        world_objects.append(
            WorldObject(
                label=label,
                rect=hit.rect,
                score=float(hit.score),
                source=source,
                entity_id=entity_id,
                poi_type=poi_type,
                family=family,
                variant=variant,
                hazard_kind=hazard_kind,
                icon_id=icon_id,
                metadata=metadata,
            )
        )
    return hits, _dedupe_boxes(world_objects)


def _detect_player_marker(minimap_bgr: np.ndarray) -> tuple[tuple[float, float] | None, float]:
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array((0, 0, 180)), np.array((179, 70, 255)))
    color_mask = cv2.inRange(hsv, np.array((20, 80, 150)), np.array((120, 255, 255)))
    mask = cv2.bitwise_or(white_mask, color_mask)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None, 0.0
    cx = float(xs.mean()) / max(1.0, minimap_bgr.shape[1] - 1)
    cy = float(ys.mean()) / max(1.0, minimap_bgr.shape[0] - 1)
    confidence = min(1.0, float(xs.size) / max(20.0, minimap_bgr.shape[0] * minimap_bgr.shape[1] * 0.02))
    return (cx, cy), confidence


def _detect_icon_pois(minimap_bgr: np.ndarray, catalogs: CuratedCatalogs | None) -> tuple[MapPoi, ...]:
    if catalogs is None:
        return ()
    found: list[MapPoi] = []
    region = (0, 0, minimap_bgr.shape[1], minimap_bgr.shape[0])
    for entry in catalogs.minimap_icon_entries():
        icon = entry.minimap_icon_bgr
        if icon is None:
            continue
        ok, (cx, cy), score = find_in_region(
            minimap_bgr,
            icon,
            region,
            threshold=0.42,
            scales=(0.8, 1.0, 1.2),
        )
        if not ok:
            continue
        found.append(
            MapPoi(
                label=entry.display_name,
                pos_norm=(
                    float(cx) / max(1.0, minimap_bgr.shape[1] - 1),
                    float(cy) / max(1.0, minimap_bgr.shape[0] - 1),
                ),
                score=float(score),
                source="icon_atlas",
                poi_type=entry.poi_type,
                icon_id=entry.icon_id or entry.entity_id,
            )
        )
    return _dedupe_pois(found, distance_thr=0.06)


def _detect_pois(minimap_bgr: np.ndarray) -> tuple[MapPoi, ...]:
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    poi_specs = [
        ("BossMarker", "boss", "boss_marker", ((0, 120, 120), (10, 255, 255))),
        ("Portal", "exit", "portal", ((95, 100, 120), (130, 255, 255))),
        ("Shrine", "buff", "shrine", ((18, 80, 160), (40, 255, 255))),
        ("Chest", "loot", "chest", ((10, 70, 120), (25, 255, 255))),
        ("Key", "key", "key", ((75, 90, 120), (95, 255, 255))),
        ("Objective", "objective", "objective", ((0, 0, 220), (179, 45, 255))),
        ("CryptExit", "transition", "crypt_exit", ((135, 80, 120), (165, 255, 255))),
    ]
    output: list[MapPoi] = []
    for label, poi_type, icon_id, (lower, upper) in poi_specs:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 8.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            pos = (
                (x + w / 2.0) / max(1.0, minimap_bgr.shape[1] - 1),
                (y + h / 2.0) / max(1.0, minimap_bgr.shape[0] - 1),
            )
            score = min(1.0, area / 50.0)
            output.append(
                MapPoi(label=label, pos_norm=pos, score=score, source="ui_cv", poi_type=poi_type, icon_id=icon_id)
            )
    return _dedupe_pois(output, distance_thr=0.08)


def _detect_full_map_overlay(frame_bgr: np.ndarray) -> tuple[bool, float]:
    h, w = frame_bgr.shape[:2]
    x0 = int(0.15 * w)
    y0 = int(0.15 * h)
    x1 = int(0.85 * w)
    y1 = int(0.85 * h)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dark_ratio = float((gray < 80).mean())
    std = float(gray.std())
    score = min(1.0, max(0.0, dark_ratio * 1.1 + (0.25 if std > 25.0 else 0.0)))
    return bool(dark_ratio >= 0.55 and std >= 15.0), score


def minimap_state_detection(
    frame_bgr: np.ndarray,
    *,
    templates: dict[str, np.ndarray] | None,
    regions: dict[str, tuple[int, int, int, int]] | None,
    minimap_enabled: bool,
    catalogs: CuratedCatalogs | None,
) -> MapState:
    if not minimap_enabled or not regions or "REG_MINIMAP" not in regions:
        return MapState()
    map_open, map_open_score = _detect_full_map_overlay(frame_bgr)
    x, y, w, h = regions["REG_MINIMAP"]
    minimap = frame_bgr[y : y + h, x : x + w]
    if minimap.size == 0:
        return MapState(map_open=map_open, minimap_visible=False, source="ui_cv", confidence=map_open_score)

    visible = False
    confidence = 0.0
    if templates and "tpl_minimap" in templates:
        visible, _, confidence = find_in_region(
            frame_bgr,
            templates["tpl_minimap"],
            regions["REG_MINIMAP"],
            threshold=0.35,
            scales=(1.0,),
        )
    if not visible:
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        confidence = max(confidence, min(1.0, float(gray.std()) / 40.0))
        visible = confidence >= 0.18

    player_norm, player_confidence = _detect_player_marker(minimap)
    icon_pois = _detect_icon_pois(minimap, catalogs)
    color_pois = _detect_pois(minimap)
    pois = _dedupe_pois(list(icon_pois) + list(color_pois), distance_thr=0.08)
    final_confidence = max(confidence, player_confidence, map_open_score, max((poi.score for poi in pois), default=0.0))
    source = "icon_atlas" if icon_pois else ("ui_cv" if (visible or map_open or player_norm is not None or pois) else "none")
    return MapState(
        map_open=bool(map_open),
        minimap_visible=bool(visible),
        player_norm=player_norm,
        pois=pois,
        source=source,
        confidence=final_confidence,
    )


def state_fusion(
    *,
    enemy_classes: list[TrackedEntity],
    projectile_classes: list[TrackedEntity],
    world_objects: list[WorldObject],
    hazards: list[WorldObject],
    projectiles: list[LabeledBox],
    map_state_cv: MapState,
    objective_ui: UiTextDetection,
    probe_result: ProbeResult | None,
    profile: DetectorProfile,
) -> tuple[PlayerPose, MapState, dict[str, str], dict[str, float], str]:
    sources = {
        "enemies": "asset_catalog" if any("asset_catalog" in item.source for item in enemy_classes) else "screen_cv",
        "world_objects": "runtime_template+catalog" if any("catalog" in item.source for item in world_objects) else "runtime_template",
        "projectiles": "asset_catalog" if any("asset_catalog" in item.source for item in projectile_classes) else "screen_cv",
        "hazards": "asset_catalog" if any("catalog" in item.source for item in hazards) else "screen_cv",
        "player_pose": map_state_cv.source,
        "map_state": map_state_cv.source,
        "objective": objective_ui.source,
        "scene_profile": profile.scene_id,
    }
    confidence = {
        "enemies": max([float(item.score) for item in enemy_classes], default=0.0),
        "world_objects": max([float(item.score) for item in world_objects], default=0.0),
        "projectiles": max([float(item.score) for item in projectile_classes], default=max([float(item.score) for item in projectiles], default=0.0)),
        "hazards": max([float(item.score) for item in hazards], default=0.0),
        "player_pose": float(map_state_cv.confidence),
        "map_state": float(map_state_cv.confidence),
        "objective": float(objective_ui.confidence) / 100.0 if objective_ui.confidence > 1.0 else float(objective_ui.confidence),
        "scene_profile": 1.0,
    }
    player_pose = PlayerPose(
        map_norm=map_state_cv.player_norm,
        world_pos=None,
        heading_deg=None,
        source=map_state_cv.source,
        confidence=float(map_state_cv.confidence if map_state_cv.player_norm is not None else 0.0),
    )
    scene_id = profile.scene_id
    objective = objective_ui.normalized
    map_state = MapState(
        map_open=map_state_cv.map_open,
        minimap_visible=map_state_cv.minimap_visible,
        player_norm=map_state_cv.player_norm,
        biome=map_state_cv.biome,
        scene_id=scene_id,
        active_room_id=map_state_cv.active_room_id,
        objective=objective,
        objective_confidence=confidence["objective"],
        pois=map_state_cv.pois,
        source=map_state_cv.source if objective is None else "ui_ocr",
        confidence=max(map_state_cv.confidence, confidence["objective"]),
    )
    memory_status = "disabled"
    if probe_result is not None:
        memory_status = str(probe_result.status)
        has_probe_data = any(
            value is not None
            for value in (
                probe_result.player_world_pos,
                probe_result.player_heading_deg,
                probe_result.map_open,
                probe_result.biome_or_scene_id,
                probe_result.scene_id,
                probe_result.active_room_or_node_id,
                probe_result.room_start,
                probe_result.room_end,
                probe_result.is_in_crypt,
                probe_result.graveyard_crypt_keys,
                probe_result.current_objective,
                probe_result.boss_spotted,
                probe_result.charged_shrines,
            )
        )
        if has_probe_data:
            objective = probe_result.current_objective or objective
            player_pose = PlayerPose(
                map_norm=player_pose.map_norm,
                world_pos=probe_result.player_world_pos,
                heading_deg=probe_result.player_heading_deg,
                source="external_memory",
                confidence=max(0.95, player_pose.confidence),
            )
            map_state = MapState(
                map_open=bool(probe_result.map_open) if probe_result.map_open is not None else bool(map_state_cv.map_open),
                minimap_visible=bool(map_state_cv.minimap_visible),
                player_norm=map_state_cv.player_norm,
                biome=probe_result.biome_or_scene_id or map_state_cv.biome,
                scene_id=probe_result.scene_id or scene_id,
                active_room_id=probe_result.active_room_or_node_id or map_state_cv.active_room_id,
                room_start=probe_result.room_start,
                room_end=probe_result.room_end,
                is_crypt=probe_result.is_in_crypt,
                objective=objective,
                objective_confidence=max(confidence["objective"], 0.95 if probe_result.current_objective else confidence["objective"]),
                boss_spotted=probe_result.boss_spotted,
                charged_shrines=probe_result.charged_shrines,
                graveyard_crypt_keys=probe_result.graveyard_crypt_keys,
                pois=map_state_cv.pois,
                source="external_memory" if (probe_result.scene_id or probe_result.current_objective) else map_state.source,
                confidence=max(0.95, map_state_cv.confidence),
            )
            sources["player_pose"] = "external_memory"
            sources["map_state"] = "external_memory"
            if probe_result.current_objective:
                sources["objective"] = "external_memory"
            confidence["player_pose"] = player_pose.confidence
            confidence["map_state"] = map_state.confidence
            confidence["objective"] = map_state.objective_confidence
    return player_pose, map_state, sources, confidence, memory_status


def analyze_scene(
    frame_bgr: np.ndarray,
    *,
    templates: dict[str, np.ndarray] | None = None,
    catalogs: CuratedCatalogs | None = None,
    regions: dict[str, tuple[int, int, int, int]] | None = None,
    probe_result: ProbeResult | None = None,
    objective_ui: UiTextDetection | None = None,
    grid_rows: int = 12,
    grid_cols: int = 20,
    enemy_hsv_lower: tuple[int, int, int] = (45, 80, 40),
    enemy_hsv_upper: tuple[int, int, int] = (85, 255, 255),
    enemy_min_area: float = 1200.0,
    interact_threshold: float = 0.65,
    enemy_classifier_mode: str = "hybrid",
    minimap_enabled: bool = True,
) -> dict[str, list]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    grid = _grid_edges(gray, grid_rows, grid_cols)
    profile = _profile_from_probe(probe_result)
    proposals = enemy_proposals(
        frame_bgr,
        enemy_hsv_lower=enemy_hsv_lower,
        enemy_hsv_upper=enemy_hsv_upper,
        enemy_min_area=enemy_min_area,
    )
    enemy_classes = enemy_classification(
        frame_bgr,
        proposals=proposals,
        catalogs=catalogs,
        mode=enemy_classifier_mode,
        profile=profile,
    )
    projectiles = projectile_detection(frame_bgr, templates=templates)
    projectile_classes, hazards = projectile_classification(
        frame_bgr,
        proposals=projectiles,
        catalogs=catalogs,
        profile=profile,
        mode=enemy_classifier_mode,
    )
    interactables, world_objects = world_object_detection(
        frame_bgr,
        templates=templates,
        catalogs=catalogs or CuratedCatalogs(),
        interact_threshold=interact_threshold,
        profile=profile,
    )
    objective_ui = objective_ui or UiTextDetection()
    map_state_cv = minimap_state_detection(
        frame_bgr,
        templates=templates,
        regions=regions,
        minimap_enabled=minimap_enabled,
        catalogs=catalogs,
    )
    player_pose, map_state, detection_sources, source_confidence, memory_status = state_fusion(
        enemy_classes=enemy_classes,
        projectile_classes=projectile_classes,
        world_objects=world_objects,
        hazards=hazards,
        projectiles=projectiles,
        map_state_cv=map_state_cv,
        objective_ui=objective_ui,
        probe_result=probe_result,
        profile=profile,
    )
    return {
        "grid": grid,
        "enemies": proposals,
        "enemy_classes": enemy_classes,
        "interactables": interactables,
        "world_objects": world_objects,
        "projectiles": projectiles,
        "projectile_classes": projectile_classes,
        "hazards": hazards,
        "player_pose": player_pose,
        "map_state": map_state,
        "objective_ui": objective_ui,
        "detector_profile": profile,
        "detection_sources": detection_sources,
        "source_confidence": source_confidence,
        "memory_probe_status": memory_status,
    }


def draw_recognition_overlay(
    frame_bgr: np.ndarray,
    analysis: dict[str, list],
    *,
    grid_alpha: float = 0.35,
    hud_alpha: float = 0.55,
    hud_values: dict | None = None,
    hud_regions: dict | None = None,
    region_overlays: list[RegionOverlay] | None = None,
    show_legend: bool = True,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    overlay = frame_bgr.copy()
    grid = analysis.get("grid", [])
    for cell in grid:
        x, y, w, h = cell.rect
        if cell.label == "surface":
            color = (60, 200, 60)
        elif cell.label == "obstacle":
            color = (40, 40, 220)
        else:
            color = (120, 120, 120)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 20, 20), 1)
        if w >= 40 and h >= 30:
            _draw_label(canvas, (x, y), cell.label, color)
    cv2.addWeighted(overlay, grid_alpha, canvas, 1 - grid_alpha, 0, canvas)

    enemy_boxes = analysis.get("enemy_classes") or analysis.get("enemies", [])
    for box in enemy_boxes:
        _draw_labeled_box(canvas, box.rect, _format_box_label(box), (0, 0, 255))
    projectile_boxes = analysis.get("projectile_classes") or analysis.get("projectiles", [])
    for box in projectile_boxes:
        _draw_labeled_box(canvas, box.rect, _format_box_label(box), (180, 0, 255))
    world_boxes = list(analysis.get("world_objects") or analysis.get("interactables", []))
    world_boxes.extend(analysis.get("hazards", []))
    for box in world_boxes:
        color = (40, 180, 255) if getattr(box, "hazard_kind", None) else (255, 140, 0)
        _draw_labeled_box(canvas, box.rect, _format_box_label(box), color)
    if region_overlays:
        for region in region_overlays:
            _draw_labeled_box(canvas, region.rect, region.label, region.color)
    if hud_values is not None:
        hud_overlay = canvas.copy()
        labels = _draw_hud_overlay(hud_overlay, frame_bgr, hud_values, hud_regions)
        cv2.addWeighted(hud_overlay, hud_alpha, canvas, 1 - hud_alpha, 0, canvas)
        for rect, label, color in labels:
            _draw_labeled_box(canvas, rect, label, color)
    if show_legend:
        _draw_default_legend(canvas)
    return canvas


def draw_hud_overlay_frame(
    frame_bgr: np.ndarray,
    *,
    hud_values: dict | None = None,
    hud_regions: dict | None = None,
    hud_alpha: float = 0.55,
) -> np.ndarray:
    if hud_values is None:
        hud_values = {"hp": None, "gold": None, "time": None}
    canvas = np.zeros_like(frame_bgr)
    overlay = canvas.copy()
    labels = _draw_hud_overlay(overlay, frame_bgr, hud_values, hud_regions)
    cv2.addWeighted(overlay, hud_alpha, canvas, 1 - hud_alpha, 0, canvas)
    for rect, label, color in labels:
        _draw_labeled_box(canvas, rect, label, color)
    return canvas


def _draw_hud_overlay(
    overlay, frame_bgr, hud_values, hud_regions
) -> list[tuple[tuple[int, int, int, int], str, tuple[int, int, int]]]:
    labels = []
    for key, value in hud_values.items():
        rect = _resolve_hud_region(frame_bgr, hud_regions, key)
        if rect is None:
            continue
        x, y, w, h = rect
        color = (0, 255, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        value_label = value if value is not None else "?"
        label = f"{key}:{value_label} ({x},{y},{w},{h})"
        labels.append((rect, label, color))
    return labels


def _resolve_hud_region(frame_bgr, regions, key):
    if regions:
        reg_key = f"REG_HUD_{key.upper()}"
        if reg_key in regions:
            return regions[reg_key]
    if key not in DEFAULT_HUD_REGIONS:
        return None
    h, w = frame_bgr.shape[:2]
    rx, ry, rw, rh = DEFAULT_HUD_REGIONS[key]
    return (int(rx * w), int(ry * h), int(rw * w), int(rh * h))


def _format_box_label(item) -> str:
    label = str(getattr(item, "label", "object"))
    family = getattr(item, "family", None)
    variant = getattr(item, "variant", None)
    hazard_kind = getattr(item, "hazard_kind", None)
    source = getattr(item, "source", None)
    parts = [label]
    if family and str(family).lower() != label.lower():
        parts.append(str(family))
    if variant:
        parts.append(str(variant))
    if hazard_kind:
        parts.append(str(hazard_kind))
    text = "/".join(parts[:3])
    if source:
        short = str(source).replace("runtime_template+", "tpl+")
        return f"{text} [{short}]"
    return text


def _draw_labeled_box(
    frame_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
):
    x, y, w, h = rect
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    text = label
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    tx = x
    ty = max(0, y - th - baseline - 4)
    _draw_label(
        frame_bgr,
        (tx, ty),
        text,
        color,
        font=font,
        scale=scale,
        thickness=thickness,
        text_size=(tw, th),
        baseline=baseline,
    )


def _draw_default_legend(frame_bgr: np.ndarray) -> None:
    items = [
        ("surface", (60, 200, 60)),
        ("obstacle", (40, 40, 220)),
        ("enemy", (0, 0, 255)),
        ("projectile", (180, 0, 255)),
        ("world", (255, 140, 0)),
        ("hazard", (40, 180, 255)),
        ("hud_roi", (0, 255, 255)),
    ]
    x0 = 10
    y0 = frame_bgr.shape[0] - 18 * (len(items) + 1) - 10
    y0 = max(10, y0)
    cv2.rectangle(frame_bgr, (x0 - 6, y0 - 18), (x0 + 260, y0 + 18 * len(items)), (0, 0, 0), -1)
    cv2.rectangle(frame_bgr, (x0 - 6, y0 - 18), (x0 + 260, y0 + 18 * len(items)), (255, 255, 255), 1)
    cv2.putText(
        frame_bgr,
        "Legend",
        (x0, y0 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y = y0 + 12
    for label, color in items:
        cv2.rectangle(frame_bgr, (x0, y - 9), (x0 + 10, y + 1), color, -1)
        cv2.putText(
            frame_bgr,
            label,
            (x0 + 16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18


def _draw_label(
    frame_bgr: np.ndarray,
    origin: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.5,
    thickness: int = 1,
    text_size: tuple[int, int] | None = None,
    baseline: int | None = None,
    font_path: str | None = None,
):
    if _has_non_ascii(text):
        font_size = max(10, int(20 * scale))
        _draw_label_pil(frame_bgr, origin, text, color, font_path, font_size)
        return
    if text_size is None or baseline is None:
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    else:
        tw, th = text_size
    tx, ty = origin
    cv2.rectangle(
        frame_bgr,
        (tx, ty),
        (tx + tw + 6, ty + th + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        frame_bgr,
        text,
        (tx + 3, ty + th + 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_label_pil(
    frame_bgr: np.ndarray,
    origin: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font_path: str | None,
    font_size: int,
) -> None:
    if frame_bgr.size == 0:
        return
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_pil_font(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = 3
    pad_y = 2
    tx, ty = origin
    rect = (tx, ty, tx + text_w + pad_x * 2, ty + text_h + pad_y * 2)
    bg_color = (color[2], color[1], color[0])
    draw.rectangle(rect, fill=bg_color)
    text_pos = (tx + pad_x - bbox[0], ty + pad_y - bbox[1])
    draw.text(text_pos, text, font=font, fill=(255, 255, 255))
    frame_bgr[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _load_pil_font(font_path: str | None, font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            pass
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)
