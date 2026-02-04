from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from megabonk_bot.hud import DEFAULT_HUD_REGIONS
from megabonk_bot.vision import find_in_region


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
            label = "поверхность"
        elif score >= high_thr:
            label = "препятствие"
        else:
            label = "неизвестно"
        labeled.append(GridCell(label=label, rect=(x0, y0, cw, ch), score=score))
    return labeled


def _color_mask_boxes(
    frame_bgr: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    min_area: float,
    label: str,
) -> list[LabeledBox]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
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
    for name, tpl in templates.items():
        if not any(hint in name.lower() for hint in name_hints):
            continue
        ok, (cx, cy), score = find_in_region(
            frame_bgr,
            tpl,
            region,
            threshold=threshold,
            scales=(1.0,),
        )
        if not ok:
            continue
        th, tw = tpl.shape[:2]
        x0 = max(0, int(cx - tw / 2))
        y0 = max(0, int(cy - th / 2))
        found.append(
            LabeledBox(label=_friendly_name(name), rect=(x0, y0, tw, th), score=score)
        )
    return found


def _friendly_name(raw: str) -> str:
    lowered = raw.lower()
    if "chest" in lowered:
        return "сундук"
    if "coin" in lowered or "gold" in lowered:
        return "монета"
    if "foliant" in lowered or "tome" in lowered or "book" in lowered:
        return "свиток"
    if "altar" in lowered:
        return "алтарь"
    if "door" in lowered:
        return "дверь"
    return raw.replace("tpl_", "").replace("_", " ")


def analyze_scene(
    frame_bgr: np.ndarray,
    *,
    templates: dict[str, np.ndarray] | None = None,
    grid_rows: int = 12,
    grid_cols: int = 20,
    enemy_hsv_lower: tuple[int, int, int] = (45, 80, 40),
    enemy_hsv_upper: tuple[int, int, int] = (85, 255, 255),
    enemy_min_area: float = 1200.0,
    interact_hints: tuple[str, ...] = (
        "chest",
        "coin",
        "loot",
        "foliant",
        "tome",
        "katana",
        "dexec",
    ),
    interact_threshold: float = 0.65,
) -> dict[str, list]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    grid = _grid_edges(gray, grid_rows, grid_cols)
    enemies = _color_mask_boxes(
        frame_bgr,
        enemy_hsv_lower,
        enemy_hsv_upper,
        enemy_min_area,
        label="враг",
    )
    interactables = _template_boxes(
        frame_bgr, templates, name_hints=interact_hints, threshold=interact_threshold
    )
    return {
        "grid": grid,
        "enemies": enemies,
        "interactables": interactables,
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
) -> np.ndarray:
    canvas = frame_bgr.copy()
    overlay = frame_bgr.copy()
    grid = analysis.get("grid", [])
    for cell in grid:
        x, y, w, h = cell.rect
        if cell.label == "поверхность":
            color = (60, 200, 60)
        elif cell.label == "препятствие":
            color = (40, 40, 220)
        else:
            color = (120, 120, 120)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 20, 20), 1)
        if w >= 40 and h >= 30:
            _draw_label(canvas, (x, y), cell.label, color)
    cv2.addWeighted(overlay, grid_alpha, canvas, 1 - grid_alpha, 0, canvas)

    for box in analysis.get("enemies", []):
        _draw_labeled_box(canvas, box.rect, box.label, (0, 0, 255))
    for box in analysis.get("interactables", []):
        _draw_labeled_box(canvas, box.rect, box.label, (255, 140, 0))
    if region_overlays:
        for region in region_overlays:
            _draw_labeled_box(canvas, region.rect, region.label, region.color)
    if hud_values is not None:
        hud_overlay = canvas.copy()
        _draw_hud_overlay(canvas, hud_overlay, frame_bgr, hud_values, hud_regions)
        cv2.addWeighted(hud_overlay, hud_alpha, canvas, 1 - hud_alpha, 0, canvas)
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
    _draw_hud_overlay(canvas, overlay, frame_bgr, hud_values, hud_regions)
    cv2.addWeighted(overlay, hud_alpha, canvas, 1 - hud_alpha, 0, canvas)
    return canvas


def _draw_hud_overlay(canvas, overlay, frame_bgr, hud_values, hud_regions):
    for key, value in hud_values.items():
        rect = _resolve_hud_region(frame_bgr, hud_regions, key)
        if rect is None:
            continue
        x, y, w, h = rect
        color = (0, 255, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        value_label = value if value is not None else "?"
        label = f"{key}:{value_label} ({x},{y},{w},{h})"
        _draw_labeled_box(canvas, rect, label, color)


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
):
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
