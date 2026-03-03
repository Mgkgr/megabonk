from __future__ import annotations

from typing import Any, Optional


def point_in_rect(x: int, y: int, rect: Optional[tuple[int, int, int, int]]) -> bool:
    if rect is None:
        return False
    rx, ry, rw, rh = rect
    return rx <= x <= (rx + rw) and ry <= y <= (ry + rh)


def build_overlay_button_rects(frame_w: int) -> dict[str, tuple[int, int, int, int]]:
    button_w = 170
    button_h = 34
    margin = 12
    x = max(margin, int(frame_w) - button_w - margin)
    return {
        "toggle": (x, 10, button_w, button_h),
        "panic": (x, 50, button_w, button_h),
    }


def draw_overlay_button(
    cv2,
    canvas,
    rect: tuple[int, int, int, int],
    label: str,
    *,
    bg_color: tuple[int, int, int],
    border_color: tuple[int, int, int] = (255, 255, 255),
):
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)
    cv2.putText(
        canvas,
        label,
        (x + 10, y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def format_float(value: Any, *, digits: int = 1) -> str:
    if value is None:
        return "?"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def draw_runtime_overlay(
    cv2,
    frame,
    analysis,
    snapshot,
    *,
    mode,
    action_reason: str,
    hud_values: dict[str, Any],
    hud_regions: dict[str, Any],
    transparent_canvas: bool = False,
):
    from megabonk_bot.recognition import draw_recognition_overlay

    hud_overlay_values = {
        "time": hud_values.get("time"),
        "kills": hud_values.get("kills"),
        "lvl": hud_values.get("lvl"),
        "gold": hud_values.get("gold"),
        "hp": hud_values.get("hp_ratio"),
    }
    base_frame = frame if not transparent_canvas else frame * 0
    overlay_analysis = analysis
    if transparent_canvas:
        overlay_analysis = {
            "grid": [],
            "enemies": analysis.get("enemies", []),
            "interactables": analysis.get("interactables", []),
            "projectiles": analysis.get("projectiles", []),
        }
    canvas = draw_recognition_overlay(
        base_frame,
        overlay_analysis,
        hud_values=hud_overlay_values,
        hud_regions=hud_regions,
    )
    h, _ = canvas.shape[:2]
    button_rects = build_overlay_button_rects(canvas.shape[1])
    if mode.value == "ACTIVE":
        toggle_label = "STOP (F8/S)"
        toggle_color = (20, 40, 160)
    else:
        toggle_label = "START (F8/S)"
        toggle_color = (20, 120, 20)
    draw_overlay_button(cv2, canvas, button_rects["toggle"], toggle_label, bg_color=toggle_color)
    draw_overlay_button(cv2, canvas, button_rects["panic"], "PANIC (F12/P)", bg_color=(0, 0, 180))

    hp_pct = (
        "?"
        if snapshot.hp_ratio is None
        else f"{float(snapshot.hp_ratio) * 100.0:.1f}%"
    )
    time_fail = hud_values.get("time_fail_reason") or "ok"
    kills_fail = hud_values.get("kills_fail_reason") or "ok"
    lines = [
        f"mode={mode.value}",
        f"reason={action_reason}",
        f"hp={hp_pct} lvl={snapshot.lvl} kills={snapshot.kills} time={snapshot.time_s}",
        f"gold={hud_values.get('gold')}  time_ocr={format_float(hud_values.get('time_ocr_ms'))}ms ({time_fail})",
        f"kills_ocr={format_float(hud_values.get('kills_ocr_ms'))}ms ({kills_fail})",
        f"enemies={len(snapshot.enemies)} obstacles={len(snapshot.obstacles)} projectiles={len(snapshot.projectiles)}",
        f"dead={snapshot.is_dead} upgrade={snapshot.is_upgrade} safe_sector={snapshot.safe_sector}",
        "legend: green=surface blue=obstacle red=enemy orange=interactable cyan=hud_roi",
        "controls: click START/STOP or PANIC | Q/Esc=quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    cv2.rectangle(
        canvas,
        (8, 8),
        (1060, min(h - 8, 8 + 24 * (len(lines) + 1))),
        (20, 20, 20) if transparent_canvas else (0, 0, 0),
        2,
    )
    return canvas, button_rects
