from __future__ import annotations

from typing import Any, Optional


def point_in_rect(x: int, y: int, rect: Optional[tuple[int, int, int, int]]) -> bool:
    if rect is None:
        return False
    rx, ry, rw, rh = rect
    return rx <= x <= (rx + rw) and ry <= y <= (ry + rh)


def handle_overlay_mouse_event(cv2, event: int, x: int, y: int, state: Any) -> None:
    if not isinstance(state, dict):
        return
    if event not in {
        getattr(cv2, "EVENT_LBUTTONDOWN", -1),
        getattr(cv2, "EVENT_LBUTTONUP", -2),
    }:
        return
    rects = state.get("rects", {})
    if point_in_rect(int(x), int(y), rects.get("toggle")):
        state["toggle"] = True
    elif point_in_rect(int(x), int(y), rects.get("panic")):
        state["panic"] = True


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
    navigation_context=None,
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
            "enemy_classes": analysis.get("enemy_classes", []),
            "interactables": analysis.get("interactables", []),
            "projectiles": analysis.get("projectiles", []),
            "projectile_classes": analysis.get("projectile_classes", []),
            "world_objects": analysis.get("world_objects", []),
            "hazards": analysis.get("hazards", []),
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
    map_state = getattr(snapshot, "map_state", None)
    player_pose = getattr(snapshot, "player_pose", None)
    map_open = getattr(map_state, "map_open", False) if map_state is not None else False
    minimap_visible = getattr(map_state, "minimap_visible", False) if map_state is not None else False
    player_norm = getattr(player_pose, "map_norm", None) if player_pose is not None else None
    player_world = getattr(player_pose, "world_pos", None) if player_pose is not None else None
    enemy_classes = getattr(snapshot, "enemy_classes", [])
    projectile_classes = getattr(snapshot, "projectile_classes", [])
    world_objects = getattr(snapshot, "world_objects", [])
    hazards = getattr(snapshot, "hazards", [])
    scene_id = getattr(map_state, "scene_id", None) if map_state is not None else None
    objective = getattr(map_state, "objective", None) if map_state is not None else None
    is_crypt = getattr(map_state, "is_crypt", None) if map_state is not None else None
    map_pos = "?"
    if player_norm is not None:
        map_pos = f"{player_norm[0]:.2f},{player_norm[1]:.2f}"
    world_pos = "?"
    if player_world is not None:
        world_pos = f"{player_world[0]:.1f},{player_world[1]:.1f},{player_world[2]:.1f}"
    detection_sources = getattr(snapshot, "detection_sources", {}) or {}
    time_fail = hud_values.get("time_fail_reason") or "ok"
    kills_fail = hud_values.get("kills_fail_reason") or "ok"
    nav_terrain = getattr(navigation_context, "terrain_kind", "unknown")
    nav_escape = getattr(navigation_context, "escape_lane", None) or "none"
    nav_conf = format_float(getattr(navigation_context, "nav_confidence", None), digits=2)
    nav_drop = format_float(getattr(navigation_context, "drop_risk", None), digits=2)
    nav_jump_gate = getattr(navigation_context, "jump_gate", "not_evaluated")
    nav_slide_gate = getattr(navigation_context, "slide_gate", "not_evaluated")
    nav_slope_source = getattr(navigation_context, "slope_source", "none")
    nav_slope_delta = format_float(getattr(navigation_context, "slope_delta_z", None), digits=3)
    lines = [
        f"mode={mode.value}",
        f"reason={action_reason}",
        f"hp={hp_pct} lvl={snapshot.lvl} kills={snapshot.kills} time={snapshot.time_s}",
        f"gold={hud_values.get('gold')}  time_ocr={format_float(hud_values.get('time_ocr_ms'))}ms ({time_fail})",
        f"kills_ocr={format_float(hud_values.get('kills_ocr_ms'))}ms ({kills_fail})",
        f"nav terrain={nav_terrain} escape={nav_escape} conf={nav_conf} drop={nav_drop}",
        f"nav jump={nav_jump_gate} slide={nav_slide_gate} slope={nav_slope_source}:{nav_slope_delta}",
        f"enemies={len(snapshot.enemies)} enemy_classes={len(enemy_classes)} projectiles={len(snapshot.projectiles)} projectile_classes={len(projectile_classes)} world={len(world_objects)} hazards={len(hazards)}",
        f"dead={snapshot.is_dead} upgrade={snapshot.is_upgrade} safe_sector={snapshot.safe_sector}",
        f"scene={scene_id} crypt={is_crypt} objective={objective}",
        f"map_open={map_open} minimap={minimap_visible} map_pos={map_pos} world_pos={world_pos}",
        f"probe={getattr(snapshot, 'memory_probe_status', 'disabled')} sources={detection_sources}",
        "legend: green=surface blue=obstacle red=enemy magenta=projectile orange=world cyan=hazard yellow=hud_roi",
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
        (1240, min(h - 8, 8 + 24 * (len(lines) + 1))),
        (20, 20, 20) if transparent_canvas else (0, 0, 0),
        2,
    )
    return canvas, button_rects
