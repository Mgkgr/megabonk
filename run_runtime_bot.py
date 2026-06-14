from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()

import argparse
import ctypes
import logging
import time
from pathlib import Path
from typing import Any, Optional

from megabonk_bot.asset_catalog import load_curated_catalogs
from megabonk_bot.config import DEFAULT_CONFIG, dump_default_config_yaml, load_config
from megabonk_bot.memory_probe import ExternalProcessProbe, NullProbe, WorldStateProbe
from megabonk_bot.navigation import StatefulNavigationPlanner
from megabonk_bot.runtime.event_logger import (
    JsonlEventLogger,
    RUNTIME_EVENT_SCHEMA_VERSION,
    build_runtime_event as _build_runtime_event,
)
from megabonk_bot.runtime.hud_cache import HudTelemetryCache
from megabonk_bot.runtime.objective_cache import ObjectiveUiCache
from megabonk_bot.runtime.input_controller import (
    apply_cam_yaw as _runtime_apply_cam_yaw,
    hold as _runtime_hold,
    key_off as _runtime_key_off,
    key_on as _runtime_key_on,
    release_all_keys as _runtime_release_all_keys,
    set_move as _runtime_set_move,
    tap as _runtime_tap,
)
from megabonk_bot.runtime.loop import RateLimiter, maybe_warn_capture_error
from megabonk_bot.runtime.overlay import (
    draw_runtime_overlay as _runtime_draw_runtime_overlay,
    handle_overlay_mouse_event as _runtime_handle_overlay_mouse_event,
)
from megabonk_bot.runtime.overlay_window import (
    ensure_overlay_window as _runtime_ensure_overlay_window,
    resize_overlay_window as _runtime_resize_overlay_window,
    set_overlay_borderless as _runtime_set_overlay_borderless,
    set_overlay_colorkey_transparent as _runtime_set_overlay_colorkey_transparent,
    set_overlay_topmost as _runtime_set_overlay_topmost,
    sync_overlay_to_game_window as _runtime_sync_overlay_to_game_window,
)
from megabonk_bot.runtime.perf_budget import (
    build_performance_sample,
    format_over_budget,
    normalize_performance_budget_ms,
)
from megabonk_bot.runtime.recovery_clicks import (
    make_window_click as _runtime_make_window_click,
    try_click_template as _runtime_try_click_template,
)
from megabonk_bot.runtime.recovery import RecoveryState, decide_recovery_action
from megabonk_bot.runtime.screen_state import RuntimeScreenDetector, is_death_like_frame
from megabonk_bot.runtime.upgrade_dialog import is_upgrade_dialog as _runtime_is_upgrade_dialog
from megabonk_bot.ui_ocr import UiTextDetection

cv2 = None
di = None

LOGGER = logging.getLogger(__name__)


class NullObjectiveUiCache:
    def start(self) -> None:
        return None

    def submit(self, _frame) -> None:
        return None

    def snapshot(self) -> UiTextDetection:
        return UiTextDetection()

    def close(self) -> None:
        return None


class NullHudTelemetryCache:
    def start(self) -> None:
        return None

    def submit(self, _frame) -> None:
        return None

    def set_regions(self, _regions) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "debug_dumped": False,
            "hud_fail_streak": 0,
            "hud_ts": None,
        }

    def close(self) -> None:
        return None


def _resolve_allow_map_scan(
    *,
    survival_only: bool,
    allow_map_scan_tab: bool,
    max_enabled: bool,
    explore_with_tab: bool,
    current_scene_id: str,
) -> bool:
    if survival_only:
        return False
    if str(current_scene_id).lower() == "finalbossmap":
        return False
    return bool(allow_map_scan_tab or (max_enabled and explore_with_tab))


def _should_auto_pick_upgrade(*, survival_only: bool, auto_pick_upgrade_with_space: bool) -> bool:
    if survival_only:
        return False
    return bool(auto_pick_upgrade_with_space)


def _resolve_runtime_detection_flags(
    *,
    survival_only: bool,
    minimap_enabled: bool,
    projectiles_enabled: bool,
    world_objects_enabled: bool,
    memory_probe_enabled: bool,
) -> dict[str, bool]:
    if survival_only:
        return {
            "minimap_enabled": False,
            "projectiles_enabled": False,
            "world_objects_enabled": False,
            "memory_probe_enabled": False,
        }
    return {
        "minimap_enabled": bool(minimap_enabled),
        "projectiles_enabled": bool(projectiles_enabled),
        "world_objects_enabled": bool(world_objects_enabled),
        "memory_probe_enabled": bool(memory_probe_enabled),
    }


def _resolve_runtime_movement_flags(
    *,
    survival_only: bool,
    bunny_hop_enabled: bool,
    sliding_enabled: bool,
) -> dict[str, bool]:
    if survival_only:
        return {
            "bunny_hop_enabled": False,
            "sliding_enabled": False,
        }
    return {
        "bunny_hop_enabled": bool(bunny_hop_enabled),
        "sliding_enabled": bool(sliding_enabled),
    }


def key_on(key: str) -> None:
    _runtime_key_on(di, key)


def key_off(key: str) -> None:
    _runtime_key_off(di, key)


def tap(key: str, dt: float = 0.01) -> None:
    _runtime_tap(di, key, dt=dt)


def hold(key: str, dt: float = 0.5) -> None:
    _runtime_hold(di, key, dt=dt)


def set_move(dir_id: int) -> None:
    _runtime_set_move(di, dir_id)


def release_all_keys() -> None:
    _runtime_release_all_keys(di)


def apply_cam_yaw(yaw_id: int, cam_yaw_pixels: int) -> None:
    _runtime_apply_cam_yaw(di, yaw_id, cam_yaw_pixels)


class WinHotkeyPoller:
    def __init__(self, *, enabled: bool, toggle_vk: int, panic_vk: int):
        self.enabled = bool(enabled) and hasattr(ctypes, "windll")
        self.toggle_vk = int(toggle_vk)
        self.panic_vk = int(panic_vk)
        self._prev: dict[int, bool] = {}
        self._user32 = getattr(ctypes.windll, "user32", None) if self.enabled else None
        if self._user32 is None:
            self.enabled = False

    def _key_state(self, vk: int) -> tuple[bool, bool]:
        if not self.enabled or self._user32 is None:
            return False, False
        state = int(self._user32.GetAsyncKeyState(vk))
        # High bit: currently held. Low bit: pressed since the previous query.
        return bool(state & 0x8000), bool(state & 0x0001)

    def _edge_down(self, vk: int) -> bool:
        now, pressed_since_last_query = self._key_state(vk)
        prev = self._prev.get(vk, False)
        self._prev[vk] = now
        return pressed_since_last_query or (now and not prev)

    def poll(self) -> tuple[bool, bool]:
        if not self.enabled:
            return False, False
        toggle = self._edge_down(self.toggle_vk)
        panic = self._edge_down(self.panic_vk)
        return toggle, panic

def build_runtime_event(
    *,
    ts: float,
    mode,
    frame_id: int,
    screen: str,
    snapshot,
    hud_values: dict[str, Any],
    action,
    action_reason: str,
    restart_event: Optional[str],
    safe_sector: str,
    boss_prep: bool,
    boss_name: Optional[str],
    preferred_direction: str,
    threats,
    loop_start: float,
    step_hz: int,
    dt: float,
    window_title: str,
    frame_width: int,
    frame_height: int,
    capture_bad_grab_count: int = 0,
    capture_last_error: Optional[str] = None,
    hud_debug_dumped: bool = False,
    hud_fail_streak: int = 0,
    navigation_context=None,
    performance: dict[str, Any] | None = None,
    schema_version: str = RUNTIME_EVENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    return _build_runtime_event(
        ts=ts,
        mode=mode,
        frame_id=frame_id,
        screen=screen,
        snapshot=snapshot,
        hud_values=hud_values,
        action=action,
        action_reason=action_reason,
        restart_event=restart_event,
        safe_sector=safe_sector,
        boss_prep=boss_prep,
        boss_name=boss_name,
        preferred_direction=preferred_direction,
        threats=threats,
        loop_start=loop_start,
        step_hz=step_hz,
        dt=dt,
        window_title=window_title,
        frame_width=frame_width,
        frame_height=frame_height,
        capture_bad_grab_count=capture_bad_grab_count,
        capture_last_error=capture_last_error,
        hud_debug_dumped=hud_debug_dumped,
        hud_fail_streak=hud_fail_streak,
        navigation_context=navigation_context,
        performance=performance,
        schema_version=schema_version,
    )


def _draw_runtime_overlay(
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
    return _runtime_draw_runtime_overlay(
        cv2,
        frame,
        analysis,
        snapshot,
        mode=mode,
        action_reason=action_reason,
        hud_values=hud_values,
        hud_regions=hud_regions,
        navigation_context=navigation_context,
        transparent_canvas=transparent_canvas,
    )


def _parse_boss_schedule(raw_items: list[dict[str, Any]], boss_window_cls) -> list[Any]:
    windows = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        spawn_s = int(item.get("spawn_s", -1))
        if spawn_s < 0:
            continue
        windows.append(
            boss_window_cls(
                name=str(item.get("name", f"boss_{spawn_s}")),
                spawn_s=spawn_s,
                prep_s=int(item.get("prep_s", 10)),
            )
        )
    return windows


def _prepare_heuristic_config(
    detect_cfg: dict[str, Any],
    autopilot_cfg: dict[str, Any],
) -> dict[str, Any]:
    heuristic_cfg = dict(autopilot_cfg.get("heuristic", {}))
    heuristic_cfg.setdefault("enemy_hsv_lower", tuple(detect_cfg["enemy_hsv_lower"]))
    heuristic_cfg.setdefault("enemy_hsv_upper", tuple(detect_cfg["enemy_hsv_upper"]))
    heuristic_cfg.setdefault("enemy_area_threshold", float(detect_cfg["enemy_min_area"]))
    return heuristic_cfg


def _prepare_navigation_config(
    navigation_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "profile": str(navigation_cfg.get("profile", "cautious")),
        "lane_count": int(navigation_cfg.get("lane_count", 5)),
        "drop_risk_threshold": float(navigation_cfg.get("drop_risk_threshold", 0.58)),
        "downhill_z_threshold": float(navigation_cfg.get("downhill_z_threshold", 0.18)),
        "memory_required_for_slide": bool(navigation_cfg.get("memory_required_for_slide", True)),
    }


def _build_threat_candidates(snapshot, tracked_entity_cls):
    threat_candidates = list(snapshot.enemy_classes)
    if not threat_candidates:
        threat_candidates = [
            tracked_entity_cls(
                label=item.label,
                rect=item.rect,
                score=item.score,
                source="screen_cv",
                entity_id=item.label,
                threat_tier=1.0,
            )
            for item in snapshot.enemies
        ]
    threat_candidates.extend(snapshot.projectile_classes)
    if not snapshot.projectile_classes:
        threat_candidates.extend(
            tracked_entity_cls(
                label=item.label,
                rect=item.rect,
                score=item.score,
                source="screen_cv",
                entity_id=item.label,
                threat_tier=1.6,
                family="projectile",
            )
            for item in snapshot.projectiles
        )
    threat_candidates.extend(
        tracked_entity_cls(
            label=item.label,
            rect=item.rect,
            score=item.score,
            source=item.source,
            entity_id=item.entity_id,
            threat_tier=max(
                1.7,
                float(item.metadata.get("threat_tier", 1.7))
                if isinstance(item.metadata, dict)
                else 1.7,
            ),
            family=item.family,
            variant=item.variant,
            metadata=dict(item.metadata),
        )
        for item in snapshot.hazards
    )
    return threat_candidates


def _resolve_optional_path(raw_path: str | None, *, base_dir: Path) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_project_base_dir(config_path: Path | None) -> Path:
    if config_path is None:
        return Path.cwd().resolve()
    resolved_config_path = config_path.resolve()
    for candidate in resolved_config_path.parents:
        config_dir = candidate / "config"
        if not config_dir.is_dir():
            continue
        try:
            resolved_config_path.relative_to(config_dir)
        except ValueError:
            continue
        return candidate.resolve()
    return resolved_config_path.parent.resolve()


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _build_world_probe(
    *,
    survival_only: bool,
    enabled: bool,
    window_title: str,
    poll_interval_s: float,
    signatures_path: Path | None,
) -> WorldStateProbe:
    if survival_only or not enabled:
        return NullProbe()
    signatures = _load_optional_json(signatures_path)
    if not signatures:
        return NullProbe()
    return ExternalProcessProbe(
        window_title=window_title,
        poll_interval_s=poll_interval_s,
        signatures=signatures,
    )


def _build_objective_cache(
    *,
    survival_only: bool,
    read_objective_ui,
    lexicon,
    interval_s: float,
):
    if survival_only:
        return NullObjectiveUiCache()
    return ObjectiveUiCache(
        read_objective_ui=read_objective_ui,
        lexicon=lexicon,
        interval_s=interval_s,
    )


def _build_hud_cache(
    *,
    survival_only: bool,
    read_hud_telemetry,
    regions,
    interval_s: float,
):
    if survival_only:
        return NullHudTelemetryCache()
    return HudTelemetryCache(
        read_hud_telemetry=read_hud_telemetry,
        regions=regions,
        interval_s=interval_s,
    )


def _build_screen_detector(
    *,
    templates,
    regions,
    autopilot_cfg: dict[str, Any],
):
    return RuntimeScreenDetector(
        templates=templates,
        regions=regions,
        template_thresholds=dict(autopilot_cfg.get("template_thresholds", {})),
    )


def run(args) -> None:
    global cv2
    global di

    import cv2 as _cv2
    import pydirectinput as _di

    from megabonk_bot.hud import read_hud_telemetry
    from megabonk_bot.max_model import (
        BossWindow,
        ObjectDetection,
        OnnxObjectDetector,
        SceneMemory360,
        build_occupancy_cost_map,
        pick_low_cost_direction,
        score_enemy_threats,
        should_enter_boss_prep,
    )
    from megabonk_bot.recognition import analyze_scene
    from megabonk_bot.regions import build_regions
    from megabonk_bot.runtime_logic import BotMode, build_scene_snapshot
    from megabonk_bot.runtime_state import RuntimeStateMachine
    from megabonk_bot.templates import load_templates
    from megabonk_bot.ui_ocr import read_objective_ui
    from megabonk_bot.world_state import TrackedEntity
    from window_capture import WindowCapture

    cv2 = _cv2
    di = _di
    di.PAUSE = 0.0
    di.FAILSAFE = False

    config_path = Path(args.config).resolve() if args.config else None
    resource_base_dir = _resolve_project_base_dir(config_path)
    config = load_config(config_path)
    runtime_cfg = config["runtime"]
    detect_cfg = config["detection"]
    mvp_cfg = config["mvp_policy"]
    max_cfg = config["max_policy"]
    navigation_cfg = config["navigation"]
    hotkey_cfg = config["hotkeys"]
    autopilot_cfg = config["autopilot"]
    survival_only = bool(mvp_cfg.get("survival_only", False))
    runtime_detection_flags = _resolve_runtime_detection_flags(
        survival_only=survival_only,
        minimap_enabled=bool(detect_cfg.get("minimap_enabled", False)),
        projectiles_enabled=bool(detect_cfg.get("projectiles_enabled", False)),
        world_objects_enabled=bool(detect_cfg.get("world_objects_enabled", False)),
        memory_probe_enabled=bool(detect_cfg.get("memory_probe_enabled", True)),
    )
    runtime_movement_flags = _resolve_runtime_movement_flags(
        survival_only=survival_only,
        bunny_hop_enabled=bool(max_cfg.get("bunny_hop_enabled", True)),
        sliding_enabled=bool(max_cfg.get("sliding_enabled", True)),
    )

    step_hz = max(1, int(runtime_cfg["step_hz"]))
    dt = 1.0 / step_hz
    cam_yaw_pixels = int(runtime_cfg["cam_yaw_pixels"])
    log_interval_s = float(runtime_cfg["event_log_interval_s"])
    upgrade_space_cooldown_s = float(runtime_cfg["upgrade_space_cooldown_s"])
    restart_cooldown_s = float(runtime_cfg["restart_cooldown_s"])
    restart_hold_s = float(runtime_cfg["restart_hold_s"])
    restart_wait_timeout_s = float(runtime_cfg["restart_wait_timeout_s"])
    restart_max_attempts = max(1, int(runtime_cfg["restart_max_attempts"]))
    capture_log_errors = bool(runtime_cfg.get("capture_log_errors", True))
    hud_ocr_every_s = float(runtime_cfg.get("hud_ocr_every_s", 0.8))
    objective_ocr_every_s = float(runtime_cfg.get("objective_ocr_every_s", 1.2))
    performance_budget_enabled = bool(runtime_cfg.get("performance_budget_enabled", True))
    performance_budget_warn_interval_s = float(
        runtime_cfg.get("performance_budget_warn_interval_s", 5.0)
    )
    performance_budget_ms = normalize_performance_budget_ms(
        runtime_cfg.get("performance_budget_ms"),
        tick_budget_ms=dt * 1000.0,
    )
    event_schema_version = str(
        runtime_cfg.get("event_schema_version", RUNTIME_EVENT_SCHEMA_VERSION)
    )
    overlay_enabled = bool(runtime_cfg["overlay_enabled"]) and not args.no_overlay
    overlay_window = str(runtime_cfg["overlay_window"])
    overlay_topmost_enabled = bool(runtime_cfg.get("overlay_topmost", True))
    overlay_transparent = bool(runtime_cfg.get("overlay_transparent", True))
    overlay_redraw_interval_ticks = max(1, int(runtime_cfg.get("overlay_redraw_interval_ticks", 1)))
    window_title = str(args.window or runtime_cfg["window_title"])
    templates_dir = str(
        _resolve_optional_path(
            str(args.templates_dir or runtime_cfg["templates_dir"]),
            base_dir=resource_base_dir,
        )
    )
    capture_backend = str(args.capture_backend or runtime_cfg.get("capture_backend", "auto"))
    if args.window_focus_interval_s is None:
        window_focus_interval_s = float(runtime_cfg.get("window_focus_interval_s", 0.25))
    else:
        window_focus_interval_s = float(args.window_focus_interval_s)

    cap = WindowCapture.create(window_title, capture_backend=capture_backend)
    cap.focus(topmost=True)
    window_click = _runtime_make_window_click(
        cap,
        input_driver=di,
        focus_interval_s=window_focus_interval_s,
    )
    bbox = cap.get_bbox()
    if overlay_enabled:
        _runtime_ensure_overlay_window(
            cv2,
            overlay_window,
            width=int(bbox.get("width", 1)),
            height=int(bbox.get("height", 1)),
        )
    regions = build_regions(bbox["width"], bbox["height"])
    templates = load_templates(templates_dir)
    asset_refs_dir = _resolve_optional_path(
        str(detect_cfg.get("asset_refs_dir", "")),
        base_dir=resource_base_dir,
    )
    enemy_catalog_path = _resolve_optional_path(
        str(detect_cfg.get("enemy_catalog_path", "")),
        base_dir=resource_base_dir,
    )
    world_catalog_path = _resolve_optional_path(
        str(detect_cfg.get("world_catalog_path", "")),
        base_dir=resource_base_dir,
    )
    projectile_catalog_path = _resolve_optional_path(
        str(detect_cfg.get("projectile_catalog_path", "")),
        base_dir=resource_base_dir,
    )
    ocr_lexicon_path = _resolve_optional_path(
        str(detect_cfg.get("ocr_lexicon_path", "")),
        base_dir=resource_base_dir,
    )
    memory_signatures_path = _resolve_optional_path(
        str(detect_cfg.get("memory_signatures_path", "")),
        base_dir=resource_base_dir,
    )
    catalogs = load_curated_catalogs(
        asset_refs_dir=asset_refs_dir,
        enemy_catalog_path=enemy_catalog_path,
        world_catalog_path=world_catalog_path,
        projectile_catalog_path=projectile_catalog_path,
        ocr_lexicon_path=ocr_lexicon_path,
    )
    world_probe = _build_world_probe(
        survival_only=survival_only,
        enabled=runtime_detection_flags["memory_probe_enabled"],
        window_title=window_title,
        poll_interval_s=float(detect_cfg.get("memory_poll_interval_s", 0.25)),
        signatures_path=memory_signatures_path,
    )
    screen_detector = _build_screen_detector(
        templates=templates,
        regions=regions,
        autopilot_cfg=autopilot_cfg,
    )
    heuristic_cfg = _prepare_heuristic_config(detect_cfg, autopilot_cfg)
    navigation_planner = StatefulNavigationPlanner(
        config=_prepare_navigation_config(navigation_cfg),
        allow_bunny_hop=runtime_movement_flags["bunny_hop_enabled"],
        sliding_enabled=runtime_movement_flags["sliding_enabled"],
        jump_cooldown=int(heuristic_cfg.get("jump_cooldown", 30)),
        slide_cooldown=int(heuristic_cfg.get("slide_cooldown", 24)),
        stuck_diff_threshold=float(heuristic_cfg.get("stuck_diff_threshold", 3.0)),
        stuck_frames_required=int(heuristic_cfg.get("stuck_frames_required", 6)),
        stuck_escape_ticks=int(heuristic_cfg.get("stuck_escape_ticks", 16)),
    )
    hotkeys = WinHotkeyPoller(
        enabled=bool(hotkey_cfg["enabled"]) and not args.no_hotkeys,
        toggle_vk=int(hotkey_cfg["toggle_vk"]),
        panic_vk=int(hotkey_cfg["panic_vk"]),
    )
    logger = JsonlEventLogger(Path(runtime_cfg["event_log_path"]))
    hud_cache = _build_hud_cache(
        survival_only=survival_only,
        read_hud_telemetry=read_hud_telemetry,
        regions=regions,
        interval_s=hud_ocr_every_s,
    )
    hud_cache.start()
    objective_cache = _build_objective_cache(
        survival_only=survival_only,
        read_objective_ui=read_objective_ui,
        lexicon=catalogs.ocr_lexicon if catalogs else None,
        interval_s=objective_ocr_every_s,
    )
    objective_cache.start()

    max_enabled = bool(max_cfg.get("enabled", False))
    onnx_detector = None
    if max_enabled and bool(detect_cfg.get("use_onnx")) and detect_cfg.get("onnx_model_path"):
        onnx_detector = OnnxObjectDetector(str(detect_cfg["onnx_model_path"]))
    scene_memory = SceneMemory360(ttl_s=float(detect_cfg.get("scene_memory_ttl_s", 2.0)))
    boss_schedule = []
    if not survival_only:
        boss_schedule = _parse_boss_schedule(config.get("boss_schedule", []), BossWindow)

    try:
        mode = BotMode(str(runtime_cfg.get("state", "OFF")).upper())
    except ValueError:
        LOGGER.warning("Invalid runtime state %r, falling back to OFF", runtime_cfg.get("state"))
        mode = BotMode.OFF
    state_machine = RuntimeStateMachine(mode=mode)

    frame_id = 0
    last_event_log_ts = 0.0
    last_upgrade_space_ts = 0.0
    recovery_state = RecoveryState()
    map_scan_tick = 0
    run_started_ts = time.time()
    capture_warn_limiter = RateLimiter(interval_s=5.0)
    performance_warn_limiter = RateLimiter(interval_s=performance_budget_warn_interval_s)
    overlay_topmost_applied = False
    overlay_transparent_applied = False
    overlay_borderless_applied = False
    overlay_mouse_callback_set = False
    overlay_mouse_callback_failed = False
    overlay_bbox_error_logged = False
    overlay_controls = {"toggle": False, "panic": False, "rects": {}}
    pending_toggle = False
    pending_panic = False
    last_scene_signature: tuple[str | None, str | None, bool | None] | None = None
    last_overlay_frame = None

    def _on_overlay_mouse(event, x, y, _flags, state):
        _runtime_handle_overlay_mouse_event(cv2, event, x, y, state)

    try:
        while True:
            loop_start = time.perf_counter()
            performance_stages_ms: dict[str, float] = {}
            should_quit = False
            hotkey_toggle, hotkey_panic = hotkeys.poll()
            toggle = bool(hotkey_toggle or pending_toggle or overlay_controls.get("toggle"))
            panic = bool(hotkey_panic or pending_panic or overlay_controls.get("panic"))
            pending_toggle = False
            pending_panic = False
            overlay_controls["toggle"] = False
            overlay_controls["panic"] = False
            if panic:
                mode = state_machine.on_events(panic=True)
                release_all_keys()
                navigation_planner.reset()
            if toggle:
                mode = state_machine.on_events(toggle=True)
                if mode == BotMode.ACTIVE:
                    recovery_state.attempts = 0
                    recovery_state.started_ts = 0.0
                    navigation_planner.reset()
                else:
                    release_all_keys()
                    navigation_planner.reset()

            stage_start = time.perf_counter()
            frame = cap.grab()
            capture_diag = cap.get_capture_diagnostics()
            capture_bad_grab_count = int(capture_diag.get("bad_grab_count", 0))
            capture_last_error = capture_diag.get("last_error")
            performance_stages_ms["capture"] = (
                time.perf_counter() - stage_start
            ) * 1000.0
            maybe_warn_capture_error(
                capture_last_error=capture_last_error,
                capture_bad_grab_count=capture_bad_grab_count,
                limiter=capture_warn_limiter,
                enabled=capture_log_errors,
            )
            if frame is None or frame.size == 0:
                time.sleep(0.05)
                continue
            h, w = frame.shape[:2]
            frame_id += 1

            if bbox["width"] != w or bbox["height"] != h:
                bbox["width"] = w
                bbox["height"] = h
                if overlay_enabled:
                    _runtime_resize_overlay_window(
                        cv2,
                        overlay_window,
                        width=int(w),
                        height=int(h),
                    )
                    last_overlay_frame = None
                regions = build_regions(w, h)
                hud_cache.set_regions(regions)
                screen_detector = _build_screen_detector(
                    templates=templates,
                    regions=regions,
                    autopilot_cfg=autopilot_cfg,
                )

            screen = screen_detector.detect(frame)
            is_dead = screen == "DEAD" or (screen != "RUNNING" and is_death_like_frame(frame))
            is_upgrade = _runtime_is_upgrade_dialog(frame, templates, regions)
            stage_start = time.perf_counter()
            hud_cache.submit(frame)
            objective_cache.submit(frame)
            hud_values = hud_cache.snapshot()
            objective_ui = objective_cache.snapshot()
            performance_stages_ms["hud"] = (
                time.perf_counter() - stage_start
            ) * 1000.0
            probe_result = world_probe.sample(now_ts=time.time())

            stage_start = time.perf_counter()
            analysis = analyze_scene(
                frame,
                templates=templates,
                catalogs=catalogs,
                regions=regions,
                probe_result=probe_result,
                objective_ui=objective_ui,
                grid_rows=int(detect_cfg["grid_rows"]),
                grid_cols=int(detect_cfg["grid_cols"]),
                enemy_hsv_lower=tuple(detect_cfg["enemy_hsv_lower"]),
                enemy_hsv_upper=tuple(detect_cfg["enemy_hsv_upper"]),
                enemy_min_area=float(detect_cfg["enemy_min_area"]),
                interact_threshold=float(detect_cfg["interact_threshold"]),
                enemy_classifier_mode=str(detect_cfg.get("enemy_classifier_mode", "hybrid")),
                minimap_enabled=runtime_detection_flags["minimap_enabled"],
                projectiles_enabled=runtime_detection_flags["projectiles_enabled"],
                world_objects_enabled=runtime_detection_flags["world_objects_enabled"],
                world_object_families=tuple(detect_cfg.get("world_object_families", [])),
                analysis_scale=float(detect_cfg.get("analysis_scale", 1.0)),
            )
            performance_stages_ms["scene_analysis"] = (
                time.perf_counter() - stage_start
            ) * 1000.0

            analysis.setdefault("projectiles", [])
            onnx_detections: list[ObjectDetection] = []
            projectiles_enabled = runtime_detection_flags["projectiles_enabled"]
            if onnx_detector is not None and onnx_detector.enabled:
                onnx_detections = onnx_detector.detect(frame)
                analysis.setdefault("enemy_classes", [])
                for det in onnx_detections:
                    lowered = str(det.label).lower()
                    if "projectile" in lowered:
                        if projectiles_enabled:
                            analysis["projectiles"].append(det)
                    else:
                        analysis["enemy_classes"].append(
                            TrackedEntity(
                                label=det.label,
                                rect=det.rect,
                                score=det.score,
                                source="onnx",
                                entity_id=det.label,
                                threat_tier=2.0 if "boss" in lowered else 1.5,
                            )
                        )
            scene_memory_items = [
                ObjectDetection(label=item.label, rect=item.rect, score=float(item.score))
                for item in analysis.get("enemy_classes", [])
            ]
            scene_memory_items.extend(
                ObjectDetection(label=item.label, rect=item.rect, score=float(item.score))
                for item in analysis.get("projectile_classes", [])
            )
            scene_memory_items.extend(
                ObjectDetection(label=item.label, rect=item.rect, score=float(item.score))
                for item in analysis.get("world_objects", [])
            )
            scene_memory_items.extend(
                ObjectDetection(label=item.label, rect=item.rect, score=float(item.score))
                for item in analysis.get("projectiles", [])
            )
            scene_memory_items.extend(
                ObjectDetection(label=item.label, rect=item.rect, score=float(item.score))
                for item in analysis.get("hazards", [])
            )
            scene_memory.update(scene_memory_items)

            snapshot = build_scene_snapshot(
                frame_id=frame_id,
                ts=time.time(),
                frame_width=w,
                frame_height=h,
                analysis=analysis,
                hud_values=hud_values,
                is_dead=is_dead,
                is_upgrade=is_upgrade,
            )

            map_state = getattr(snapshot, "map_state", None)
            scene_signature = (
                getattr(map_state, "scene_id", None) if map_state is not None else None,
                getattr(map_state, "active_room_id", None) if map_state is not None else None,
                getattr(map_state, "is_crypt", None) if map_state is not None else None,
            )
            if last_scene_signature is None:
                last_scene_signature = scene_signature
            elif scene_signature != last_scene_signature:
                scene_memory.clear()
                last_scene_signature = scene_signature

            if mode == BotMode.ACTIVE and snapshot.is_dead:
                mode = state_machine.on_events(dead_detected=True)
                recovery_state.started_ts = time.time()
                recovery_state.attempts = 0

            allow_tab_scan = _resolve_allow_map_scan(
                survival_only=survival_only,
                allow_map_scan_tab=bool(mvp_cfg.get("allow_map_scan_tab", False)),
                max_enabled=max_enabled,
                explore_with_tab=bool(max_cfg.get("explore_with_tab", False)),
                current_scene_id=str(getattr(map_state, "scene_id", "") or ""),
            )
            map_scan_now = False
            if allow_tab_scan:
                map_scan_tick += 1
                if map_scan_tick >= int(mvp_cfg.get("map_scan_interval_ticks", 180)):
                    map_scan_now = True
                    map_scan_tick = 0

            threat_candidates = _build_threat_candidates(snapshot, TrackedEntity)
            threats = score_enemy_threats(
                threat_candidates,
                frame_w=w,
                frame_h=h,
            )
            occupancy = build_occupancy_cost_map(
                frame_w=w,
                frame_h=h,
                obstacles=[
                    ObjectDetection(label=item.label, rect=item.rect, score=item.score)
                    for item in snapshot.obstacles
                ],
                rows=int(detect_cfg["grid_rows"]),
                cols=int(detect_cfg["grid_cols"]),
            )
            preferred_direction = pick_low_cost_direction(occupancy)
            elapsed = int(time.time() - run_started_ts)
            boss_prep = False
            boss_name = None
            if not survival_only:
                boss_prep, boss_name = should_enter_boss_prep(elapsed, boss_schedule)
            action, navigation_context = navigation_planner.evaluate(
                frame,
                snapshot,
                mode=mode,
                threats=threats,
                allow_map_scan=allow_tab_scan,
                map_scan_now=map_scan_now,
            )

            if mode in {BotMode.ACTIVE, BotMode.RECOVERY}:
                cap.focus_if_needed(
                    topmost=False,
                    min_interval_s=window_focus_interval_s,
                )

            restart_event = None
            if mode == BotMode.RECOVERY:
                set_move(0)
                now = time.time()
                decision = decide_recovery_action(
                    now=now,
                    screen=screen,
                    is_dead=snapshot.is_dead,
                    press_restart=bool(action.press_r),
                    state=recovery_state,
                    restart_cooldown_s=restart_cooldown_s,
                    restart_max_attempts=restart_max_attempts,
                    restart_wait_timeout_s=restart_wait_timeout_s,
                )
                restart_event = decision.restart_event
                if decision.running_restored:
                    mode = state_machine.on_events(running_restored=True)
                else:
                    if decision.hold_restart:
                        release_all_keys()
                        hold("r", dt=restart_hold_s)
                    if decision.try_fallback_click:
                        clicked = (
                            _runtime_try_click_template(
                                frame,
                                templates,
                                regions,
                                "tpl_confirm",
                                "REG_DEAD_CONFIRM",
                                0.6,
                                click_fn=window_click,
                            )
                            or _runtime_try_click_template(
                                frame,
                                templates,
                                regions,
                                "tpl_play",
                                "REG_MAIN_PLAY",
                                0.65,
                                click_fn=window_click,
                            )
                        )
                        if clicked:
                            restart_event = "fallback_click_menu"
                            recovery_state.started_ts = now

            elif mode == BotMode.ACTIVE:
                apply_cam_yaw(action.yaw, cam_yaw_pixels=cam_yaw_pixels)
                set_move(action.dir_id)
                if int(action.jump) == 1:
                    tap("space", dt=0.005)
                if int(action.slide) == 1:
                    tap("shift", dt=0.005)
                if action.press_tab:
                    tap("tab", dt=0.01)
                if _should_auto_pick_upgrade(
                    survival_only=survival_only,
                    auto_pick_upgrade_with_space=bool(
                        mvp_cfg.get("auto_pick_upgrade_with_space", True)
                    ),
                ) and action.press_space:
                    now = time.time()
                    if (now - last_upgrade_space_ts) >= upgrade_space_cooldown_s:
                        tap("space", dt=0.01)
                        last_upgrade_space_ts = now
            else:
                set_move(0)

            if mode == BotMode.PANIC:
                set_move(0)
                release_all_keys()

            stage_start = time.perf_counter()
            if overlay_enabled:
                redraw_overlay = (
                    last_overlay_frame is None
                    or (frame_id % overlay_redraw_interval_ticks) == 0
                )
                if redraw_overlay:
                    overlay, button_rects = _draw_runtime_overlay(
                        frame,
                        analysis,
                        snapshot,
                        mode=mode,
                        action_reason=action.reason,
                        hud_values=hud_values,
                        hud_regions=regions,
                        navigation_context=navigation_context,
                        transparent_canvas=overlay_transparent,
                    )
                    last_overlay_frame = overlay
                    overlay_controls["rects"] = button_rects
                    cv2.imshow(overlay_window, overlay)
                    if not overlay_mouse_callback_set and not overlay_mouse_callback_failed:
                        try:
                            cv2.setMouseCallback(overlay_window, _on_overlay_mouse, overlay_controls)
                            overlay_mouse_callback_set = True
                        except Exception:
                            LOGGER.warning("Failed to bind overlay mouse callback", exc_info=True)
                            overlay_mouse_callback_failed = True
                            overlay_mouse_callback_set = False
                    if overlay_transparent:
                        try:
                            game_bbox = cap.get_bbox()
                        except Exception:
                            if not overlay_bbox_error_logged:
                                LOGGER.warning(
                                    "Failed to read game window bbox for overlay sync; using capture bbox fallback",
                                    exc_info=True,
                                )
                                overlay_bbox_error_logged = True
                            game_bbox = bbox
                        _runtime_resize_overlay_window(
                            cv2,
                            overlay_window,
                            width=int(game_bbox.get("width", w)),
                            height=int(game_bbox.get("height", h)),
                        )
                        _runtime_sync_overlay_to_game_window(
                            overlay_window,
                            game_bbox,
                            topmost=overlay_topmost_enabled,
                        )
                        if not overlay_borderless_applied:
                            overlay_borderless_applied = _runtime_set_overlay_borderless(
                                overlay_window
                            )
                        if not overlay_transparent_applied:
                            overlay_transparent_applied = (
                                _runtime_set_overlay_colorkey_transparent(
                                    overlay_window,
                                    colorkey=(0, 0, 0),
                                )
                            )
                        overlay_topmost_applied = overlay_topmost_enabled
                    elif not overlay_topmost_applied and overlay_topmost_enabled:
                        overlay_topmost_applied = _runtime_set_overlay_topmost(overlay_window)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    should_quit = True
                if key in (ord("s"), ord("S"), ord(" ")):
                    pending_toggle = True
                elif key in (ord("p"), ord("P")):
                    pending_panic = True
            performance_stages_ms["overlay"] = (
                time.perf_counter() - stage_start
            ) * 1000.0

            performance_sample = None
            if performance_budget_enabled:
                tick_ms = (time.perf_counter() - loop_start) * 1000.0
                performance_sample = build_performance_sample(
                    performance_stages_ms,
                    budget_ms=performance_budget_ms,
                    tick_ms=tick_ms,
                )
                if performance_sample["over_budget_ms"] and performance_warn_limiter.allow():
                    LOGGER.warning(
                        "Runtime performance budget exceeded: %s",
                        format_over_budget(performance_sample),
                    )

            now = time.time()
            if now - last_event_log_ts >= log_interval_s:
                last_event_log_ts = now
                event = build_runtime_event(
                    ts=now,
                    mode=mode,
                    frame_id=frame_id,
                    screen=screen,
                    snapshot=snapshot,
                    hud_values=hud_values,
                    action=action,
                    action_reason=action.reason,
                    restart_event=restart_event,
                    safe_sector=snapshot.safe_sector,
                    boss_prep=boss_prep,
                    boss_name=boss_name,
                    preferred_direction=preferred_direction,
                    threats=threats,
                    loop_start=loop_start,
                    step_hz=step_hz,
                    dt=dt,
                    window_title=window_title,
                    frame_width=w,
                    frame_height=h,
                    capture_bad_grab_count=capture_bad_grab_count,
                    capture_last_error=(
                        str(capture_last_error)
                        if capture_last_error is not None
                        else None
                    ),
                    hud_debug_dumped=bool(hud_values.get("debug_dumped", False)),
                    hud_fail_streak=int(hud_values.get("hud_fail_streak", 0) or 0),
                    navigation_context=navigation_context,
                    performance=performance_sample,
                    schema_version=event_schema_version,
                )
                logger.log(event)

            if should_quit:
                break

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        release_all_keys()
        objective_cache.close()
        hud_cache.close()
        try:
            world_probe.close()
        except Exception:
            LOGGER.warning("Failed to close world probe cleanly", exc_info=True)
        logger.close()
        if overlay_enabled:
            cv2.destroyAllWindows()


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Runtime bot (MVP): F8 toggle, F12 panic, JSONL + overlay.",
    )
    parser.add_argument("--window", default=None, help="Window title substring")
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "printwindow", "mss"),
        default=None,
        help="Capture backend for the game window",
    )
    parser.add_argument(
        "--window-focus-interval-s",
        type=float,
        default=None,
        help="How often to enforce focus on the target window",
    )
    parser.add_argument("--config", default="config/survival_mvp.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--templates-dir", default=None, help="Templates directory")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay window")
    parser.add_argument("--no-hotkeys", action="store_true", help="Disable WinAPI hotkeys")
    parser.add_argument(
        "--print-default-config",
        action="store_true",
        help="Print default config in YAML and exit",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.print_default_config:
        if DEFAULT_CONFIG is None:
            raise RuntimeError("DEFAULT_CONFIG is not initialized")
        print(dump_default_config_yaml(), end="")
    else:
        run(args)
