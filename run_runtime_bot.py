from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()

import argparse
import ctypes
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from megabonk_bot.config import DEFAULT_CONFIG, dump_default_config_yaml, load_config

cv2 = None
di = None
RUNTIME_EVENT_SCHEMA_VERSION = "runtime_events_v1"


def key_on(key: str) -> None:
    di.keyDown(key)


def key_off(key: str) -> None:
    di.keyUp(key)


def tap(key: str, dt: float = 0.01) -> None:
    di.keyDown(key)
    time.sleep(max(0.0, float(dt)))
    di.keyUp(key)


def hold(key: str, dt: float = 0.5) -> None:
    di.keyDown(key)
    time.sleep(max(0.0, float(dt)))
    di.keyUp(key)


def set_move(dir_id: int) -> None:
    mapping = {
        0: [],
        1: ["w"],
        2: ["s"],
        3: ["a"],
        4: ["d"],
        5: ["w", "a"],
        6: ["w", "d"],
        7: ["s", "a"],
        8: ["s", "d"],
    }
    want = set(mapping.get(int(dir_id), []))
    for key in ["w", "a", "s", "d"]:
        (key_on if key in want else key_off)(key)


def release_all_keys() -> None:
    for key in [
        "w",
        "a",
        "s",
        "d",
        "left",
        "right",
        "up",
        "down",
        "space",
        "lctrl",
        "shift",
        "tab",
    ]:
        try:
            di.keyUp(key)
        except Exception:
            pass


def apply_cam_yaw(yaw_id: int, cam_yaw_pixels: int) -> None:
    mapping = {0: -int(cam_yaw_pixels), 1: 0, 2: int(cam_yaw_pixels)}
    dx = mapping.get(int(yaw_id), 0)
    if dx != 0:
        di.moveRel(dx, 0, duration=0)


def _is_upgrade_dialog(frame_bgr, templates, regions, threshold=0.62):
    from megabonk_bot.vision import find_in_region

    if frame_bgr is None or not templates or not regions:
        return False
    region = regions.get("REG_CHEST")
    if not region:
        return False
    for name in (
        "tpl_katana",
        "tpl_dexec",
        "tpl_foliant_bottom1",
        "tpl_foliant_bottom2",
        "tpl_foliant_bottom3",
        "tpl_blood_tome",
    ):
        tpl = templates.get(name)
        if tpl is None:
            continue
        found, _, _ = find_in_region(frame_bgr, tpl, region, threshold=threshold)
        if found:
            return True
    return False


def _try_click_template(frame, templates, regions, tpl_name, region_name, threshold) -> bool:
    from megabonk_bot.vision import find_in_region

    if tpl_name not in templates or region_name not in regions:
        return False
    found, (cx, cy), score = find_in_region(
        frame,
        templates[tpl_name],
        regions[region_name],
        threshold=threshold,
    )
    if not found:
        return False
    di.moveTo(cx, cy)
    di.click()
    return score >= threshold


def _set_overlay_topmost(window_name: str) -> None:
    if not hasattr(ctypes, "windll"):
        return
    user32 = getattr(ctypes.windll, "user32", None)
    if user32 is None:
        return
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return
    HWND_TOPMOST = -1
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040
    user32.SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
    )


class WinHotkeyPoller:
    def __init__(self, *, enabled: bool, toggle_vk: int, panic_vk: int):
        self.enabled = bool(enabled) and hasattr(ctypes, "windll")
        self.toggle_vk = int(toggle_vk)
        self.panic_vk = int(panic_vk)
        self._prev: dict[int, bool] = {}
        self._user32 = getattr(ctypes.windll, "user32", None) if self.enabled else None
        if self._user32 is None:
            self.enabled = False

    def _pressed(self, vk: int) -> bool:
        if not self.enabled or self._user32 is None:
            return False
        # Старший бит: клавиша сейчас нажата.
        return bool(self._user32.GetAsyncKeyState(vk) & 0x8000)

    def _edge_down(self, vk: int) -> bool:
        now = self._pressed(vk)
        prev = self._prev.get(vk, False)
        self._prev[vk] = now
        return now and not prev

    def poll(self) -> tuple[bool, bool]:
        if not self.enabled:
            return False, False
        toggle = self._edge_down(self.toggle_vk)
        panic = self._edge_down(self.panic_vk)
        return toggle, panic


class JsonlEventLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        self._fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def _serialize_detection_list(items) -> list[dict[str, Any]]:
    output = []
    for item in items:
        if hasattr(item, "label") and hasattr(item, "rect") and hasattr(item, "score"):
            output.append(
                {
                    "label": getattr(item, "label"),
                    "rect": list(getattr(item, "rect")),
                    "score": float(getattr(item, "score")),
                }
            )
    return output


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
    schema_version: str = RUNTIME_EVENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "ts": ts,
        "mode": mode.value,
        "frame_id": int(frame_id),
        "step_hz": int(step_hz),
        "dt_ms": float(dt) * 1000.0,
        "window_title": str(window_title),
        "frame_size": {"width": int(frame_width), "height": int(frame_height)},
        "screen": screen,
        "telemetry": {
            "time": getattr(snapshot, "time_s", None),
            "hp_ratio": getattr(snapshot, "hp_ratio", None),
            "lvl": getattr(snapshot, "lvl", None),
            "kills": getattr(snapshot, "kills", None),
            "gold": hud_values.get("gold"),
        },
        "detections": {
            "enemies": _serialize_detection_list(getattr(snapshot, "enemies", [])),
            "obstacles": _serialize_detection_list(getattr(snapshot, "obstacles", [])),
            "projectiles": _serialize_detection_list(getattr(snapshot, "projectiles", [])),
            "interactables": _serialize_detection_list(getattr(snapshot, "interactables", [])),
        },
        "action": asdict(action),
        "reason": action_reason,
        "restart": restart_event,
        "safe_sector": safe_sector,
        "is_dead": bool(getattr(snapshot, "is_dead", False)),
        "is_upgrade": bool(getattr(snapshot, "is_upgrade", False)),
        "boss_prep": bool(boss_prep),
        "boss_name": boss_name,
        "preferred_direction": preferred_direction,
        "top_threats": [
            {
                "label": threat.label,
                "rect": list(threat.rect),
                "priority": threat.priority,
                "distance_norm": threat.distance_norm,
            }
            for threat in threats[:3]
        ],
        "latency_ms": (time.perf_counter() - loop_start) * 1000.0,
    }


def _draw_runtime_overlay(frame, analysis, snapshot, *, mode, action_reason: str):
    from megabonk_bot.recognition import draw_recognition_overlay

    canvas = draw_recognition_overlay(frame, analysis)
    h, _ = canvas.shape[:2]
    lines = [
        f"mode={mode.value}",
        f"reason={action_reason}",
        f"hp={snapshot.hp_ratio} lvl={snapshot.lvl} kills={snapshot.kills} time={snapshot.time_s}",
        f"enemies={len(snapshot.enemies)} obstacles={len(snapshot.obstacles)} projectiles={len(snapshot.projectiles)}",
        f"dead={snapshot.is_dead} upgrade={snapshot.is_upgrade} safe_sector={snapshot.safe_sector}",
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
    cv2.rectangle(canvas, (8, 8), (890, min(h - 8, 8 + 24 * (len(lines) + 1))), (0, 0, 0), 2)
    return canvas


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


def run(args) -> None:
    global cv2
    global di

    import cv2 as _cv2
    import pydirectinput as _di

    from autopilot import AutoPilot, HeuristicAutoPilot, is_death_like_frame
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
    from megabonk_bot.runtime_logic import BotMode, build_scene_snapshot, choose_mvp_action
    from megabonk_bot.runtime_state import RuntimeStateMachine
    from megabonk_bot.templates import load_templates
    from window_capture import WindowCapture

    cv2 = _cv2
    di = _di
    di.PAUSE = 0.0
    di.FAILSAFE = False

    config = load_config(Path(args.config).resolve() if args.config else None)
    runtime_cfg = config["runtime"]
    detect_cfg = config["detection"]
    mvp_cfg = config["mvp_policy"]
    max_cfg = config["max_policy"]
    hotkey_cfg = config["hotkeys"]
    autopilot_cfg = config["autopilot"]

    step_hz = max(1, int(runtime_cfg["step_hz"]))
    dt = 1.0 / step_hz
    cam_yaw_pixels = int(runtime_cfg["cam_yaw_pixels"])
    log_interval_s = float(runtime_cfg["event_log_interval_s"])
    upgrade_space_cooldown_s = float(runtime_cfg["upgrade_space_cooldown_s"])
    restart_cooldown_s = float(runtime_cfg["restart_cooldown_s"])
    restart_hold_s = float(runtime_cfg["restart_hold_s"])
    restart_wait_timeout_s = float(runtime_cfg["restart_wait_timeout_s"])
    restart_max_attempts = max(1, int(runtime_cfg["restart_max_attempts"]))
    overlay_enabled = bool(runtime_cfg["overlay_enabled"]) and not args.no_overlay
    overlay_window = str(runtime_cfg["overlay_window"])
    window_title = str(args.window or runtime_cfg["window_title"])
    templates_dir = str(args.templates_dir or runtime_cfg["templates_dir"])

    cap = WindowCapture.create(window_title)
    cap.focus(topmost=True)
    bbox = cap.get_bbox()
    regions = build_regions(bbox["width"], bbox["height"])
    templates = load_templates(templates_dir)
    autopilot = AutoPilot(
        templates=templates,
        regions=regions,
        click_cooldown_s=float(autopilot_cfg.get("click_cooldown_s", 0.5)),
        template_thresholds=dict(autopilot_cfg.get("template_thresholds", {})),
    )
    heuristic_cfg = _prepare_heuristic_config(detect_cfg, autopilot_cfg)
    heuristic_pilot = HeuristicAutoPilot(
        config=heuristic_cfg,
    )
    hotkeys = WinHotkeyPoller(
        enabled=bool(hotkey_cfg["enabled"]) and not args.no_hotkeys,
        toggle_vk=int(hotkey_cfg["toggle_vk"]),
        panic_vk=int(hotkey_cfg["panic_vk"]),
    )
    logger = JsonlEventLogger(Path(runtime_cfg["event_log_path"]))

    max_enabled = bool(max_cfg.get("enabled", False))
    onnx_detector = None
    if max_enabled and bool(detect_cfg.get("use_onnx")) and detect_cfg.get("onnx_model_path"):
        onnx_detector = OnnxObjectDetector(str(detect_cfg["onnx_model_path"]))
    scene_memory = SceneMemory360(ttl_s=float(detect_cfg.get("scene_memory_ttl_s", 2.0)))
    boss_schedule = _parse_boss_schedule(config.get("boss_schedule", []), BossWindow)

    try:
        mode = BotMode(str(runtime_cfg.get("state", "OFF")).upper())
    except Exception:
        mode = BotMode.OFF
    state_machine = RuntimeStateMachine(mode=mode)

    frame_id = 0
    last_event_log_ts = 0.0
    last_upgrade_space_ts = 0.0
    last_restart_ts = 0.0
    recovery_started_ts = 0.0
    recovery_attempts = 0
    map_scan_tick = 0
    run_started_ts = time.time()
    overlay_topmost_applied = False

    try:
        while True:
            loop_start = time.perf_counter()
            toggle, panic = hotkeys.poll()
            if panic:
                mode = state_machine.on_events(panic=True)
                release_all_keys()
            if toggle:
                mode = state_machine.on_events(toggle=True)
                if mode == BotMode.ACTIVE:
                    recovery_attempts = 0
                    recovery_started_ts = 0.0
                else:
                    release_all_keys()

            frame = cap.grab()
            if frame is None or frame.size == 0:
                time.sleep(0.05)
                continue
            h, w = frame.shape[:2]
            frame_id += 1

            if bbox["width"] != w or bbox["height"] != h:
                bbox["width"] = w
                bbox["height"] = h
                regions = build_regions(w, h)
                autopilot = AutoPilot(
                    templates=templates,
                    regions=regions,
                    click_cooldown_s=float(autopilot_cfg.get("click_cooldown_s", 0.5)),
                    template_thresholds=dict(autopilot_cfg.get("template_thresholds", {})),
                )

            screen = autopilot.detect_screen(frame)
            is_dead = screen == "DEAD" or (screen != "RUNNING" and is_death_like_frame(frame))
            is_upgrade = _is_upgrade_dialog(frame, templates, regions)
            hud_values = read_hud_telemetry(frame, regions=regions)

            analysis = analyze_scene(
                frame,
                templates=templates,
                grid_rows=int(detect_cfg["grid_rows"]),
                grid_cols=int(detect_cfg["grid_cols"]),
                enemy_hsv_lower=tuple(detect_cfg["enemy_hsv_lower"]),
                enemy_hsv_upper=tuple(detect_cfg["enemy_hsv_upper"]),
                enemy_min_area=float(detect_cfg["enemy_min_area"]),
                interact_threshold=float(detect_cfg["interact_threshold"]),
            )

            analysis.setdefault("projectiles", [])
            onnx_detections: list[ObjectDetection] = []
            if onnx_detector is not None and onnx_detector.enabled:
                onnx_detections = onnx_detector.detect(frame)
                analysis["projectiles"].extend(
                    [det for det in onnx_detections if det.label == "projectile"]
                )
            scene_memory.update(onnx_detections)

            snapshot = build_scene_snapshot(
                frame_id=frame_id,
                ts=time.time(),
                frame_width=w,
                analysis=analysis,
                hud_values=hud_values,
                is_dead=is_dead,
                is_upgrade=is_upgrade,
            )

            if mode == BotMode.ACTIVE and snapshot.is_dead:
                mode = state_machine.on_events(dead_detected=True)
                recovery_started_ts = time.time()
                recovery_attempts = 0

            allow_tab_scan = bool(mvp_cfg.get("allow_map_scan_tab", False)) or bool(
                max_enabled and max_cfg.get("explore_with_tab", False)
            )
            map_scan_now = False
            if allow_tab_scan:
                map_scan_tick += 1
                if map_scan_tick >= int(mvp_cfg.get("map_scan_interval_ticks", 180)):
                    map_scan_now = True
                    map_scan_tick = 0

            heuristic_action = None
            if mode == BotMode.ACTIVE and not snapshot.is_dead and not snapshot.is_upgrade:
                heuristic_action = heuristic_pilot.act(frame, include_cam_yaw=True)

            action = choose_mvp_action(
                snapshot,
                mode=mode,
                heuristic_action=heuristic_action,
                allow_map_scan=allow_tab_scan,
                map_scan_now=map_scan_now,
            )

            restart_event = None
            if mode == BotMode.RECOVERY:
                set_move(0)
                if screen == "RUNNING" and not snapshot.is_dead:
                    mode = state_machine.on_events(running_restored=True)
                    recovery_attempts = 0
                    recovery_started_ts = 0.0
                    restart_event = "running_restored"
                else:
                    now = time.time()
                    if (
                        action.press_r
                        and (now - last_restart_ts) >= restart_cooldown_s
                        and recovery_attempts < restart_max_attempts
                    ):
                        release_all_keys()
                        hold("r", dt=restart_hold_s)
                        last_restart_ts = now
                        recovery_attempts += 1
                        restart_event = f"hold_r_attempt_{recovery_attempts}"
                    if (now - recovery_started_ts) >= restart_wait_timeout_s:
                        clicked = (
                            _try_click_template(frame, templates, regions, "tpl_confirm", "REG_DEAD_CONFIRM", 0.6)
                            or _try_click_template(frame, templates, regions, "tpl_play", "REG_MAIN_PLAY", 0.65)
                        )
                        if clicked:
                            restart_event = "fallback_click_menu"
                            recovery_started_ts = now

            elif mode == BotMode.ACTIVE:
                apply_cam_yaw(action.yaw, cam_yaw_pixels=cam_yaw_pixels)
                set_move(action.dir_id)
                if int(action.jump) == 1:
                    tap("space", dt=0.005)
                if int(action.slide) == 1:
                    tap("shift", dt=0.005)
                if action.press_tab:
                    tap("tab", dt=0.01)
                if bool(mvp_cfg.get("auto_pick_upgrade_with_space", True)) and action.press_space:
                    now = time.time()
                    if (now - last_upgrade_space_ts) >= upgrade_space_cooldown_s:
                        tap("space", dt=0.01)
                        last_upgrade_space_ts = now
            else:
                set_move(0)

            if mode == BotMode.PANIC:
                set_move(0)
                release_all_keys()

            if max_enabled and bool(max_cfg.get("threat_scoring", True)):
                threats = score_enemy_threats(
                    [
                        ObjectDetection(label=item.label, rect=item.rect, score=item.score)
                        for item in snapshot.enemies
                    ],
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
                boss_prep, boss_name = should_enter_boss_prep(elapsed, boss_schedule)
            else:
                threats = []
                preferred_direction = "center"
                boss_prep = False
                boss_name = None

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
                )
                logger.log(event)

            if overlay_enabled:
                overlay = _draw_runtime_overlay(
                    frame,
                    analysis,
                    snapshot,
                    mode=mode,
                    action_reason=action.reason,
                )
                cv2.imshow(overlay_window, overlay)
                if not overlay_topmost_applied and runtime_cfg.get("overlay_topmost", True):
                    _set_overlay_topmost(overlay_window)
                    overlay_topmost_applied = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        release_all_keys()
        logger.close()
        if overlay_enabled:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runtime bot (MVP): F8 toggle, F12 panic, JSONL + overlay.",
    )
    parser.add_argument("--window", default=None, help="Window title substring")
    parser.add_argument("--config", default="config/bot_profile.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--templates-dir", default=None, help="Templates directory")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay window")
    parser.add_argument("--no-hotkeys", action="store_true", help="Disable WinAPI hotkeys")
    parser.add_argument(
        "--print-default-config",
        action="store_true",
        help="Print default config in YAML and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.print_default_config:
        if DEFAULT_CONFIG is None:
            raise RuntimeError("DEFAULT_CONFIG is not initialized")
        print(dump_default_config_yaml(), end="")
    else:
        run(args)
