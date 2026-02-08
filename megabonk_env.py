from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()

import ctypes  # noqa: E402
import time  # noqa: E402
import random  # noqa: E402
import atexit  # noqa: E402
from pathlib import Path  # noqa: E402
from collections import deque  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import mss  # noqa: E402
import numpy as np  # noqa: E402
import pydirectinput as di  # noqa: E402
from gymnasium import spaces  # noqa: E402

from autopilot import AutoPilot, HeuristicAutoPilot  # noqa: E402
from megabonk_bot.recognition import analyze_scene, draw_recognition_overlay  # noqa: E402
from megabonk_bot.hud import read_hud_values  # noqa: E402
from megabonk_bot.regions import build_regions  # noqa: E402
from megabonk_bot.templates import load_templates  # noqa: E402
from megabonk_bot.vision import find_in_region  # noqa: E402
from window_capture import WindowCapture  # noqa: E402

di.PAUSE = 0.0
di.FAILSAFE = False

# ---- управление ----
def key_on(key: str): di.keyDown(key)
def key_off(key: str): di.keyUp(key)

def tap(key: str, dt=0.01):
    di.keyDown(key)
    time.sleep(dt)
    di.keyUp(key)


def hold(key: str, dt=0.5):
    di.keyDown(key)
    time.sleep(max(0.0, float(dt)))
    di.keyUp(key)

def set_move(dir_id: int):
    # 0 стоп, 1 W, 2 S, 3 A, 4 D, 5 WA, 6 WD, 7 SA, 8 SD
    mapping = {
        0: [],
        1: ["w"], 2: ["s"], 3: ["a"], 4: ["d"],
        5: ["w","a"], 6: ["w","d"], 7: ["s","a"], 8: ["s","d"]
    }
    want = set(mapping[int(dir_id)])
    for k in ["w","a","s","d"]:
        (key_on if k in want else key_off)(k)

def release_all_keys():
    for k in [
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
            di.keyUp(k)
        except Exception:
            pass

atexit.register(release_all_keys)

DEATH_TPL_HINTS = ("dead", "game_over", "gameover", "death")
CONFIRM_TPL_HINTS = ("confirm",)
RUNNING_TPL_HINTS = ("minimap", "lvl", "level")

HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
GWL_EXSTYLE = -20
GWL_STYLE = -16
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
LWA_COLORKEY = 0x00000001
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZE = 0x20000000
WS_MAXIMIZE = 0x01000000
WS_SYSMENU = 0x00080000


def _set_window_topmost(window_name: str, topmost: bool = True) -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    user32 = getattr(ctypes.windll, "user32", None)
    if user32 is None:
        return False
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return False
    insert_after = HWND_TOPMOST if topmost else HWND_NOTOPMOST
    return bool(
        user32.SetWindowPos(
            hwnd,
            insert_after,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        )
    )


def _set_window_transparent(window_name: str, colorkey=(0, 0, 0)) -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    user32 = getattr(ctypes.windll, "user32", None)
    if user32 is None:
        return False
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return False
    exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    exstyle |= WS_EX_LAYERED | WS_EX_TRANSPARENT
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, exstyle)
    r, g, b = colorkey
    colorref = int(r) | (int(g) << 8) | (int(b) << 16)
    return bool(user32.SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY))


def _move_window(window_name: str, x: int, y: int, w: int, h: int) -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    user32 = getattr(ctypes.windll, "user32", None)
    if user32 is None:
        return False
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return False
    return bool(
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            int(x),
            int(y),
            int(w),
            int(h),
            SWP_SHOWWINDOW,
        )
    )


def _set_window_borderless(window_name: str) -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    user32 = getattr(ctypes.windll, "user32", None)
    if user32 is None:
        return False
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return False
    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
    style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZE | WS_MAXIMIZE | WS_SYSMENU)
    user32.SetWindowLongW(hwnd, GWL_STYLE, style)
    return bool(
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        )
    )


def _pick_death_templates(templates):
    if not templates:
        return []
    return [
        name
        for name in templates
        if any(hint in name.lower() for hint in DEATH_TPL_HINTS)
    ]

def _pick_confirm_templates(templates):
    if not templates:
        return []
    return [
        name
        for name in templates
        if any(hint in name.lower() for hint in CONFIRM_TPL_HINTS)
    ]


def _pick_running_templates(templates):
    if not templates:
        return []
    return [
        name
        for name in templates
        if any(hint in name.lower() for hint in RUNNING_TPL_HINTS)
    ]


def _patch_mean(gray, cx, cy, half=1):
    h, w = gray.shape[:2]
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half + 1)
    y1 = min(h, cy + half + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(gray[y0:y1, x0:x1].mean())


def _hud_pixels_mean(gray84):
    h, w = gray84.shape[:2]
    points = [
        (int(w * 0.85), int(h * 0.06)),
        (int(w * 0.92), int(h * 0.08)),
        (int(w * 0.78), int(h * 0.05)),
        (int(w * 0.08), int(h * 0.05)),
    ]
    values = [_patch_mean(gray84, cx, cy, half=1) for cx, cy in points]
    return float(np.mean(values))


def _hud_pixels_dark(gray84):
    return _hud_pixels_mean(gray84) < 45.0


def _top_rainbow_like(
    frame_bgr,
    height_ratio=0.035,
    sat_thr=120,
    val_thr=120,
    colorful_ratio_thr=0.25,
    hue_bins=12,
    min_bins=5,
):
    h, w = frame_bgr.shape[:2]
    strip_h = max(1, int(h * height_ratio))
    strip = frame_bgr[:strip_h, :w]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > sat_thr) & (val > val_thr)
    colorful = int(mask.sum())
    total = mask.size
    if total == 0:
        return False
    if colorful / total < colorful_ratio_thr:
        return False
    hue_vals = hue[mask].ravel()
    if hue_vals.size == 0:
        return False
    hist, _ = np.histogram(hue_vals, bins=hue_bins, range=(0, 180))
    min_bin_count = max(1, int(colorful * 0.05))
    active_bins = int((hist >= min_bin_count).sum())
    return active_bins >= min_bins


def _is_upgrade_dialog(frame_bgr, templates, regions, threshold=0.62):
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




def _top_left_time_cells_black(
    gray84,
    rows=12,
    cols=20,
    cells=2,
    black_thr=10.0,
    dark_ratio_thr=0.98,
):
    h, w = gray84.shape[:2]
    cell_h = max(1, h // max(1, int(rows)))
    cell_w = max(1, w // max(1, int(cols)))
    for idx in range(max(1, int(cells))):
        x0 = idx * cell_w
        x1 = w if idx == cells - 1 else min(w, x0 + cell_w)
        patch = gray84[0:cell_h, x0:x1]
        if patch.size == 0:
            return False
        patch_mean = float(patch.mean())
        dark_ratio = float((patch <= black_thr).mean())
        if patch_mean > black_thr or dark_ratio < dark_ratio_thr:
            return False
    return True


def _fast_death_check(gray84, mean_thr=35.0, std_thr=12.0):
    mean = float(gray84.mean())
    std = float(gray84.std())
    return mean < mean_thr and std < std_thr

def is_death_like(
    frame_gray_84,
    frame_bgr=None,
    templates=None,
    regions=None,
    death_tpl_names=None,
    confirm_tpl_names=None,
    hard_tpl_threshold=0.62,
    soft_tpl_threshold=0.50,
    confirm_threshold=0.48,
    confirm_mean_threshold=95.0,
    debug_info=None,
):
    """Грубая детекция смерти/меню.

    1) Ищем шаблоны UI (“DEAD”/“GAME OVER”) в REG_DEAD с порогами.
    2) Проверяем фиксированные пиксели HUD (если они темнеют — чаще меню/смерть).
    3) Дополнительно фильтруем по средней яркости/дисперсии всего кадра.
    """
    best_score = 0.0
    best_tpl = None
    confirm_score = 0.0
    confirm_tpl = None
    if frame_bgr is not None and templates and regions:
        region = regions.get("REG_DEAD")
        if region:
            names = death_tpl_names or _pick_death_templates(templates)
            for name in names:
                tpl = templates.get(name)
                if tpl is None:
                    continue
                found, _, score = find_in_region(
                    frame_bgr, tpl, region, threshold=hard_tpl_threshold
                )
                score = float(score)
                if score > best_score:
                    best_score = score
                    best_tpl = name
                if found:
                    if debug_info is not None:
                        debug_info.update(
                            {
                                "best_tpl": name,
                                "best_score": best_score,
                            }
                        )
                    return True

    mean = float(frame_gray_84.mean())
    std = float(frame_gray_84.std())
    dark_overlay = mean < 38.0 and std < 14.0
    hud_mean = _hud_pixels_mean(frame_gray_84)
    hud_dark = hud_mean < 45.0
    time_cells_black = _top_left_time_cells_black(frame_gray_84)
    confirm_bright_ok = mean < confirm_mean_threshold

    if frame_bgr is not None and templates and regions:
        region = regions.get("REG_DEAD_CONFIRM")
        if region:
            names = confirm_tpl_names or _pick_confirm_templates(templates)
            for name in names:
                tpl = templates.get(name)
                if tpl is None:
                    continue
                found, _, score = find_in_region(
                    frame_bgr, tpl, region, threshold=confirm_threshold
                )
                score = float(score)
                if score > confirm_score:
                    confirm_score = score
                    confirm_tpl = name
                if found and (dark_overlay or hud_dark or confirm_bright_ok):
                    if debug_info is not None:
                        debug_info.update(
                            {
                                "confirm_tpl": name,
                                "confirm_score": confirm_score,
                            }
                        )
                    return True

    if time_cells_black:
        if debug_info is not None:
            debug_info.update({"time_cells_black": True})
        return True

    if best_score >= soft_tpl_threshold and (dark_overlay or hud_dark):
        if debug_info is not None:
            debug_info.update(
                {
                    "best_tpl": best_tpl,
                    "best_score": best_score,
                }
            )
        return True
    if confirm_score >= confirm_threshold and (dark_overlay or hud_dark):
        if debug_info is not None:
            debug_info.update(
                {
                    "confirm_tpl": confirm_tpl,
                    "confirm_score": confirm_score,
                }
            )
        return True
    if confirm_score >= confirm_threshold and confirm_bright_ok:
        if debug_info is not None:
            debug_info.update(
                {
                    "confirm_tpl": confirm_tpl,
                    "confirm_score": confirm_score,
                }
            )
        return True
    if debug_info is not None:
        debug_info.update(
            {
                "best_tpl": best_tpl,
                "best_score": best_score,
                "confirm_tpl": confirm_tpl,
                "confirm_score": confirm_score,
                "mean": mean,
                "std": std,
                "hud_mean": hud_mean,
                "hud_dark": hud_dark,
                "dark_overlay": dark_overlay,
                "time_cells_black": time_cells_black,
            }
        )
    return (dark_overlay and hud_dark) or time_cells_black

class MegabonkEnv(gym.Env):
    """
    Observations: uint8 (84, 84, 4) — стек 4 кадров (grayscale).
    Actions:
        - if include_cam_yaw and include_cam_pitch:
          MultiDiscrete [dir(0..8), cam_yaw(0..2), cam_pitch(0..2), jump(0/1), slide(0/1)]
        - if include_cam_yaw and not include_cam_pitch:
          MultiDiscrete [dir(0..8), cam_yaw(0..2), jump(0/1), slide(0/1)]
        - if not include_cam_yaw and include_cam_pitch:
          MultiDiscrete [dir(0..8), cam_pitch(0..2), jump(0/1), slide(0/1)]
        - if neither:
          MultiDiscrete [dir(0..8), jump(0/1), slide(0/1)]
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        region: dict | None = None,
        step_hz: int = 12,
        frame_stack: int = 4,
        frame_skip_range: tuple[int, int] = (6, 10),
        sticky_steps_range: tuple[int, int] = (2, 4),
        jump_key: str = "space",
        slide_key: str = "shift",
        include_cam_yaw: bool = True,
        include_cam_pitch: bool = True,
        cam_yaw_pixels: int = 160,
        cam_pitch_pixels: int = 60,
        use_arrow_cam: bool = False,
        use_heuristic_autopilot: bool = False,
        dead_r_cooldown: float = 3.5,
        restart_cooldown_s: float | None = None,
        restart_hold_s: float = 3.5,
        restart_wait_timeout_s: float = 8.0,
        death_hysteresis_frames: int = 3,
        reset_sequence=None,
        templates_dir: str | None = "templates",
        regions_builder=build_regions,
        window_title: str | None = None,
        cap: WindowCapture | None = None,
        debug_recognition: bool = False,
        debug_recognition_dir: str = "dbg",
        debug_recognition_every_s: float = 2.0,
        recognition_grid: tuple[int, int] = (12, 20),
        debug_recognition_show: bool = False,
        debug_recognition_window: str = "Megabonk Recognition",
        debug_recognition_topmost: bool = True,
        debug_recognition_transparent: bool = True,
        hud_ocr_every_s: float = 1.5,
        death_check_every: int = 8,
        reward_danger_k: float = 0.15,
        reward_stuck_k: float = 0.08,
        reward_loot_k: float = 0.12,
        reward_enemy_roi: tuple[float, float, float, float] = (0.35, 0.55, 0.30, 0.35),
        reward_loot_roi: tuple[float, float, float, float] = (0.35, 0.38, 0.30, 0.36),
        reward_enemy_hsv_lower: tuple[int, int, int] = (45, 80, 40),
        reward_enemy_hsv_upper: tuple[int, int, int] = (85, 255, 255),
        reward_coin_hsv_lower: tuple[int, int, int] = (18, 120, 80),
        reward_coin_hsv_upper: tuple[int, int, int] = (35, 255, 255),
        reward_stuck_diff_threshold: float = 3.0,
    ):
        super().__init__()
        self.cap = cap
        if self.cap is None and window_title:
            self.cap = WindowCapture.create(window_title)
        if self.cap is not None:
            self.cap.focus(topmost=True)

        if self.cap is not None:
            self.region = self.cap.get_bbox()
            self.sct = None
        else:
            if region is None:
                raise ValueError("region is required when window_title/cap not provided")
            self.region = region
            self.sct = mss.mss()
        self.dt = 1.0 / step_hz
        self.frame_skip_range = frame_skip_range
        self.sticky_steps_range = sticky_steps_range
        self.jump_key = jump_key
        self.slide_key = slide_key
        self.include_cam_yaw = include_cam_yaw
        self.include_cam_pitch = include_cam_pitch
        self.cam_yaw_pixels = int(cam_yaw_pixels)
        self.cam_pitch_pixels = int(cam_pitch_pixels)
        self.use_arrow_cam = bool(use_arrow_cam)
        if self.use_arrow_cam:
            print("[WARN] use_arrow_cam=True больше не поддерживается, используется движение мышью.")
            self.use_arrow_cam = False
        self.use_heuristic_autopilot = use_heuristic_autopilot
        if restart_cooldown_s is None:
            self.restart_cooldown_s = float(dead_r_cooldown)
        else:
            self.restart_cooldown_s = float(restart_cooldown_s)
        self.restart_hold_s = float(restart_hold_s)
        self.restart_wait_timeout_s = float(restart_wait_timeout_s)
        self.death_hysteresis_frames = max(1, int(death_hysteresis_frames))

        if self.include_cam_yaw and self.include_cam_pitch:
            self.action_space = spaces.MultiDiscrete([9, 3, 3, 2, 2])
        elif self.include_cam_yaw:
            self.action_space = spaces.MultiDiscrete([9, 3, 2, 2])
        elif self.include_cam_pitch:
            self.action_space = spaces.MultiDiscrete([9, 3, 2, 2])
        else:
            self.action_space = spaces.MultiDiscrete([9, 2, 2])
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, frame_stack), dtype=np.uint8)

        self.frames = deque(maxlen=frame_stack)

        self.autopilot = None
        self.heuristic_pilot = None
        self.templates = load_templates(templates_dir) if templates_dir else {}
        self._regions_builder = regions_builder
        self.regions = (
            self._regions_builder(self.region["width"], self.region["height"])
            if templates_dir
            else {}
        )
        self._death_tpl_names = _pick_death_templates(self.templates)
        self._confirm_tpl_names = _pick_confirm_templates(self.templates)
        self._running_tpl_names = _pick_running_templates(self.templates)
        if templates_dir:
            self.autopilot = AutoPilot(templates=self.templates, regions=self.regions)
        if self.use_heuristic_autopilot:
            self.heuristic_pilot = HeuristicAutoPilot()

        self.debug_recognition = bool(debug_recognition)
        self.debug_recognition_dir = Path(debug_recognition_dir)
        self.debug_recognition_every_s = float(debug_recognition_every_s)
        self.hud_ocr_every_s = float(hud_ocr_every_s)
        self.recognition_grid = recognition_grid
        self._dbg_recognition_idx = 0
        if self.debug_recognition:
            self.debug_recognition_dir.mkdir(parents=True, exist_ok=True)

        # как “перезапускать” ран (подстроишь под меню)
        self.reset_sequence = reset_sequence or [
            ("hold", "r", 3.5),
            ("sleep", None, 0.4),
        ]

        self._last_obs = None
        self._last_frame = None
        self._sticky_dir = 0
        self._sticky_left = 0
        self._last_dead_r_time = 0.0
        self._death_check_every = max(1, int(death_check_every))
        self._death_check_idx = 0
        self._death_like_streak = 0
        self._dbg_death_ts = 0.0
        self._dbg_recognition_ts = 0.0
        self._dbg_hud_ts = 0.0
        self._hud_ocr_ts = 0.0
        self._last_hud_values: dict[str, str] = {}
        self._event_idx = 0
        self._prof_last_log_ts = time.time()
        self._prof_step_count = 0
        self._prof_accum = {
            "capture": 0.0,
            "preprocess": 0.0,
            "death": 0.0,
            "hud": 0.0,
            "policy": 0.0,
            "input": 0.0,
        }
        self.reward_danger_k = float(reward_danger_k)
        self.reward_stuck_k = float(reward_stuck_k)
        self.reward_loot_k = float(reward_loot_k)
        self.reward_enemy_roi = reward_enemy_roi
        self.reward_loot_roi = reward_loot_roi
        self.reward_enemy_hsv_lower = np.array(reward_enemy_hsv_lower, dtype=np.uint8)
        self.reward_enemy_hsv_upper = np.array(reward_enemy_hsv_upper, dtype=np.uint8)
        self.reward_coin_hsv_lower = np.array(reward_coin_hsv_lower, dtype=np.uint8)
        self.reward_coin_hsv_upper = np.array(reward_coin_hsv_upper, dtype=np.uint8)
        self.reward_stuck_diff_threshold = float(reward_stuck_diff_threshold)
        self._last_reward_gray = None
        self._last_reward_forward = False

    def _debug_death_like(self, frame, every_s=2.0):
        now = time.time()
        if now - self._dbg_death_ts < every_s:
            return
        self._dbg_death_ts = now
        gray84 = self._to_gray84(frame)
        debug_info = {}
        death_like = is_death_like(
            gray84,
            frame_bgr=frame,
            templates=self.templates,
            regions=self.regions,
            death_tpl_names=self._death_tpl_names,
            confirm_tpl_names=self._confirm_tpl_names,
            debug_info=debug_info,
        )
        best_tpl = debug_info.get("best_tpl")
        best_score = debug_info.get("best_score", 0.0)
        confirm_tpl = debug_info.get("confirm_tpl")
        confirm_score = debug_info.get("confirm_score", 0.0)
        mean = debug_info.get("mean", 0.0)
        std = debug_info.get("std", 0.0)
        hud_mean = debug_info.get("hud_mean", 0.0)
        hud_dark = debug_info.get("hud_dark", False)
        dark_overlay = debug_info.get("dark_overlay", False)
        time_cells_black = debug_info.get("time_cells_black", False)
        print(
            "[DBG] DEATH "
            f"like={death_like} mean={mean:.1f} std={std:.1f} "
            f"hud_mean={hud_mean:.1f} hud_dark={hud_dark} "
            f"overlay_dark={dark_overlay} time_cells_black={time_cells_black} "
            f"best_tpl={best_tpl} best_score={best_score:.3f} "
            f"confirm_tpl={confirm_tpl} confirm_score={confirm_score:.3f}"
        )

    def _log_event(self, name: str, **fields):
        self._event_idx += 1
        parts = [f"[EVT] {self._event_idx:06d} {name}"]
        for key, value in sorted(fields.items()):
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")
        print(" ".join(parts))

    def _prof_add(self, stage: str, duration_s: float):
        if stage in self._prof_accum:
            self._prof_accum[stage] += float(duration_s)

    def _maybe_log_profile(self):
        now = time.time()
        dt = now - self._prof_last_log_ts
        if dt < 1.0:
            return
        steps = max(1, self._prof_step_count)
        capture_ms = self._prof_accum["capture"] / steps * 1000.0
        preprocess_ms = self._prof_accum["preprocess"] / steps * 1000.0
        death_ms = self._prof_accum["death"] / steps * 1000.0
        hud_ms = self._prof_accum["hud"] / steps * 1000.0
        policy_ms = self._prof_accum["policy"] / steps * 1000.0
        input_ms = self._prof_accum["input"] / steps * 1000.0
        fps_est = steps / dt
        print(
            "dt="
            f"{dt:.2f}s capture={capture_ms:.2f}ms preprocess={preprocess_ms:.2f}ms "
            f"death={death_ms:.2f}ms hud={hud_ms:.2f}ms policy={policy_ms:.2f}ms "
            f"input={input_ms:.2f}ms fps_est={fps_est:.2f}"
        )
        self._prof_last_log_ts = now
        self._prof_step_count = 0
        for key in self._prof_accum:
            self._prof_accum[key] = 0.0

    def _maybe_update_hud_values(self, frame, now):
        if self.hud_ocr_every_s <= 0:
            return None
        if now - self._hud_ocr_ts < self.hud_ocr_every_s:
            return self._last_hud_values.get("time")
        self._hud_ocr_ts = now
        hud_start = time.perf_counter()
        hud_values = read_hud_values(frame, regions=self.regions)
        self._prof_add("hud", time.perf_counter() - hud_start)
        self._last_hud_values = hud_values
        time_val = hud_values.get("time")
        if time_val is None:
            self._log_event("HUD_TIME_FAIL", time=time_val)
        else:
            self._log_event("HUD_TIME_OK", time=time_val)
        return time_val

    def _finish_step_profile(self):
        self._prof_step_count += 1
        self._maybe_log_profile()

    def _apply_cam_yaw(self, yaw_id: int):
        if not self.include_cam_yaw:
            return
        mapping = {0: -self.cam_yaw_pixels, 1: 0, 2: self.cam_yaw_pixels}
        dx = mapping.get(int(yaw_id), 0)
        if dx != 0:
            di.moveRel(dx, 0, duration=0)

    def _apply_cam_pitch(self, pitch_id: int):
        if not self.include_cam_pitch:
            return
        mapping = {0: -self.cam_pitch_pixels, 1: 0, 2: self.cam_pitch_pixels}
        dy = mapping.get(int(pitch_id), 0)
        if dy != 0:
            di.moveRel(0, dy, duration=0)

    def _grab_frame(self):
        frame = None
        if self.cap is not None:
            try:
                frame = self.cap.grab()
            except Exception:
                frame = None
            if frame is None or frame.size == 0:
                try:
                    self.cap.focus(topmost=True)
                    time.sleep(0.05)
                    frame = self.cap.grab()
                except Exception:
                    frame = None
        else:
            try:
                frame = np.array(self.sct.grab(self.region))[:, :, :3]
            except Exception:
                frame = None
        if frame is None or frame.size == 0:
            if self._last_frame is not None:
                return self._last_frame
            height = int(self.region["height"])
            width = int(self.region["width"])
            return np.zeros((height, width, 3), dtype=np.uint8)
        self._refresh_regions_for_frame(frame)
        self._last_frame = frame
        return frame

    def _refresh_regions_for_frame(self, frame):
        h, w = frame.shape[:2]
        if self.region["width"] == w and self.region["height"] == h:
            return
        self.region["width"] = w
        self.region["height"] = h
        if self.regions:
            self.regions = self._regions_builder(w, h)

    def _to_gray84(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return gray.astype(np.uint8)

    def _find_center_area_in_roi(self, frame, lower, upper, roi_rel):
        h, w = frame.shape[:2]
        x0 = int(roi_rel[0] * w)
        y0 = int(roi_rel[1] * h)
        rw = int(roi_rel[2] * w)
        rh = int(roi_rel[3] * h)
        x1 = min(w, x0 + rw)
        y1 = min(h, y0 + rh)
        if x1 <= x0 or y1 <= y0:
            return None, 0.0, 0.0
        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        area = 0.0
        center = None
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            if area > 0.0:
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"]) + x0
                    cy = int(moments["m01"] / moments["m00"]) + y0
                    center = (cx, cy)
        roi_area = float((x1 - x0) * (y1 - y0))
        area_frac = area / roi_area if roi_area > 0 else 0.0
        return center, area, area_frac

    def _is_reward_stuck(self, frame, forwardish):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_reward_gray is None:
            self._last_reward_gray = gray
            self._last_reward_forward = forwardish
            return False
        diff = cv2.absdiff(gray, self._last_reward_gray)
        self._last_reward_gray = gray
        was_forward = self._last_reward_forward
        self._last_reward_forward = forwardish
        if not was_forward:
            return False
        return float(diff.mean()) < self.reward_stuck_diff_threshold

    def _get_obs(self):
        # гарантируем ровно frame_stack кадров всегда
        if len(self.frames) == 0:
            f = self._to_gray84(self._grab_frame())
            self.frames.append(f)

        while len(self.frames) < self.frames.maxlen:
            self.frames.append(self.frames[-1])

        obs = np.stack(list(self.frames), axis=-1)  # (84,84,stack)
        return obs.astype(np.uint8)

    def _do_reset_sequence(self):
        # отпустить движение на всякий
        set_move(0)
        key_off(self.jump_key)
        key_off(self.slide_key)

        for kind, key, t in self.reset_sequence:
            if kind == "tap":
                tap(key, dt=0.01)
                time.sleep(t)
            elif kind == "hold":
                hold(key, dt=t)
            elif kind == "sleep":
                time.sleep(t)

    def _dump_recognition_debug(self, frame, analysis):
        overlay = draw_recognition_overlay(frame, analysis)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._dbg_recognition_idx += 1
        filename = f"rec_{ts}_{self._dbg_recognition_idx:06d}.png"
        out_path = self.debug_recognition_dir / filename
        cv2.imwrite(str(out_path), overlay)
        return out_path

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._do_reset_sequence()
        self._sticky_dir = 0
        self._sticky_left = 0
        self._death_check_idx = 0
        self._death_like_streak = 0

        # ждём, пока не будет “похоже на игру”
        self.frames.clear()
        frame = self._grab_frame()
        f = self._to_gray84(frame)
        self.frames.append(f)
        obs = self._get_obs()
        t0 = time.time()
        while time.time() - t0 < 8.0:
            frame = self._grab_frame()
            f = self._to_gray84(frame)
            self.frames.append(f)
            if self._is_running_frame(frame, f):
                break
            time.sleep(0.1)

        obs = self._get_obs()
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        capture_start = time.perf_counter()
        frame = self._grab_frame()
        self._prof_add("capture", time.perf_counter() - capture_start)
        now = time.time()
        screen = "UNKNOWN"
        autopilot_action = None
        death_like_now = False
        action_source = "agent"

        hud_time_val = self._maybe_update_hud_values(frame, now)
        preprocess_start = time.perf_counter()
        # Упрощённый режим: оставляем только распознавание поверхностей/врагов в логах,
        # отключаем логику HUD/апгрейдов/меню и любые HUD-оверлеи.
        if self.debug_recognition:
            now = time.time()
            if now - self._dbg_recognition_ts >= self.debug_recognition_every_s:
                self._dbg_recognition_ts = now
                rows, cols = self.recognition_grid
                analysis = analyze_scene(
                    frame,
                    templates=self.templates,
                    grid_rows=rows,
                    grid_cols=cols,
                )
                surfaces = sum(1 for cell in analysis.get("grid", []) if cell.label == "surface")
                enemies = len(analysis.get("enemies", []))
                dbg_path = self._dump_recognition_debug(frame, analysis)
                time_val = (
                    hud_time_val
                    if hud_time_val is not None
                    else self._last_hud_values.get("time")
                )
                if time_val is not None:
                    print(
                        f"[DBG] scene surfaces={surfaces} enemies={enemies} "
                        f"time={time_val} shot={dbg_path}"
                    )
                else:
                    print(f"[DBG] scene surfaces={surfaces} enemies={enemies} shot={dbg_path}")
        yaw = 1
        pitch = 1
        if self.include_cam_yaw and self.include_cam_pitch:
            if len(action) == 3:
                dir_id, jump, slide = action
            elif len(action) == 4:
                dir_id, yaw, jump, slide = action
            else:
                dir_id, yaw, pitch, jump, slide = action
        elif self.include_cam_yaw:
            if len(action) == 3:
                dir_id, jump, slide = action
            else:
                dir_id, yaw, jump, slide = action
        elif self.include_cam_pitch:
            if len(action) == 3:
                dir_id, jump, slide = action
            else:
                dir_id, pitch, jump, slide = action
        else:
            dir_id, jump, slide = action
        self._prof_add("preprocess", time.perf_counter() - preprocess_start)

        if self.use_heuristic_autopilot and self.heuristic_pilot:
            death_start = time.perf_counter()
            gray84 = self._to_gray84(frame)
            death_suspect = _fast_death_check(gray84)
            if death_suspect:
                death_like_now = is_death_like(
                    gray84,
                    frame_bgr=frame,
                    templates=self.templates,
                    regions=self.regions,
                    death_tpl_names=self._death_tpl_names,
                    confirm_tpl_names=self._confirm_tpl_names,
                )
            self._prof_add("death", time.perf_counter() - death_start)
            screen = "DEAD" if death_like_now else "RUNNING"

        if self.use_heuristic_autopilot and self.heuristic_pilot and not death_like_now:
            policy_start = time.perf_counter()
            if self.include_cam_yaw:
                dir_id, yaw, jump, slide, reason = self.heuristic_pilot.act(
                    frame, include_cam_yaw=True
                )
            else:
                dir_id, jump, slide, reason = self.heuristic_pilot.act(
                    frame, include_cam_yaw=False
                )
            self._prof_add("policy", time.perf_counter() - policy_start)
            autopilot_action = f"heuristic:{reason}"
            action_source = "heuristic"
            if reason == "unstuck":
                self._log_event("UNSTUCK", reason=reason)

        input_start = time.perf_counter()
        self._apply_cam_yaw(yaw)
        self._apply_cam_pitch(pitch)
        if self._sticky_left <= 0:
            self._sticky_dir = int(dir_id)
            self._sticky_left = random.randint(*self.sticky_steps_range)
        dir_id = self._sticky_dir
        set_move(int(dir_id))

        if int(jump) == 1:
            self._log_event("JUMP", source=action_source, reason=autopilot_action)
            tap(self.jump_key, dt=0.005)
        if int(slide) == 1:
            self._log_event("SLIDE", source=action_source, reason=autopilot_action)
            tap(self.slide_key, dt=0.005)
        self._prof_add("input", time.perf_counter() - input_start)

        terminated = False
        r_alive = 0.0
        r_xp = 0.0
        r_dmg = 0.0
        r_danger = 0.0
        r_stuck = 0.0
        r_loot = 0.0
        frame_skip = random.randint(*self.frame_skip_range)
        for _ in range(frame_skip):
            time.sleep(self.dt)
            capture_start = time.perf_counter()
            frame_bgr = self._grab_frame()
            self._prof_add("capture", time.perf_counter() - capture_start)
            death_start = time.perf_counter()
            f = self._to_gray84(frame_bgr)
            self.frames.append(f)
            self._death_check_idx += 1
            now_loop = time.time()
            cooldown_active = (now_loop - self._last_dead_r_time) < self.restart_cooldown_s
            death_suspect = False if cooldown_active else _fast_death_check(f)
            death_like = False
            debug_info = {}
            hard_tpl_threshold = 0.62
            soft_tpl_threshold = 0.50
            confirm_threshold = 0.48
            confirm_mean_threshold = 95.0
            if death_suspect:
                death_like = is_death_like(
                    f,
                    frame_bgr=frame_bgr,
                    templates=self.templates,
                    regions=self.regions,
                    death_tpl_names=self._death_tpl_names,
                    confirm_tpl_names=self._confirm_tpl_names,
                    hard_tpl_threshold=hard_tpl_threshold,
                    soft_tpl_threshold=soft_tpl_threshold,
                    confirm_threshold=confirm_threshold,
                    confirm_mean_threshold=confirm_mean_threshold,
                    debug_info=debug_info,
                )
            if death_like:
                self._death_like_streak += 1
            else:
                self._death_like_streak = 0
            self._prof_add("death", time.perf_counter() - death_start)
            if death_like and self._death_like_streak >= self.death_hysteresis_frames:
                self._log_event(
                    "DEATH_DETECTED",
                    best_score=debug_info.get("best_score", 0.0),
                    best_tpl=debug_info.get("best_tpl"),
                    confirm_score=debug_info.get("confirm_score", 0.0),
                    confirm_tpl=debug_info.get("confirm_tpl"),
                    dark_overlay=debug_info.get("dark_overlay", False),
                    death_suspect=death_suspect,
                    death_streak=self._death_like_streak,
                    forced_check=False,
                    hud_dark=debug_info.get("hud_dark", False),
                    hud_mean=debug_info.get("hud_mean", 0.0),
                    mean=debug_info.get("mean", 0.0),
                    soft_tpl_threshold=soft_tpl_threshold,
                    hard_tpl_threshold=hard_tpl_threshold,
                    confirm_threshold=confirm_threshold,
                    confirm_mean_threshold=confirm_mean_threshold,
                    std=debug_info.get("std", 0.0),
                    time_cells_black=debug_info.get("time_cells_black", False),
                )
                terminated = True
                r_alive = -1.0
                self._restart_after_death(
                    death_streak=self._death_like_streak,
                    reason="death_like",
                )
                obs = self._get_obs()
                reward = r_alive + r_xp + r_dmg
                self._last_obs = obs
                self._sticky_left = max(0, self._sticky_left - 1)
                info = {
                    "screen": screen,
                    "r_alive": r_alive,
                    "r_xp": r_xp,
                    "r_dmg": r_dmg,
                    "autopilot": autopilot_action,
                    "time": self._last_hud_values.get("time"),
                }
                self._finish_step_profile()
                return obs, float(reward), terminated, False, info
            r_alive += 0.01

        forwardish = dir_id in (1, 5, 6)
        danger_center, danger_area, danger_frac = self._find_center_area_in_roi(
            frame,
            self.reward_enemy_hsv_lower,
            self.reward_enemy_hsv_upper,
            self.reward_enemy_roi,
        )
        loot_center, loot_area, loot_frac = self._find_center_area_in_roi(
            frame,
            self.reward_coin_hsv_lower,
            self.reward_coin_hsv_upper,
            self.reward_loot_roi,
        )
        if danger_center is not None and danger_area > 0:
            r_danger = -self.reward_danger_k * danger_frac
        if loot_center is not None and loot_area > 0:
            r_loot = self.reward_loot_k * loot_frac
        if self._is_reward_stuck(frame, forwardish):
            r_stuck = -self.reward_stuck_k

        obs = self._get_obs()
        reward = r_alive + r_xp + r_dmg + r_danger + r_stuck + r_loot

        # --- опционально: “прогресс” по HUD ---
        # Например, мерить среднюю яркость в ROI полоски XP/HP (нужно найти координаты на 84x84).
        # Это сильно помогает учиться не просто жить, а “качать прогресс”.

        self._last_obs = obs
        self._sticky_left = max(0, self._sticky_left - 1)
        info = {
            "screen": screen,
            "r_alive": r_alive,
            "r_xp": r_xp,
            "r_dmg": r_dmg,
            "r_danger": r_danger,
            "r_stuck": r_stuck,
            "r_loot": r_loot,
            "autopilot": autopilot_action,
            "time": self._last_hud_values.get("time"),
        }
        self._finish_step_profile()
        return obs, float(reward), terminated, False, info

    def _release_inputs_for_restart(self):
        set_move(0)
        key_off(self.jump_key)
        key_off(self.slide_key)
        key_off("space")
        key_off("shift")
        release_all_keys()

    def _wait_for_running(self, timeout_s: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            frame = self._grab_frame()
            gray = self._to_gray84(frame)
            if self._is_running_frame(frame, gray):
                return True
            time.sleep(0.1)
        return False

    def _restart_after_death(self, *, death_streak: int, reason: str):
        now = time.time()
        if now - self._last_dead_r_time < self.restart_cooldown_s:
            self._log_event(
                "RESTART_SKIPPED_COOLDOWN",
                reason=reason,
                death_streak=death_streak,
                cooldown_s=self.restart_cooldown_s,
            )
            return
        self._release_inputs_for_restart()
        self._log_event(
            "RESTART_START",
            reason=reason,
            death_streak=death_streak,
        )
        hold("r", dt=self.restart_hold_s)
        self._last_dead_r_time = time.time()
        if not self._wait_for_running(self.restart_wait_timeout_s):
            self._log_event(
                "RESTART_TIMEOUT",
                reason=reason,
                death_streak=death_streak,
            )
            hold("r", dt=self.restart_hold_s)
            self._last_dead_r_time = time.time()
            self._wait_for_running(self.restart_wait_timeout_s)
        self._log_event(
            "RESTART_DONE",
            reason=reason,
            death_streak=death_streak,
        )

    def _is_running_frame(self, frame_bgr, gray84):
        if frame_bgr is not None and self.templates and self.regions and self._running_tpl_names:
            for name in self._running_tpl_names:
                tpl = self.templates.get(name)
                if tpl is None:
                    continue
                region = self.regions.get("REG_MINIMAP")
                if "lvl" in name or "level" in name:
                    region = self.regions.get("REG_HUD")
                if not region:
                    continue
                found, _, _ = find_in_region(frame_bgr, tpl, region, threshold=0.55)
                if found:
                    return True
        if gray84 is None:
            return False
        time_cells_black = _top_left_time_cells_black(gray84)
        hud_dark = _hud_pixels_dark(gray84)
        return (not time_cells_black) and (not hud_dark)
