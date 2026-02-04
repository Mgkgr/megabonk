from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()

import time  # noqa: E402
import random  # noqa: E402
import atexit  # noqa: E402
from collections import deque  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import mss  # noqa: E402
import numpy as np  # noqa: E402
import pydirectinput as di  # noqa: E402
from gymnasium import spaces  # noqa: E402

from autopilot import AutoPilot, HeuristicAutoPilot  # noqa: E402
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
    for k in ["w", "a", "s", "d", "space", "lctrl", "shift", "enter", "esc"]:
        try:
            di.keyUp(k)
        except Exception:
            pass

atexit.register(release_all_keys)

DEATH_TPL_HINTS = ("dead", "game_over", "gameover", "death")
CONFIRM_TPL_HINTS = ("confirm",)


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


def is_death_like(
    frame_gray_84,
    frame_bgr=None,
    templates=None,
    regions=None,
    death_tpl_names=None,
    confirm_tpl_names=None,
    hard_tpl_threshold=0.62,
    soft_tpl_threshold=0.50,
    confirm_threshold=0.62,
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
                if found and (dark_overlay or hud_dark):
                    if debug_info is not None:
                        debug_info.update(
                            {
                                "confirm_tpl": name,
                                "confirm_score": confirm_score,
                            }
                        )
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
            }
        )
    return dark_overlay and hud_dark

class MegabonkEnv(gym.Env):
    """
    Observations: uint8 (84, 84, 4) — стек 4 кадров (grayscale).
    Actions: MultiDiscrete [dir(0..8), cam_yaw(0..2), jump(0/1), slide(0/1)]
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
        cam_yaw_pixels: int = 80,
        use_heuristic_autopilot: bool = False,
        dead_r_cooldown: float = 1.2,
        reset_sequence=None,
        templates_dir: str | None = "templates",
        regions_builder=build_regions,
        window_title: str | None = None,
        cap: WindowCapture | None = None,
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
        self.cam_yaw_pixels = int(cam_yaw_pixels)
        self.use_heuristic_autopilot = use_heuristic_autopilot
        self.dead_r_cooldown = float(dead_r_cooldown)

        if self.include_cam_yaw:
            self.action_space = spaces.MultiDiscrete([9, 3, 2, 2])
        else:
            self.action_space = spaces.MultiDiscrete([9, 2, 2])
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, frame_stack), dtype=np.uint8)

        self.frames = deque(maxlen=frame_stack)

        self.autopilot = None
        self.heuristic_pilot = None
        self.templates = load_templates(templates_dir) if templates_dir else {}
        self.regions = (
            regions_builder(self.region["width"], self.region["height"])
            if templates_dir
            else {}
        )
        self._death_tpl_names = _pick_death_templates(self.templates)
        self._confirm_tpl_names = _pick_confirm_templates(self.templates)
        if templates_dir:
            self.autopilot = AutoPilot(templates=self.templates, regions=self.regions)
        if self.use_heuristic_autopilot:
            self.heuristic_pilot = HeuristicAutoPilot()

        # как “перезапускать” ран (подстроишь под меню)
        self.reset_sequence = reset_sequence or [
            ("tap", "esc", 0.05),
            ("tap", "r", 0.05),
            ("sleep", None, 0.4),
            ("tap", "enter", 0.05),
            ("sleep", None, 0.6),
        ]

        self._last_obs = None
        self._last_frame = None
        self._sticky_dir = 0
        self._sticky_left = 0
        self._last_dead_r_time = 0.0
        self._dbg_death_ts = 0.0

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
        print(
            "[DBG] DEATH "
            f"like={death_like} mean={mean:.1f} std={std:.1f} "
            f"hud_mean={hud_mean:.1f} hud_dark={hud_dark} "
            f"overlay_dark={dark_overlay} "
            f"best_tpl={best_tpl} best_score={best_score:.3f} "
            f"confirm_tpl={confirm_tpl} confirm_score={confirm_score:.3f}"
        )

    def _apply_cam_yaw(self, yaw_id: int):
        if not self.include_cam_yaw:
            return
        mapping = {0: -self.cam_yaw_pixels, 1: 0, 2: self.cam_yaw_pixels}
        dx = mapping.get(int(yaw_id), 0)
        if dx != 0:
            di.moveRel(dx, 0, duration=0)

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
        self._last_frame = frame
        return frame

    def _to_gray84(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return gray.astype(np.uint8)

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
            elif kind == "sleep":
                time.sleep(t)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._do_reset_sequence()
        self._sticky_dir = 0
        self._sticky_left = 0

        # ждём, пока не будет “похоже на игру”
        self.frames.clear()
        frame = self._grab_frame()
        f = self._to_gray84(frame)
        self.frames.append(f)
        obs = self._get_obs()
        t0 = time.time()
        while time.time() - t0 < 6.0:
            frame = self._grab_frame()
            f = self._to_gray84(frame)
            self.frames.append(f)
            if not is_death_like(
                f,
                frame_bgr=frame,
                templates=self.templates,
                regions=self.regions,
                death_tpl_names=self._death_tpl_names,
                confirm_tpl_names=self._confirm_tpl_names,
            ):
                break
            time.sleep(0.1)

        obs = self._get_obs()
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        frame = self._grab_frame()
        screen = "UNKNOWN"
        autopilot_action = None

        if self.autopilot:
            self.autopilot.debug_scores(frame)
            self._debug_death_like(frame)
            screen = self.autopilot.detect_screen(frame)
            if screen == "DEAD":
                # DEAD: всегда жмём R, Enter только при явном подтверждающем шаблоне.
                now = time.time()
                if now - self._last_dead_r_time >= self.dead_r_cooldown:
                    tap("r", dt=0.01)
                    if self.autopilot._seen(frame, "tpl_confirm", "REG_DEAD_CONFIRM", 0.70):
                        time.sleep(0.05)
                        tap("enter", dt=0.01)
                    self._last_dead_r_time = now
                time.sleep(0.2)
                f = self._to_gray84(self._grab_frame())
                self.frames.append(f)
                obs = self._get_obs()
                info = {
                    "screen": screen,
                    "r_alive": -1.0,
                    "r_xp": 0.0,
                    "r_dmg": 0.0,
                    "xp_fill": 0.0,
                    "hp_fill": 0.0,
                    "autopilot": "dead_r",
                }
                return obs, -1.0, True, False, info
            if screen in ("MAIN_MENU", "CHAR_SELECT", "UNKNOWN"):
                set_move(0)
                key_off(self.jump_key)
                key_off(self.slide_key)
                di.keyUp("enter")
                di.keyUp("esc")
                self._sticky_dir = 0
                self._sticky_left = 0
                if screen == "CHAR_SELECT":
                    acted = self.autopilot.pick_fox_and_confirm(frame)
                    autopilot_action = "pick_fox" if acted else "char_wait"
                else:
                    acted = self.autopilot.ensure_running(frame)
                    if acted:
                        autopilot_action = "menu_click"
                    elif screen == "UNKNOWN":
                        autopilot_action = "menu_wait_unknown"
                        time.sleep(self.dt)
                    else:
                        dead_like = is_death_like(
                            self._to_gray84(frame),
                            frame_bgr=frame,
                            templates=self.templates,
                            regions=self.regions,
                            death_tpl_names=self._death_tpl_names,
                            confirm_tpl_names=self._confirm_tpl_names,
                        )
                        max_enters = 1 if dead_like else 6
                        did_enter = self.autopilot.ensure_running_fallback_enter(
                            max_enters=max_enters,
                            screen=screen,
                            death_like=dead_like,
                        )
                        autopilot_action = "enter_fallback" if did_enter else "menu_wait"
                time.sleep(self.dt)
                f = self._to_gray84(self._grab_frame())
                self.frames.append(f)
                obs = self._get_obs()
                info = {
                    "screen": screen,
                    "r_alive": 0.0,
                    "r_xp": 0.0,
                    "r_dmg": 0.0,
                    "xp_fill": 0.0,
                    "hp_fill": 0.0,
                    "autopilot": autopilot_action,
                }
                return obs, 0.0, False, False, info
            if screen == "RUNNING":
                self.autopilot.reset_enter_series()

        if self.include_cam_yaw:
            if len(action) == 3:
                dir_id, jump, slide = action
                yaw = 1
            else:
                dir_id, yaw, jump, slide = action
        else:
            dir_id, jump, slide = action
            yaw = 1

        if self.use_heuristic_autopilot and screen == "RUNNING" and self.heuristic_pilot:
            if self.include_cam_yaw:
                dir_id, yaw, jump, slide, reason = self.heuristic_pilot.act(
                    frame, include_cam_yaw=True
                )
            else:
                dir_id, jump, slide, reason = self.heuristic_pilot.act(
                    frame, include_cam_yaw=False
                )
                yaw = 1
            autopilot_action = f"heuristic:{reason}"

        self._apply_cam_yaw(yaw)
        if self._sticky_left <= 0:
            self._sticky_dir = int(dir_id)
            self._sticky_left = random.randint(*self.sticky_steps_range)
        dir_id = self._sticky_dir
        set_move(int(dir_id))

        if int(jump) == 1:
            tap(self.jump_key, dt=0.005)
        if int(slide) == 1:
            tap(self.slide_key, dt=0.005)

        terminated = False
        r_alive = 0.0
        r_xp = 0.0
        r_dmg = 0.0
        frame_skip = random.randint(*self.frame_skip_range)
        for _ in range(frame_skip):
            time.sleep(self.dt)
            frame_bgr = self._grab_frame()
            f = self._to_gray84(frame_bgr)
            self.frames.append(f)
            if is_death_like(
                f,
                frame_bgr=frame_bgr,
                templates=self.templates,
                regions=self.regions,
                death_tpl_names=self._death_tpl_names,
                confirm_tpl_names=self._confirm_tpl_names,
            ):
                terminated = True
                r_alive = -1.0
                break
            r_alive += 0.01

        obs = self._get_obs()
        reward = r_alive + r_xp + r_dmg

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
            "xp_fill": 0.0,
            "hp_fill": 0.0,
            "autopilot": autopilot_action,
        }
        return obs, float(reward), terminated, False, info
