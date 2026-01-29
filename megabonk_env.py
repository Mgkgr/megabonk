import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mss
import cv2
import pydirectinput as di
from collections import deque

from autopilot import AutoPilot
from megabonk_bot.regions import build_regions
from megabonk_bot.templates import load_templates

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

# ---- детектор смерти/меню (очень грубо) ----
def is_death_like(frame_gray_84):
    # Заглушка: если экран очень тёмный/однородный, считаем смерть/меню.
    # Под себя лучше заменить на:
    # - шаблон (template matching) по слову "DEAD"/"GAME OVER"
    # - или проверку пикселей в точках HUD
    mean = float(frame_gray_84.mean())
    std = float(frame_gray_84.std())
    return (mean < 25 and std < 8)

class MegabonkEnv(gym.Env):
    """
    Observations: uint8 (84, 84, 4) — стек 4 кадров (grayscale).
    Actions: MultiDiscrete [dir(0..8), jump(0/1), slide(0/1)]
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        region: dict,
        step_hz: int = 12,
        frame_stack: int = 4,
        frame_skip_range: tuple[int, int] = (6, 10),
        sticky_steps_range: tuple[int, int] = (2, 4),
        jump_key: str = "space",
        slide_key: str = "shift",
        reset_sequence=None,
        templates_dir: str | None = "templates",
        regions_builder=build_regions,
    ):
        super().__init__()
        self.region = region
        self.dt = 1.0 / step_hz
        self.frame_skip_range = frame_skip_range
        self.sticky_steps_range = sticky_steps_range
        self.jump_key = jump_key
        self.slide_key = slide_key

        self.action_space = spaces.MultiDiscrete([9, 2, 2])
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, frame_stack), dtype=np.uint8)

        self.sct = mss.mss()
        self.frames = deque(maxlen=frame_stack)

        self.autopilot = None
        if templates_dir:
            templates = load_templates(templates_dir)
            regions = regions_builder(self.region["width"], self.region["height"])
            self.autopilot = AutoPilot(templates=templates, regions=regions)

        # как “перезапускать” ран (подстроишь под меню)
        self.reset_sequence = reset_sequence or [
            ("tap", "esc", 0.05),
            ("tap", "r", 0.05),
            ("sleep", None, 0.4),
            ("tap", "enter", 0.05),
            ("sleep", None, 0.6),
        ]

        self._last_obs = None
        self._sticky_dir = 0
        self._sticky_left = 0

    def _grab_frame(self):
        return np.array(self.sct.grab(self.region))[:, :, :3]

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
        f = self._to_gray84(self._grab_frame())
        self.frames.append(f)
        obs = self._get_obs()
        t0 = time.time()
        while time.time() - t0 < 6.0:
            f = self._to_gray84(self._grab_frame())
            self.frames.append(f)
            if not is_death_like(f):
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
            screen = self.autopilot.detect_screen(frame)
            if screen != "RUNNING":
                set_move(0)
                key_off(self.jump_key)
                key_off(self.slide_key)
                self._sticky_dir = 0
                self._sticky_left = 0
                autopilot_action = self.autopilot.ensure_running(frame)
                ok = self.autopilot.ensure_running(frame)
                if not ok:
                    dead_like = is_death_like(self._to_gray84(frame))
                    if dead_like:
                        self.autopilot.ensure_running_fallback_enter(max_enters=1)
                    else:
                        self.autopilot.ensure_running_fallback_enter(max_enters=6)
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
            self.autopilot.reset_enter_series()

        dir_id, jump, slide = action
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
            f = self._to_gray84(self._grab_frame())
            self.frames.append(f)
            if is_death_like(f):
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
