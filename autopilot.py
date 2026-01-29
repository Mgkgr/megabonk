import time
from pathlib import Path

import cv2
import pydirectinput as di

from megabonk_bot.vision import find_in_region

di.PAUSE = 0.0
di.FAILSAFE = False


def click(x, y, delay=0.05):
    di.moveTo(x, y)
    di.click()
    time.sleep(delay)


def tap(key, delay=0.05):
    di.keyDown(key)
    time.sleep(0.01)
    di.keyUp(key)
    time.sleep(delay)


class AutoPilot:
    def __init__(self, templates, regions):
        self.t = templates
        self.r = regions
        self.enter_budget = 0
        self.enter_last_ts = 0.0
        self.enter_cooldown = 0.35
        self.click_last_ts = 0.0
        self.click_cooldown = 0.5

    def detect_screen(self, frame):
        if self._seen(frame, "tpl_dead", "REG_DEAD", 0.55):
            return "DEAD"
        if self._seen(frame, "tpl_char_select_title", "REG_CHAR_SELECT", 0.60):
            return "CHAR_SELECT"
        if self._seen(frame, "tpl_play", "REG_MAIN_PLAY", 0.60):
            return "MAIN_MENU"
        if self._seen(frame, "tpl_unlocks_title", "REG_UNLOCKS", 0.60):
            return "UNLOCKS_WEAPONS"
        if self._seen(frame, "tpl_katana", "REG_CHEST", 0.60) or self._seen(
            frame, "tpl_dexec", "REG_CHEST", 0.60
        ):
            return "CHEST_WEAPON_PICK"
        if self._seen(frame, "tpl_blood_tome", "REG_CHEST", 0.60) or self._seen(
            frame, "tpl_foliant_bottom1", "REG_CHEST", 0.60
        ):
            return "CHEST_FOLIANT_PICK"
        if self._seen(frame, "tpl_lvl", "REG_HUD", 0.55):
            return "RUNNING"
        if self._seen(frame, "tpl_minimap", "REG_MINIMAP", 0.55):
            return "RUNNING"
        return "UNKNOWN"

    def tap_enter(self):
        di.keyDown("enter")
        time.sleep(0.01)
        di.keyUp("enter")

    def ensure_running_fallback_enter(self, max_enters=6, screen=None, frame=None):
        if screen is None and frame is not None:
            screen = self.detect_screen(frame)
        if screen == "CHAR_SELECT":
            self.reset_enter_series()
            return False
        now = time.time()
        if self.enter_budget == 0 and self.enter_last_ts == 0.0:
            self.enter_budget = max_enters

        if self.enter_budget > 0 and (now - self.enter_last_ts) >= self.enter_cooldown:
            self.tap_enter()
            self.enter_last_ts = now
            self.enter_budget -= 1
            return True
        return False

    def reset_enter_series(self):
        self.enter_budget = 0
        self.enter_last_ts = 0.0

    def safe_click_if_found(self, found, pos, score, thr):
        if not found or score < thr:
            return False
        now = time.time()
        if (now - self.click_last_ts) < self.click_cooldown:
            return False
        click(*pos)
        self.click_last_ts = now
        return True

    def ensure_running(self, frame):
        scr = self.detect_screen(frame)

        if scr == "MAIN_MENU":
            thr = 0.75
            found, (cx, cy), score = self._find(frame, "tpl_play", "REG_MAIN_PLAY", thr)
            if self.safe_click_if_found(found, (cx, cy), score, thr):
                return True

        if scr == "CHEST_WEAPON_PICK":
            self.handle_chest_weapon(frame)
            return True

        if scr == "CHEST_FOLIANT_PICK":
            self.handle_chest_foliant(frame)
            return True

        return False

    def pick_fox_and_confirm(self, frame):
        thr = 0.70
        found, (cx, cy), score = self._find(frame, "tpl_fox_face", "REG_CHAR_GRID", thr)
        if found and score >= thr and (time.time() - self.click_last_ts) >= self.click_cooldown:
            click(cx, cy, delay=0.08)
            self.click_last_ts = time.time()
        thr2 = 0.75
        found2, (cx2, cy2), score2 = self._find(frame, "tpl_confirm", "REG_CHAR_CONFIRM", thr2)
        if found2 and score2 >= thr2 and (time.time() - self.click_last_ts) >= self.click_cooldown:
            click(cx2, cy2, delay=0.2)
            self.click_last_ts = time.time()
            return True
        return False

    def handle_chest_weapon(self, frame):
        thr = 0.70
        ok, (cx, cy), score = self._find(frame, "tpl_katana", "REG_CHEST", thr)
        if self.safe_click_if_found(ok, (cx, cy), score, thr):
            return "picked_katana"

        ok, (cx, cy), score = self._find(frame, "tpl_dexec", "REG_CHEST", thr)
        if self.safe_click_if_found(ok, (cx, cy), score, thr):
            return "picked_dexec"

        return "no_pick"

    def handle_chest_foliant(self, frame):
        for name in [
            "tpl_foliant_bottom1",
            "tpl_foliant_bottom2",
            "tpl_foliant_bottom3",
            "tpl_blood_tome",
        ]:
            thr = 0.70
            ok, (cx, cy), score = self._find(frame, name, "REG_CHEST", thr)
            if self.safe_click_if_found(ok, (cx, cy), score, thr):
                return f"picked_{name}"
        return "no_pick"

    def debug_scores(self, frame, out_dir="dbg", every_s=2.0):
        Path(out_dir).mkdir(exist_ok=True)
        now = time.time()
        if not hasattr(self, "_dbg_ts"):
            self._dbg_ts = 0.0
        if now - self._dbg_ts < every_s:
            return
        self._dbg_ts = now

        cv2.imwrite(f"{out_dir}/frame_{int(now)}.png", frame)

        checks = [
            ("MAIN_PLAY", "tpl_play", "REG_MAIN_PLAY", 0.60),
            ("CHAR", "tpl_char_select_title", "REG_CHAR_SELECT", 0.60),
            ("RUN_LVL", "tpl_lvl", "REG_HUD", 0.55),
            ("RUN_MINIMAP", "tpl_minimap", "REG_MINIMAP", 0.55),
            ("DEAD", "tpl_dead", "REG_DEAD", 0.55),
        ]
        for name, tpl, reg, thr in checks:
            ok, _, sc = self._find(frame, tpl, reg, thr)
            print(f"[DBG] {name:8} ok={ok} sc={sc:.3f}")

    def _seen(self, frame, tpl_name, reg_name, thr):
        ok, _, _ = self._find(frame, tpl_name, reg_name, thr)
        return ok

    def _find(self, frame, tpl_name, reg_name, thr):
        if tpl_name not in self.t or reg_name not in self.r:
            return False, (0, 0), 0.0
        tpl = self.t[tpl_name]
        reg = self.r[reg_name]
        return find_in_region(frame, tpl, reg, threshold=thr)
