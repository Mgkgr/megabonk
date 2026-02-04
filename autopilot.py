import time
from pathlib import Path

import cv2
import numpy as np
import pydirectinput as di

from megabonk_bot.vision import find_in_region

di.PAUSE = 0.0
di.FAILSAFE = False

AUTOPILOT_SCENARIOS = {
    "DEAD": {
        "templates": ["tpl_dead", "tpl_confirm"],
        "regions": ["REG_DEAD", "REG_DEAD_CONFIRM"],
    },
    "CHAR_SELECT": {
        "templates": ["tpl_char_select_title", "tpl_fox_face", "tpl_confirm"],
        "regions": ["REG_CHAR_SELECT", "REG_CHAR_GRID", "REG_CHAR_CONFIRM"],
    },
    "MAIN_MENU": {
        "templates": ["tpl_play"],
        "regions": ["REG_MAIN_PLAY"],
    },
    "UNLOCKS_WEAPONS": {
        "templates": ["tpl_unlocks_title"],
        "regions": ["REG_UNLOCKS"],
    },
    "CHEST_WEAPON_PICK": {
        "templates": ["tpl_katana", "tpl_dexec"],
        "regions": ["REG_CHEST"],
    },
    "CHEST_FOLIANT_PICK": {
        "templates": [
            "tpl_foliant_bottom1",
            "tpl_foliant_bottom2",
            "tpl_foliant_bottom3",
            "tpl_blood_tome",
        ],
        "regions": ["REG_CHEST"],
    },
    "RUNNING": {
        "templates": ["tpl_lvl", "tpl_minimap"],
        "regions": ["REG_HUD", "REG_MINIMAP"],
    },
}

REQUIRED_TEMPLATES = sorted(
    {tpl for scenario in AUTOPILOT_SCENARIOS.values() for tpl in scenario["templates"]}
)


def click(x, y, delay=0.05):
    di.moveTo(x, y)
    di.click()
    time.sleep(delay)


def tap(key, delay=0.05):
    di.keyDown(key)
    time.sleep(0.01)
    di.keyUp(key)
    time.sleep(delay)


def is_death_like_frame(frame, mean_thr=35.0, std_thr=12.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    return mean < mean_thr and std < std_thr


class AutoPilot:
    def __init__(self, templates, regions):
        self.t = templates
        self.r = regions
        self.missing_templates = [tpl for tpl in REQUIRED_TEMPLATES if tpl not in templates]
        self.click_last_ts = 0.0
        self.click_cooldown = 0.5

    def detect_screen(self, frame):
        if self._seen(frame, "tpl_dead", "REG_DEAD", 0.55):
            return "DEAD"
        if self._seen(frame, "tpl_dead", "REG_DEAD", 0.35) and is_death_like_frame(frame):
            return "DEAD"
        if self._seen(frame, "tpl_confirm", "REG_DEAD_CONFIRM", 0.50):
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
            thr = 0.65
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
        thr = 0.65
        found, (cx, cy), score = self._find(frame, "tpl_fox_face", "REG_CHAR_GRID", thr)
        if found and score >= thr and (time.time() - self.click_last_ts) >= self.click_cooldown:
            click(cx, cy, delay=0.08)
            self.click_last_ts = time.time()
        thr2 = 0.70
        found2, (cx2, cy2), score2 = self._find(frame, "tpl_confirm", "REG_CHAR_CONFIRM", thr2)
        if found2 and score2 >= thr2 and (time.time() - self.click_last_ts) >= self.click_cooldown:
            click(cx2, cy2, delay=0.2)
            self.click_last_ts = time.time()
            return True
        return False

    def handle_chest_weapon(self, frame):
        thr = 0.65
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
            thr = 0.65
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
        death_like = is_death_like_frame(frame)
        print(f"[DBG] DEATH_LIKE ok={death_like}")

    def debug_hud(self, hud_values):
        if not hud_values:
            print("[DBG] HUD hp=None gold=None time=None")
            return
        hp = hud_values.get("hp")
        gold = hud_values.get("gold")
        time_val = hud_values.get("time")
        print(f"[DBG] HUD hp={hp} gold={gold} time={time_val}")

    def _seen(self, frame, tpl_name, reg_name, thr):
        ok, _, _ = self._find(frame, tpl_name, reg_name, thr)
        return ok

    def _find(self, frame, tpl_name, reg_name, thr):
        if tpl_name not in self.t or reg_name not in self.r:
            return False, (0, 0), 0.0
        tpl = self.t[tpl_name]
        reg = self.r[reg_name]
        return find_in_region(frame, tpl, reg, threshold=thr)


class HeuristicAutoPilot:
    def __init__(
        self,
        enemy_hsv_lower=(45, 80, 40),
        enemy_hsv_upper=(85, 255, 255),
        coin_hsv_lower=(18, 120, 80),
        coin_hsv_upper=(35, 255, 255),
        enemy_area_threshold=1400,
        coin_area_threshold=900,
        center_roi=(0.35, 0.38, 0.30, 0.36),
        center_lower_roi=(0.35, 0.55, 0.30, 0.35),
        stuck_diff_threshold=3.0,
        scan_period=12,
    ):
        self.enemy_hsv_lower = np.array(enemy_hsv_lower, dtype=np.uint8)
        self.enemy_hsv_upper = np.array(enemy_hsv_upper, dtype=np.uint8)
        self.coin_hsv_lower = np.array(coin_hsv_lower, dtype=np.uint8)
        self.coin_hsv_upper = np.array(coin_hsv_upper, dtype=np.uint8)
        self.enemy_area_threshold = float(enemy_area_threshold)
        self.coin_area_threshold = float(coin_area_threshold)
        self.center_roi = center_roi
        self.center_lower_roi = center_lower_roi
        self.stuck_diff_threshold = float(stuck_diff_threshold)
        self.scan_period = int(scan_period)
        self._scan_dir = -1
        self._scan_ticks = 0
        self._last_gray = None
        self._last_forward = False
        self._stuck_toggle = False

    def act(self, frame, include_cam_yaw=True):
        h, w = frame.shape[:2]
        danger_center, danger_area = self._find_center_in_roi(
            frame, self.enemy_hsv_lower, self.enemy_hsv_upper, self.center_lower_roi
        )
        loot_center, loot_area = self._find_center_in_roi(
            frame, self.coin_hsv_lower, self.coin_hsv_upper, self.center_roi
        )
        danger = danger_area >= self.enemy_area_threshold
        loot = loot_area >= self.coin_area_threshold

        center_x = w * 0.5
        yaw = 1
        dir_id = 1
        jump = 0
        slide = 0
        reason = "cruise"

        if danger and danger_center is not None:
            slide = 1
            if danger_center[0] < center_x:
                dir_id = 4
                yaw = 2
            else:
                dir_id = 3
                yaw = 0
            reason = "evade_enemy"
        elif loot and loot_center is not None:
            offset = loot_center[0] - center_x
            if offset < -w * 0.06:
                dir_id = 5
                yaw = 0
            elif offset > w * 0.06:
                dir_id = 6
                yaw = 2
            else:
                dir_id = 1
                yaw = 1
            reason = "seek_loot"
        else:
            yaw = 0 if self._scan_dir < 0 else 2
            self._scan_ticks += 1
            if self._scan_ticks >= self.scan_period:
                self._scan_ticks = 0
                self._scan_dir *= -1
            reason = "scan_forward"

        stuck = self._is_stuck(frame)
        if stuck and not danger:
            self._stuck_toggle = not self._stuck_toggle
            dir_id = 3 if self._stuck_toggle else 4
            yaw = 0 if dir_id == 3 else 2
            jump = 1
            reason = "unstuck"

        forwardish = dir_id in (1, 5, 6)
        self._last_forward = forwardish

        if include_cam_yaw:
            return (dir_id, yaw, jump, slide, reason)
        return (dir_id, jump, slide, reason)

    def _is_stuck(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_gray is None:
            self._last_gray = gray
            return False
        diff = cv2.absdiff(gray, self._last_gray)
        self._last_gray = gray
        if not self._last_forward:
            return False
        return float(diff.mean()) < self.stuck_diff_threshold

    def _find_center_in_roi(self, frame, lower, upper, roi_rel):
        h, w = frame.shape[:2]
        x0 = int(roi_rel[0] * w)
        y0 = int(roi_rel[1] * h)
        rw = int(roi_rel[2] * w)
        rh = int(roi_rel[3] * h)
        x1 = min(w, x0 + rw)
        y1 = min(h, y0 + rh)
        if x1 <= x0 or y1 <= y0:
            return None, 0.0

        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        if not contours:
            return None, 0.0
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            return None, 0.0
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None, area
        cx = int(moments["m10"] / moments["m00"]) + x0
        cy = int(moments["m01"] / moments["m00"]) + y0
        return (cx, cy), area
