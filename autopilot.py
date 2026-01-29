import time
import pydirectinput as di

from vision import find_in_region


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

    def detect_screen(self, frame):
        if self._seen(frame, "tpl_char_select_title", "REG_CHAR_SELECT", 0.75):
            return "CHAR_SELECT"
        if self._seen(frame, "tpl_unlocks_title", "REG_UNLOCKS", 0.75):
            return "UNLOCKS_WEAPONS"
        if self._seen(frame, "tpl_play", "REG_MAIN_PLAY", 0.75):
            return "MAIN_MENU"
        if self._seen(frame, "tpl_katana", "REG_CHEST", 0.70) or self._seen(
            frame, "tpl_dexec", "REG_CHEST", 0.70
        ):
            return "CHEST_WEAPON_PICK"
        if self._seen(frame, "tpl_blood_tome", "REG_CHEST", 0.70) or self._seen(
            frame, "tpl_foliant_bottom1", "REG_CHEST", 0.70
        ):
            return "CHEST_FOLIANT_PICK"
        if self._seen(frame, "tpl_hud", "REG_HUD", 0.75):
            return "RUNNING"
        return "UNKNOWN"

    def ensure_running(self, frame):
        scr = self.detect_screen(frame)

        if scr == "MAIN_MENU":
            found, (cx, cy), _ = self._find(frame, "tpl_play", "REG_MAIN_PLAY", 0.75)
            if found:
                click(cx, cy)
                return True

        if scr == "CHAR_SELECT":
            found, (cx, cy), _ = self._find(frame, "tpl_fox_face", "REG_CHAR_GRID", 0.70)
            if found:
                click(cx, cy, delay=0.08)
            found2, (cx2, cy2), _ = self._find(frame, "tpl_confirm", "REG_CHAR_CONFIRM", 0.75)
            if found2:
                click(cx2, cy2, delay=0.2)
                return True

        if scr == "CHEST_WEAPON_PICK":
            self.handle_chest_weapon(frame)
            return True

        if scr == "CHEST_FOLIANT_PICK":
            self.handle_chest_foliant(frame)
            return True

        if scr == "UNKNOWN":
            tap("esc", delay=0.1)
            return True

        return False

    def handle_chest_weapon(self, frame):
        ok, (cx, cy), _ = self._find(frame, "tpl_katana", "REG_CHEST", 0.70)
        if ok:
            click(cx, cy)
            return "picked_katana"

        ok, (cx, cy), _ = self._find(frame, "tpl_dexec", "REG_CHEST", 0.70)
        if ok:
            click(cx, cy)
            return "picked_dexec"

        return "no_pick"

    def handle_chest_foliant(self, frame):
        for name in [
            "tpl_foliant_bottom1",
            "tpl_foliant_bottom2",
            "tpl_foliant_bottom3",
            "tpl_blood_tome",
        ]:
            ok, (cx, cy), _ = self._find(frame, name, "REG_CHEST", 0.70)
            if ok:
                click(cx, cy)
                return f"picked_{name}"
        return "no_pick"

    def _seen(self, frame, tpl_name, reg_name, thr):
        ok, _, _ = self._find(frame, tpl_name, reg_name, thr)
        return ok

    def _find(self, frame, tpl_name, reg_name, thr):
        if tpl_name not in self.t or reg_name not in self.r:
            return False, (0, 0), 0.0
        tpl = self.t[tpl_name]
        reg = self.r[reg_name]
        return find_in_region(frame, tpl, reg, threshold=thr)
