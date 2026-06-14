from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import cv2

from megabonk_bot.vision import find_in_region

DEFAULT_RUNTIME_SCREEN_THRESHOLDS = {
    "dead_hard": 0.55,
    "dead_soft": 0.35,
    "dead_confirm": 0.50,
    "char_select_title": 0.60,
    "main_play_detect": 0.60,
    "unlocks_title": 0.60,
    "chest_weapon_detect": 0.60,
    "chest_foliant_detect": 0.60,
    "running_lvl": 0.55,
    "running_minimap": 0.55,
}


def is_death_like_frame(frame, mean_thr: float = 35.0, std_thr: float = 12.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    return mean < mean_thr and std < std_thr


class RuntimeScreenDetector:
    def __init__(
        self,
        *,
        templates: Mapping[str, Any],
        regions: Mapping[str, Any],
        template_thresholds: Mapping[str, float] | None = None,
        finder: Callable[..., tuple[bool, tuple[int, int], float]] | None = None,
        death_like_fn: Callable[..., bool] = is_death_like_frame,
    ) -> None:
        self.templates = templates
        self.regions = regions
        self.finder = finder or find_in_region
        self.death_like_fn = death_like_fn
        self.template_thresholds = dict(DEFAULT_RUNTIME_SCREEN_THRESHOLDS)
        if template_thresholds:
            self.template_thresholds.update(template_thresholds)

    def _thr(self, key: str) -> float:
        return float(
            self.template_thresholds.get(key, DEFAULT_RUNTIME_SCREEN_THRESHOLDS[key])
        )

    def _seen(self, frame, tpl_name: str, region_name: str, thr: float) -> bool:
        tpl = self.templates.get(tpl_name)
        region = self.regions.get(region_name)
        if tpl is None or region is None:
            return False
        found, _center, score = self.finder(frame, tpl, region, threshold=thr)
        return bool(found and score >= thr)

    def detect(self, frame) -> str:
        if self._seen(frame, "tpl_dead", "REG_DEAD", self._thr("dead_hard")):
            return "DEAD"
        if self._seen(frame, "tpl_dead", "REG_DEAD", self._thr("dead_soft")) and self.death_like_fn(
            frame
        ):
            return "DEAD"
        if self._seen(frame, "tpl_confirm", "REG_DEAD_CONFIRM", self._thr("dead_confirm")):
            return "DEAD"
        if self._seen(
            frame,
            "tpl_char_select_title",
            "REG_CHAR_SELECT",
            self._thr("char_select_title"),
        ):
            return "CHAR_SELECT"
        if self._seen(frame, "tpl_play", "REG_MAIN_PLAY", self._thr("main_play_detect")):
            return "MAIN_MENU"
        if self._seen(
            frame,
            "tpl_unlocks_title",
            "REG_UNLOCKS",
            self._thr("unlocks_title"),
        ):
            return "UNLOCKS_WEAPONS"
        if self._seen(frame, "tpl_katana", "REG_CHEST", self._thr("chest_weapon_detect")) or self._seen(
            frame,
            "tpl_dexec",
            "REG_CHEST",
            self._thr("chest_weapon_detect"),
        ):
            return "CHEST_WEAPON_PICK"
        if self._seen(
            frame,
            "tpl_blood_tome",
            "REG_CHEST",
            self._thr("chest_foliant_detect"),
        ) or self._seen(
            frame,
            "tpl_foliant_bottom1",
            "REG_CHEST",
            self._thr("chest_foliant_detect"),
        ):
            return "CHEST_FOLIANT_PICK"
        if self._seen(frame, "tpl_lvl", "REG_HUD", self._thr("running_lvl")):
            return "RUNNING"
        if self._seen(frame, "tpl_minimap", "REG_MINIMAP", self._thr("running_minimap")):
            return "RUNNING"
        return "UNKNOWN"
