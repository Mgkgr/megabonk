from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from megabonk_bot.vision import find_in_region


def make_window_click(
    cap,
    *,
    input_driver,
    focus_interval_s: float,
    sleep_fn: Callable[[float], None] = time.sleep,
):
    def _click(client_x: int, client_y: int, delay: float = 0.0) -> None:
        cap.focus_if_needed(topmost=False, min_interval_s=focus_interval_s)
        bbox = cap.get_bbox()
        x, y = cap.client_to_screen(int(client_x), int(client_y), bbox=bbox)
        input_driver.moveTo(x, y)
        input_driver.click()
        if delay > 0:
            sleep_fn(delay)

    return _click


def try_click_template(
    frame,
    templates: Mapping[str, Any],
    regions: Mapping[str, Any],
    tpl_name: str,
    region_name: str,
    threshold: float,
    *,
    click_fn: Callable[[int, int, float], None] | None = None,
    finder: Callable[..., tuple[bool, tuple[int, int], float]] = find_in_region,
) -> bool:
    tpl = templates.get(tpl_name)
    region = regions.get(region_name)
    if tpl is None or region is None:
        return False
    found, (cx, cy), score = finder(frame, tpl, region, threshold=threshold)
    if not found:
        return False
    if click_fn is not None:
        click_fn(cx, cy, 0.0)
    return bool(score >= threshold)
