from __future__ import annotations

import time

MOVE_MAPPING = {
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

RELEASE_KEYS = [
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
]


def key_on(di, key: str) -> None:
    di.keyDown(key)


def key_off(di, key: str) -> None:
    di.keyUp(key)


def tap(di, key: str, dt: float = 0.01) -> None:
    di.keyDown(key)
    time.sleep(max(0.0, float(dt)))
    di.keyUp(key)


def hold(di, key: str, dt: float = 0.5) -> None:
    di.keyDown(key)
    time.sleep(max(0.0, float(dt)))
    di.keyUp(key)


def set_move(di, dir_id: int) -> None:
    want = set(MOVE_MAPPING.get(int(dir_id), []))
    for key in ("w", "a", "s", "d"):
        (key_on if key in want else key_off)(di, key)


def release_all_keys(di) -> None:
    for key in RELEASE_KEYS:
        try:
            di.keyUp(key)
        except Exception:
            pass


def apply_cam_yaw(di, yaw_id: int, cam_yaw_pixels: int) -> None:
    mapping = {0: -int(cam_yaw_pixels), 1: 0, 2: int(cam_yaw_pixels)}
    dx = mapping.get(int(yaw_id), 0)
    if dx != 0:
        di.moveRel(dx, 0, duration=0)
