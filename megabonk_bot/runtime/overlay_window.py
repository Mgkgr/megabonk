from __future__ import annotations

import ctypes
from typing import Any


HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
GWL_EXSTYLE = -20
GWL_STYLE = -16
WS_EX_LAYERED = 0x00080000
LWA_COLORKEY = 0x00000001
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZE = 0x20000000
WS_MAXIMIZE = 0x01000000
WS_SYSMENU = 0x00080000


def set_overlay_topmost(window_name: str) -> bool:
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
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
    )


def move_overlay_window(
    window_name: str,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    topmost: bool,
) -> bool:
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
            int(x),
            int(y),
            int(w),
            int(h),
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
    )


def set_overlay_borderless(window_name: str) -> bool:
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
            SWP_NOMOVE
            | SWP_NOSIZE
            | SWP_NOACTIVATE
            | SWP_FRAMECHANGED
            | SWP_SHOWWINDOW,
        )
    )


def set_overlay_colorkey_transparent(window_name: str, colorkey=(0, 0, 0)) -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    user32 = getattr(ctypes.windll, "user32", None)
    if user32 is None:
        return False
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return False
    exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    exstyle |= WS_EX_LAYERED
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, exstyle)
    r, g, b = colorkey
    colorref = int(r) | (int(g) << 8) | (int(b) << 16)
    return bool(user32.SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY))


def sync_overlay_to_game_window(
    window_name: str,
    bbox: dict[str, Any],
    *,
    topmost: bool,
) -> bool:
    return move_overlay_window(
        window_name,
        x=int(bbox.get("left", 0)),
        y=int(bbox.get("top", 0)),
        w=max(1, int(bbox.get("width", 1))),
        h=max(1, int(bbox.get("height", 1))),
        topmost=topmost,
    )


def ensure_overlay_window(cv2, window_name: str, *, width: int, height: int) -> None:
    if cv2 is None:
        return
    cv2.namedWindow(window_name, getattr(cv2, "WINDOW_NORMAL", 0))
    resize_overlay_window(cv2, window_name, width=width, height=height)


def resize_overlay_window(cv2, window_name: str, *, width: int, height: int) -> None:
    if cv2 is None:
        return
    cv2.resizeWindow(window_name, max(1, int(width)), max(1, int(height)))
