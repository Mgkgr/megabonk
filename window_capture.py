import ctypes
from ctypes import wintypes

import win32gui


user32 = ctypes.WinDLL("user32", use_last_error=True)


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def _get_client_rect_screen(hwnd: int):
    rect = RECT()
    if not user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())

    pt = wintypes.POINT(rect.left, rect.top)
    if not user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(pt)):
        raise ctypes.WinError(ctypes.get_last_error())

    left = pt.x
    top = pt.y
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    return left, top, width, height

def get_window_region(title_contains: str):
    hwnd = win32gui.FindWindow(None, title_contains)
    if hwnd == 0:
        matches = []

        def enum_cb(h, acc):
            title = win32gui.GetWindowText(h)
            if title_contains.lower() in title.lower():
                acc.append(h)

        win32gui.EnumWindows(enum_cb, matches)
        if not matches:
            raise RuntimeError(f"Не нашёл окно с '{title_contains}'. Проверь заголовок окна игры.")
        hwnd = matches[0]

    left, top, width, height = _get_client_rect_screen(hwnd)
    return dict(left=left, top=top, width=width, height=height)
