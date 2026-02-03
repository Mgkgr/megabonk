import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

import mss
import numpy as np


user32 = ctypes.windll.user32
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL

user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL

user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int

user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int

user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.GetClientRect.restype = wintypes.BOOL

user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
user32.ClientToScreen.restype = wintypes.BOOL

user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL

user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL

user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint,
]
user32.SetWindowPos.restype = wintypes.BOOL


SW_RESTORE = 9
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040


def find_hwnd_by_title_substr(title_substr: str) -> int | None:
    target = title_substr.lower().strip()
    found = {"hwnd": None}

    def enum_proc(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        if target in title.lower():
            found["hwnd"] = hwnd
            return False
        return True

    cb = WNDENUMPROC(enum_proc)
    user32.EnumWindows(cb, 0)
    return found["hwnd"]


def get_client_bbox_on_screen(hwnd: int) -> dict:
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetClientRect failed")

    pt = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        raise RuntimeError("ClientToScreen failed")

    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    return {
        "left": int(pt.x),
        "top": int(pt.y),
        "width": width,
        "height": height,
    }


@dataclass
class WindowCapture:
    window_title: str
    hwnd: int
    sct: mss.mss
    _last_grab_ts: float = 0.0
    _last_grab_hz: float = 0.0
    _bad_grab_count: int = 0

    @classmethod
    def create(cls, window_title: str = "MEGABONK"):
        hwnd = find_hwnd_by_title_substr(window_title)
        if not hwnd:
            raise RuntimeError(
                f"Window not found by title substring: {window_title!r}"
            )
        return cls(window_title=window_title, hwnd=hwnd, sct=mss.mss())

    def focus(self, topmost: bool = True):
        user32.ShowWindow(self.hwnd, SW_RESTORE)
        user32.SetWindowPos(
            self.hwnd,
            HWND_TOPMOST if topmost else HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        )
        user32.SetForegroundWindow(self.hwnd)

    def get_bbox(self) -> dict:
        return get_client_bbox_on_screen(self.hwnd)

    def grab(self):
        bbox = self.get_bbox()
        expected_h = int(bbox["height"])
        expected_w = int(bbox["width"])
        attempts = 3
        base_backoff_s = 0.05

        now = time.time()
        if self._last_grab_ts > 0:
            dt = now - self._last_grab_ts
            if dt > 0:
                self._last_grab_hz = 1.0 / dt
        self._last_grab_ts = now

        for attempt in range(attempts):
            try:
                img = np.array(self.sct.grab(bbox), dtype=np.uint8)
                if img.size == 0:
                    raise ValueError("Empty frame grab")
                frame = img[:, :, :3]
                if frame.shape[0] != expected_h or frame.shape[1] != expected_w:
                    raise ValueError(
                        f"Unexpected frame size: {frame.shape} "
                        f"expected ({expected_h}, {expected_w}, 3)"
                    )
                return frame
            except Exception:
                self._bad_grab_count += 1
                self.focus(topmost=True)
                time.sleep(base_backoff_s * (2 ** attempt))
                bbox = self.get_bbox()
                expected_h = int(bbox["height"])
                expected_w = int(bbox["width"])
        return None

    def debug_print(self):
        bbox = self.get_bbox()
        print(f"[CAP] bbox={bbox}")
        frame = self.grab()
        print(f"[CAP] frame.shape={frame.shape} (H,W,C)")
        return frame


def get_window_region(title_contains: str):
    hwnd = find_hwnd_by_title_substr(title_contains)
    if not hwnd:
        raise RuntimeError(
            f"Window not found by title substring: {title_contains!r}"
        )
    bbox = get_client_bbox_on_screen(hwnd)
    bbox["hwnd"] = hwnd
    return bbox
