import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

import mss
import numpy as np


user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
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


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL

user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL

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

user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND

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

user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC

user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int

user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC

gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL

gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HANDLE

gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
gdi32.SelectObject.restype = wintypes.HANDLE

gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
gdi32.DeleteObject.restype = wintypes.BOOL

gdi32.GetDIBits.argtypes = [
    wintypes.HDC,
    wintypes.HANDLE,
    wintypes.UINT,
    wintypes.UINT,
    ctypes.c_void_p,
    ctypes.POINTER(BITMAPINFO),
    wintypes.UINT,
]
gdi32.GetDIBits.restype = ctypes.c_int


SW_RESTORE = 9
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002
BI_RGB = 0
DIB_RGB_COLORS = 0


def _coerce_capture_backend(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in {"auto", "printwindow", "mss"}:
        raise ValueError(
            f"Unsupported capture backend: {value!r}. Expected auto|printwindow|mss."
        )
    return normalized


def _grab_with_printwindow(hwnd: int, width: int, height: int):
    if width <= 0 or height <= 0:
        return None

    hdc_window = None
    hdc_mem = None
    hbmp = None
    old_obj = None
    try:
        hdc_window = user32.GetDC(hwnd)
        if not hdc_window:
            return None
        hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
        if not hdc_mem:
            return None
        hbmp = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
        if not hbmp:
            return None
        old_obj = gdi32.SelectObject(hdc_mem, hbmp)
        if not old_obj:
            return None

        flags = PW_CLIENTONLY | PW_RENDERFULLCONTENT
        ok = bool(user32.PrintWindow(hwnd, hdc_mem, flags))
        if not ok:
            ok = bool(user32.PrintWindow(hwnd, hdc_mem, PW_CLIENTONLY))
        if not ok:
            return None

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        # Negative height = top-down bitmap, so no vertical flip is required.
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        buf = (ctypes.c_ubyte * (width * height * 4))()
        lines = gdi32.GetDIBits(
            hdc_mem,
            hbmp,
            0,
            height,
            ctypes.cast(buf, ctypes.c_void_p),
            ctypes.byref(bmi),
            DIB_RGB_COLORS,
        )
        if lines != height:
            return None
        bgra = np.ctypeslib.as_array(buf).reshape((height, width, 4))
        return bgra[:, :, :3].copy()
    finally:
        if hdc_mem and old_obj:
            gdi32.SelectObject(hdc_mem, old_obj)
        if hbmp:
            gdi32.DeleteObject(hbmp)
        if hdc_mem:
            gdi32.DeleteDC(hdc_mem)
        if hdc_window:
            user32.ReleaseDC(hwnd, hdc_window)


def find_hwnd_by_title_substr(title_substr: str) -> int | None:
    target = title_substr.lower().strip()
    found = {"hwnd": None, "title": None}
    substring_matches: list[tuple[int, str, int]] = []

    def enum_proc(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        title_lower = title.lower()
        if target == title_lower:
            found["hwnd"] = hwnd
            found["title"] = title
            return False
        if target in title_lower:
            substring_matches.append((len(title_lower), title, hwnd))
        return True

    cb = WNDENUMPROC(enum_proc)
    user32.EnumWindows(cb, 0)
    if found["hwnd"]:
        return found["hwnd"]
    if substring_matches:
        substring_matches.sort(key=lambda item: item[0])
        return substring_matches[0][2]
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
    capture_backend: str = "auto"
    _last_grab_ts: float = 0.0
    _last_grab_hz: float = 0.0
    _bad_grab_count: int = 0
    _last_focus_ts: float = 0.0

    @classmethod
    def create(cls, window_title: str = "MEGABONK", capture_backend: str = "auto"):
        hwnd = find_hwnd_by_title_substr(window_title)
        if not hwnd:
            raise RuntimeError(
                f"Window not found by title substring: {window_title!r}"
            )
        return cls(
            window_title=window_title,
            hwnd=hwnd,
            sct=mss.mss(),
            capture_backend=_coerce_capture_backend(capture_backend),
        )

    def _refresh_hwnd(self):
        if self.hwnd and bool(user32.IsWindow(self.hwnd)):
            return
        hwnd = find_hwnd_by_title_substr(self.window_title)
        if not hwnd:
            raise RuntimeError(
                f"Window not found by title substring: {self.window_title!r}"
            )
        self.hwnd = hwnd

    def focus(self, topmost: bool = True):
        self._refresh_hwnd()
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
        self._last_focus_ts = time.time()

    def focus_if_needed(self, topmost: bool = False, min_interval_s: float = 0.25):
        now = time.time()
        if (now - self._last_focus_ts) < max(0.0, float(min_interval_s)):
            return
        self._refresh_hwnd()
        if user32.GetForegroundWindow() != self.hwnd:
            self.focus(topmost=topmost)

    def get_bbox(self) -> dict:
        self._refresh_hwnd()
        return get_client_bbox_on_screen(self.hwnd)

    def client_to_screen(self, x: int, y: int, bbox: dict | None = None):
        box = bbox or self.get_bbox()
        return int(box["left"]) + int(x), int(box["top"]) + int(y)

    def _grab_with_mss(self, bbox: dict):
        img = np.array(self.sct.grab(bbox), dtype=np.uint8)
        if img.size == 0:
            raise ValueError("Empty frame grab")
        return img[:, :, :3]

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
                frame = None
                if self.capture_backend in {"auto", "printwindow"}:
                    frame = _grab_with_printwindow(self.hwnd, expected_w, expected_h)
                if frame is None and self.capture_backend in {"auto", "mss"}:
                    frame = self._grab_with_mss(bbox)
                if frame is None:
                    raise ValueError("Capture backend returned empty frame")
                if frame.shape[0] != expected_h or frame.shape[1] != expected_w:
                    raise ValueError(
                        f"Unexpected frame size: {frame.shape} "
                        f"expected ({expected_h}, {expected_w}, 3)"
                    )
                return frame
            except Exception:
                self._bad_grab_count += 1
                self.focus_if_needed(topmost=False, min_interval_s=0.05)
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
