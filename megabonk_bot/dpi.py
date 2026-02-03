import ctypes


def enable_dpi_awareness() -> None:
    """Вызывать как можно раньше (до cv2/mss/window_capture), чтобы избежать DPI-артефактов."""
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
