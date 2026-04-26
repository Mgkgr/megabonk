import types

import run_runtime_bot as runtime_bot
from megabonk_bot.runtime.overlay import handle_overlay_mouse_event


class _FakeCv2:
    EVENT_LBUTTONDOWN = 1
    EVENT_LBUTTONUP = 4
    WINDOW_NORMAL = 16

    def __init__(self):
        self.calls: list[tuple] = []

    def namedWindow(self, name, flags):
        self.calls.append(("namedWindow", name, flags))

    def resizeWindow(self, name, width, height):
        self.calls.append(("resizeWindow", name, width, height))


class _FakeUser32:
    def __init__(self):
        self.style = (
            runtime_bot.WS_CAPTION
            | runtime_bot.WS_THICKFRAME
            | runtime_bot.WS_MINIMIZE
            | runtime_bot.WS_MAXIMIZE
            | runtime_bot.WS_SYSMENU
        )
        self.exstyle = 0
        self.set_window_pos_calls: list[tuple] = []

    def FindWindowW(self, _class_name, window_name):
        return 100 if window_name == "Overlay" else 0

    def GetWindowLongW(self, _hwnd, index):
        if index == runtime_bot.GWL_STYLE:
            return self.style
        if index == runtime_bot.GWL_EXSTYLE:
            return self.exstyle
        return 0

    def SetWindowLongW(self, _hwnd, index, value):
        if index == runtime_bot.GWL_STYLE:
            self.style = value
        elif index == runtime_bot.GWL_EXSTYLE:
            self.exstyle = value
        return value

    def SetWindowPos(self, hwnd, insert_after, x, y, w, h, flags):
        self.set_window_pos_calls.append((hwnd, insert_after, x, y, w, h, flags))
        return 1

    def SetLayeredWindowAttributes(self, _hwnd, _colorref, _alpha, _flags):
        return 1


def test_handle_overlay_mouse_event_triggers_toggle_on_left_click():
    state = {
        "toggle": False,
        "panic": False,
        "rects": {
            "toggle": (10, 10, 100, 30),
            "panic": (10, 50, 100, 30),
        },
    }

    handle_overlay_mouse_event(_FakeCv2, _FakeCv2.EVENT_LBUTTONDOWN, 25, 20, state)

    assert state["toggle"] is True
    assert state["panic"] is False


def test_handle_overlay_mouse_event_triggers_panic_on_left_release():
    state = {
        "toggle": False,
        "panic": False,
        "rects": {
            "toggle": (10, 10, 100, 30),
            "panic": (10, 50, 100, 30),
        },
    }

    handle_overlay_mouse_event(_FakeCv2, _FakeCv2.EVENT_LBUTTONUP, 25, 60, state)

    assert state["toggle"] is False
    assert state["panic"] is True


def test_ensure_overlay_window_creates_resizable_window(monkeypatch):
    fake_cv2 = _FakeCv2()
    monkeypatch.setattr(runtime_bot, "cv2", fake_cv2)

    runtime_bot._ensure_overlay_window("Overlay", width=0, height=-5)

    assert fake_cv2.calls == [
        ("namedWindow", "Overlay", _FakeCv2.WINDOW_NORMAL),
        ("resizeWindow", "Overlay", 1, 1),
    ]


def test_set_overlay_borderless_rebuilds_frame_without_activation(monkeypatch):
    fake_user32 = _FakeUser32()
    monkeypatch.setattr(
        runtime_bot.ctypes,
        "windll",
        types.SimpleNamespace(user32=fake_user32),
    )

    assert runtime_bot._set_overlay_borderless("Overlay") is True

    assert fake_user32.style & runtime_bot.WS_CAPTION == 0
    assert fake_user32.style & runtime_bot.WS_THICKFRAME == 0
    assert fake_user32.style & runtime_bot.WS_MINIMIZE == 0
    assert fake_user32.style & runtime_bot.WS_MAXIMIZE == 0
    assert fake_user32.style & runtime_bot.WS_SYSMENU == 0
    assert fake_user32.set_window_pos_calls
    _hwnd, insert_after, _x, _y, _w, _h, flags = fake_user32.set_window_pos_calls[-1]
    assert insert_after == runtime_bot.HWND_TOPMOST
    assert flags & runtime_bot.SWP_FRAMECHANGED
    assert flags & runtime_bot.SWP_NOACTIVATE


def test_move_overlay_window_keeps_overlay_topmost_without_activation(monkeypatch):
    fake_user32 = _FakeUser32()
    monkeypatch.setattr(
        runtime_bot.ctypes,
        "windll",
        types.SimpleNamespace(user32=fake_user32),
    )

    assert runtime_bot._move_overlay_window(
        "Overlay",
        x=11,
        y=22,
        w=333,
        h=444,
        topmost=True,
    )

    assert fake_user32.set_window_pos_calls == [
        (
            100,
            runtime_bot.HWND_TOPMOST,
            11,
            22,
            333,
            444,
            runtime_bot.SWP_NOACTIVATE | runtime_bot.SWP_SHOWWINDOW,
        )
    ]
