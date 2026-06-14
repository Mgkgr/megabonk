import types

import numpy as np

import run_runtime_bot as runtime_bot
from megabonk_bot.runtime.overlay import draw_runtime_overlay, handle_overlay_mouse_event
from megabonk_bot.runtime import overlay_window


class _FakeCv2:
    EVENT_LBUTTONDOWN = 1
    EVENT_LBUTTONUP = 4
    WINDOW_NORMAL = 16
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self):
        self.calls: list[tuple] = []
        self.rectangles: list[tuple] = []
        self.texts: list[tuple[str, tuple[int, int]]] = []

    def namedWindow(self, name, flags):
        self.calls.append(("namedWindow", name, flags))

    def resizeWindow(self, name, width, height):
        self.calls.append(("resizeWindow", name, width, height))

    def rectangle(self, _image, pt1, pt2, color, thickness):
        self.rectangles.append((pt1, pt2, color, thickness))

    def putText(self, _image, text, org, _font_face, _font_scale, _color, _thickness, _line_type):
        self.texts.append((text, org))


class _FakeUser32:
    def __init__(self):
        self.style = (
            overlay_window.WS_CAPTION
            | overlay_window.WS_THICKFRAME
            | overlay_window.WS_MINIMIZE
            | overlay_window.WS_MAXIMIZE
            | overlay_window.WS_SYSMENU
        )
        self.exstyle = 0
        self.set_window_pos_calls: list[tuple] = []

    def FindWindowW(self, _class_name, window_name):
        return 100 if window_name == "Overlay" else 0

    def GetWindowLongW(self, _hwnd, index):
        if index == overlay_window.GWL_STYLE:
            return self.style
        if index == overlay_window.GWL_EXSTYLE:
            return self.exstyle
        return 0

    def SetWindowLongW(self, _hwnd, index, value):
        if index == overlay_window.GWL_STYLE:
            self.style = value
        elif index == overlay_window.GWL_EXSTYLE:
            self.exstyle = value
        return value

    def SetWindowPos(self, hwnd, insert_after, x, y, w, h, flags):
        self.set_window_pos_calls.append((hwnd, insert_after, x, y, w, h, flags))
        return 1

    def SetLayeredWindowAttributes(self, _hwnd, _colorref, _alpha, _flags):
        return 1


def test_handle_overlay_mouse_event_triggers_toggle_once_on_full_left_click():
    state = {
        "toggle": False,
        "panic": False,
        "rects": {
            "toggle": (10, 10, 100, 30),
            "panic": (10, 50, 100, 30),
        },
    }

    handle_overlay_mouse_event(_FakeCv2, _FakeCv2.EVENT_LBUTTONDOWN, 25, 20, state)

    assert state["toggle"] is False
    assert state["panic"] is False

    handle_overlay_mouse_event(_FakeCv2, _FakeCv2.EVENT_LBUTTONUP, 25, 20, state)

    assert state["toggle"] is True
    assert state["panic"] is False


def test_handle_overlay_mouse_event_triggers_panic_once_on_full_left_click():
    state = {
        "toggle": False,
        "panic": False,
        "rects": {
            "toggle": (10, 10, 100, 30),
            "panic": (10, 50, 100, 30),
        },
    }

    handle_overlay_mouse_event(_FakeCv2, _FakeCv2.EVENT_LBUTTONDOWN, 25, 60, state)

    assert state["toggle"] is False
    assert state["panic"] is False

    handle_overlay_mouse_event(_FakeCv2, _FakeCv2.EVENT_LBUTTONUP, 25, 60, state)

    assert state["toggle"] is False
    assert state["panic"] is True


def test_handle_overlay_mouse_event_ignores_release_on_different_button():
    state = {
        "toggle": False,
        "panic": False,
        "rects": {
            "toggle": (10, 10, 100, 30),
            "panic": (10, 50, 100, 30),
        },
    }

    handle_overlay_mouse_event(_FakeCv2, _FakeCv2.EVENT_LBUTTONDOWN, 25, 20, state)
    handle_overlay_mouse_event(_FakeCv2, _FakeCv2.EVENT_LBUTTONUP, 25, 60, state)

    assert state["toggle"] is False
    assert state["panic"] is False


def test_draw_runtime_overlay_moves_debug_panel_below_game_hud_and_omits_noisy_lines(monkeypatch):
    import megabonk_bot.recognition as recognition

    def _draw_recognition_overlay(frame, _analysis, **_kwargs):
        return frame.copy()

    monkeypatch.setattr(recognition, "draw_recognition_overlay", _draw_recognition_overlay)
    fake_cv2 = _FakeCv2()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    snapshot = types.SimpleNamespace(
        hp_ratio=0.75,
        lvl=4,
        kills=12,
        time_s=203,
        enemies=[object(), object()],
        enemy_classes=["enemy"],
        obstacles=[object()],
        local_map=None,
        map_state=types.SimpleNamespace(
            map_open=False,
            minimap_visible=True,
            scene_id="GeneratedMap",
            objective="ui_ocr",
            is_crypt=False,
        ),
        player_pose=types.SimpleNamespace(
            map_norm=(0.45, 0.50),
            world_pos=(1.0, 2.0, 3.0),
        ),
        detection_sources={"scene_profile": "GeneratedMap", "enemies": "screen_cv"},
        memory_probe_status="ready",
        is_dead=False,
        is_upgrade=False,
        safe_sector="right",
    )
    navigation_context = types.SimpleNamespace(
        terrain_kind="unknown",
        escape_lane=None,
        nav_confidence=0.05,
        drop_risk=0.01,
        jump_gate="not_evaluated",
        slide_gate="not_evaluated",
        slope_source="none",
        slope_delta_z=None,
    )

    draw_runtime_overlay(
        fake_cv2,
        frame,
        {"enemies": [], "enemy_classes": [], "local_map": None},
        snapshot,
        mode=types.SimpleNamespace(value="OFF"),
        action_reason="testing",
        hud_values={"gold": 1, "time_ocr_ms": 0.5, "kills_ocr_ms": 1.0},
        hud_regions={},
        navigation_context=navigation_context,
    )

    debug_texts = [
        (text, org)
        for text, org in fake_cv2.texts
        if not text.startswith(("START", "STOP", "PANIC"))
    ]
    assert debug_texts[0] == ("mode=OFF", (12, 164))
    blocked_fragments = ("nav terrain", "nav jump", "scene=", "probe=", "controls:", "surface")
    debug_text = "\n".join(text for text, _org in debug_texts).lower()
    for fragment in blocked_fragments:
        assert fragment not in debug_text

    diagnostic_panels = [rect for rect in fake_cv2.rectangles if rect[3] == 2]
    assert diagnostic_panels[-1][0][1] >= 148


def test_ensure_overlay_window_creates_resizable_window(monkeypatch):
    fake_cv2 = _FakeCv2()
    overlay_window.ensure_overlay_window(fake_cv2, "Overlay", width=0, height=-5)

    assert fake_cv2.calls == [
        ("namedWindow", "Overlay", _FakeCv2.WINDOW_NORMAL),
        ("resizeWindow", "Overlay", 1, 1),
    ]


def test_set_overlay_borderless_rebuilds_frame_without_activation(monkeypatch):
    fake_user32 = _FakeUser32()
    monkeypatch.setattr(
        overlay_window.ctypes,
        "windll",
        types.SimpleNamespace(user32=fake_user32),
    )

    assert overlay_window.set_overlay_borderless("Overlay") is True

    assert fake_user32.style & overlay_window.WS_CAPTION == 0
    assert fake_user32.style & overlay_window.WS_THICKFRAME == 0
    assert fake_user32.style & overlay_window.WS_MINIMIZE == 0
    assert fake_user32.style & overlay_window.WS_MAXIMIZE == 0
    assert fake_user32.style & overlay_window.WS_SYSMENU == 0
    assert fake_user32.set_window_pos_calls
    _hwnd, insert_after, _x, _y, _w, _h, flags = fake_user32.set_window_pos_calls[-1]
    assert insert_after == overlay_window.HWND_TOPMOST
    assert flags & overlay_window.SWP_FRAMECHANGED
    assert flags & overlay_window.SWP_NOACTIVATE


def test_move_overlay_window_keeps_overlay_topmost_without_activation(monkeypatch):
    fake_user32 = _FakeUser32()
    monkeypatch.setattr(
        overlay_window.ctypes,
        "windll",
        types.SimpleNamespace(user32=fake_user32),
    )

    assert overlay_window.move_overlay_window(
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
            overlay_window.HWND_TOPMOST,
            11,
            22,
            333,
            444,
            overlay_window.SWP_NOACTIVATE | overlay_window.SWP_SHOWWINDOW,
        )
    ]
