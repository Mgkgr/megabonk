import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from megabonk_bot.hud import resolve_hud_debug_rects  # noqa: E402


def test_resolve_hud_debug_rects_includes_expected_keys():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rects = resolve_hud_debug_rects(frame)
    for key in ("time", "kills", "lvl", "gold", "hp"):
        assert key in rects


def test_resolve_hud_debug_rects_prefers_custom_time_region():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    custom_time = (100, 30, 70, 20)
    rects = resolve_hud_debug_rects(frame, regions={"REG_HUD_TIME": custom_time})
    assert rects["time"] == custom_time


def test_resolve_hud_debug_rects_empty_frame_returns_empty_dict():
    rects = resolve_hud_debug_rects(None)
    assert rects == {}
