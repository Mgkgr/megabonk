import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from megabonk_bot.hud import (  # noqa: E402
    DEFAULT_HUD_REGIONS,
    HUD_TIME_RECT,
    _parse_time,
    read_hud_time,
)


def test_read_hud_time_prefers_reg_hud_time(monkeypatch):
    monkeypatch.setattr("megabonk_bot.hud._get_tesseract", lambda: None)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    custom_rect = (100, 20, 60, 25)

    data = read_hud_time(frame, regions={"REG_HUD_TIME": custom_rect})

    assert data["rect"] == custom_rect
    assert data["fail_reason"] == "tesseract_missing"


def test_read_hud_time_uses_fixed_fallback_without_regions(monkeypatch):
    monkeypatch.setattr("megabonk_bot.hud._get_tesseract", lambda: None)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    data = read_hud_time(frame)

    assert data["rect"] == HUD_TIME_RECT
    assert data["fail_reason"] == "tesseract_missing"


def test_read_hud_time_uses_relative_fallback_if_fixed_out_of_frame(monkeypatch):
    monkeypatch.setattr("megabonk_bot.hud._get_tesseract", lambda: None)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    expected = (
        int(DEFAULT_HUD_REGIONS["time"][0] * 120),
        int(DEFAULT_HUD_REGIONS["time"][1] * 80),
        int(DEFAULT_HUD_REGIONS["time"][2] * 120),
        int(DEFAULT_HUD_REGIONS["time"][3] * 80),
    )

    data = read_hud_time(frame)

    assert data["rect"] == expected
    assert data["fail_reason"] == "tesseract_missing"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12:34", 754),
        ("5:07", 307),
        ("мусор", None),
    ],
)
def test_parse_time_formats(text, expected):
    assert _parse_time(text) == expected
