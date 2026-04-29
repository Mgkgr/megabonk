import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from megabonk_bot.recognition import _draw_label


def _label_crop(frame, origin, text, *, scale=1.0, thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    return frame[y : y + th + baseline + 4, x : x + tw + 6]


def test_draw_label_uses_dark_text_on_bright_background():
    frame = np.zeros((80, 220, 3), dtype=np.uint8)
    origin = (4, 4)

    _draw_label(frame, origin, "hud_roi", (0, 255, 255), scale=1.0, thickness=2)

    crop = _label_crop(frame, origin, "hud_roi")
    assert np.any(np.all(crop <= 40, axis=2))


def test_draw_label_keeps_bright_text_on_dark_background():
    frame = np.zeros((80, 220, 3), dtype=np.uint8)
    origin = (4, 4)

    _draw_label(frame, origin, "enemy", (0, 0, 255), scale=1.0, thickness=2)

    crop = _label_crop(frame, origin, "enemy")
    assert np.any(np.all(crop >= 220, axis=2))
