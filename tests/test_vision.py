import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from megabonk_bot.vision import find_in_region  # noqa: E402


def test_find_in_region_finds_simple_patch():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[50:70, 80:100] = 255

    tpl = np.zeros((20, 20, 3), dtype=np.uint8)
    tpl[:] = 255

    region = (0, 0, 300, 200)
    found, (cx, cy), score = find_in_region(frame, tpl, region, threshold=0.80)

    assert found is True
    assert 80 <= cx <= 100
    assert 50 <= cy <= 70
    assert score >= 0.80


def test_find_in_region_respects_threshold():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    tpl = np.ones((20, 20, 3), dtype=np.uint8) * 255
    region = (0, 0, 300, 200)

    found, _, score = find_in_region(frame, tpl, region, threshold=0.99)
    assert found is False
