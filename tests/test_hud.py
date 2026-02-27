import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from megabonk_bot import hud  # noqa: E402
from megabonk_bot.hud import read_hud_telemetry  # noqa: E402


def test_read_hud_telemetry_has_lvl_and_kills_keys():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    telemetry = read_hud_telemetry(frame)
    for key in [
        "time",
        "time_fail_reason",
        "time_ocr_ms",
        "time_rect",
        "hp_ratio",
        "hp_fail_reason",
        "gold",
        "lvl",
        "kills",
        "gold_fail_reason",
        "gold_ocr_ms",
        "gold_rect",
        "lvl_fail_reason",
        "lvl_ocr_ms",
        "lvl_rect",
        "kills_fail_reason",
        "kills_ocr_ms",
        "kills_rect",
        "tesseract_cmd",
    ]:
        assert key in telemetry


def test_read_hud_telemetry_passes_regions_to_time(monkeypatch):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    regions = {"REG_HUD_TIME": (101, 20, 40, 20)}
    seen = {"regions": None}

    def _fake_time(_frame, regions=None, min_conf=45.0):
        seen["regions"] = regions
        return {
            "time": None,
            "fail_reason": "tesseract_missing",
            "ocr_ms": 0.0,
            "rect": (0, 0, 1, 1),
            "tesseract_cmd": None,
        }

    monkeypatch.setattr(hud, "read_hud_time", _fake_time)

    read_hud_telemetry(frame, regions=regions)

    assert seen["regions"] is regions
