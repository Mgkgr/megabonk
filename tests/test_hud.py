import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from megabonk_bot.hud import read_hud_telemetry  # noqa: E402


def test_read_hud_telemetry_has_lvl_and_kills_keys():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    telemetry = read_hud_telemetry(frame)
    for key in [
        "time",
        "hp_ratio",
        "gold",
        "lvl",
        "kills",
        "gold_fail_reason",
        "lvl_fail_reason",
        "kills_fail_reason",
    ]:
        assert key in telemetry
