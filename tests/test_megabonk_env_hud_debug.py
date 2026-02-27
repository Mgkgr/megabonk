import threading

import pytest


def test_get_cached_hud_debug_contains_time_metadata():
    megabonk_env = pytest.importorskip("megabonk_env")
    env = megabonk_env.MegabonkEnv.__new__(megabonk_env.MegabonkEnv)
    env._hud_lock = threading.Lock()
    env._last_hud_ts = 123.45
    env._last_hud_values = {
        "time_ocr_ms": 6.7,
        "time_fail_reason": "low_conf:41.0",
        "tesseract_cmd": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    }

    debug = env._get_cached_hud_debug()

    assert debug["hud_ts"] == 123.45
    assert debug["time_ocr_ms"] == 6.7
    assert debug["time_fail_reason"] == "low_conf:41.0"
    assert debug["tesseract_cmd"] == r"C:\Program Files\Tesseract-OCR\tesseract.exe"
