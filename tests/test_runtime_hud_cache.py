import time

import pytest


def test_hud_cache_refresh_updates_snapshot():
    runtime_hud_cache = pytest.importorskip("megabonk_bot.runtime.hud_cache")
    cache = runtime_hud_cache.HudTelemetryCache(
        read_hud_telemetry=lambda frame, regions=None: {
            "time": frame,
            "regions_seen": regions,
            "time_ocr_ms": 4.2,
        },
        regions={"REG_HUD_TIME": (1, 2, 3, 4)},
        interval_s=0.8,
    )

    cache.refresh(321, now=12.5)
    snapshot = cache.snapshot()

    assert snapshot["time"] == 321
    assert snapshot["regions_seen"] == {"REG_HUD_TIME": (1, 2, 3, 4)}
    assert snapshot["time_ocr_ms"] == 4.2
    assert snapshot["hud_ts"] == 12.5
    assert snapshot["debug_dumped"] is False
    assert snapshot["hud_fail_streak"] == 0


def test_hud_cache_refresh_uses_updated_regions():
    runtime_hud_cache = pytest.importorskip("megabonk_bot.runtime.hud_cache")
    seen = {"regions": None}

    def _fake_read(frame, regions=None):
        seen["regions"] = regions
        return {"time": frame}

    cache = runtime_hud_cache.HudTelemetryCache(
        read_hud_telemetry=_fake_read,
        regions={"REG_HUD_TIME": (1, 2, 3, 4)},
        interval_s=0.8,
    )
    updated_regions = {"REG_HUD_TIME": (10, 20, 30, 40)}

    cache.set_regions(updated_regions)
    cache.refresh(111, now=1.0)

    assert seen["regions"] is updated_regions


def test_hud_cache_background_poll_uses_latest_submitted_frame():
    runtime_hud_cache = pytest.importorskip("megabonk_bot.runtime.hud_cache")
    seen = []

    def _fake_read(frame, regions=None):
        seen.append(frame)
        return {"time": frame}

    cache = runtime_hud_cache.HudTelemetryCache(
        read_hud_telemetry=_fake_read,
        regions=None,
        interval_s=0.01,
    )
    cache.start()
    try:
        cache.submit(100)
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if cache.snapshot().get("time") == 100:
                break
            time.sleep(0.01)

        cache.submit(200)
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if cache.snapshot().get("time") == 200:
                break
            time.sleep(0.01)
    finally:
        cache.close()

    assert seen
    assert cache.snapshot()["time"] == 200
