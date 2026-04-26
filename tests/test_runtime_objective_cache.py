import time

import pytest

from megabonk_bot.ui_ocr import UiTextDetection


def test_objective_cache_refresh_updates_snapshot():
    runtime_objective_cache = pytest.importorskip("megabonk_bot.runtime.objective_cache")
    cache = runtime_objective_cache.ObjectiveUiCache(
        read_objective_ui=lambda frame, lexicon=None: UiTextDetection(
            text=str(frame),
            normalized=f"objective_{frame}",
            confidence=77.0,
            region=(1, 2, 3, 4),
            source="objective_cache",
        ),
        lexicon=None,
        interval_s=1.2,
    )

    value = cache.refresh(321, now=12.5)
    snapshot = cache.snapshot()

    assert value == snapshot
    assert snapshot.normalized == "objective_321"
    assert snapshot.source == "objective_cache"


def test_objective_cache_background_poll_uses_latest_submitted_frame():
    runtime_objective_cache = pytest.importorskip("megabonk_bot.runtime.objective_cache")
    seen = []

    def _fake_read(frame, lexicon=None):
        seen.append(frame)
        return UiTextDetection(
            text=str(frame),
            normalized=f"objective_{frame}",
            confidence=70.0,
            source="objective_cache",
        )

    cache = runtime_objective_cache.ObjectiveUiCache(
        read_objective_ui=_fake_read,
        lexicon=None,
        interval_s=0.01,
    )
    cache.start()
    try:
        cache.submit(100)
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if cache.snapshot().normalized == "objective_100":
                break
            time.sleep(0.01)

        cache.submit(200)
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if cache.snapshot().normalized == "objective_200":
                break
            time.sleep(0.01)
    finally:
        cache.close()

    assert seen
    assert cache.snapshot().normalized == "objective_200"
