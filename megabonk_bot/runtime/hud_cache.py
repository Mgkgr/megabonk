from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


class HudTelemetryCache:
    def __init__(
        self,
        *,
        read_hud_telemetry: Callable[..., dict[str, Any]],
        regions: dict[str, Any] | None,
        interval_s: float,
    ) -> None:
        self._read_hud_telemetry = read_hud_telemetry
        self._regions = regions
        self._interval_s = max(0.0, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = None
        self._latest_values: dict[str, Any] = {}
        self._latest_ts = 0.0
        self._last_poll_ts = 0.0

    @property
    def enabled(self) -> bool:
        return self._interval_s > 0.0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="runtime-hud-ocr",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def submit(self, frame) -> None:
        if not self.enabled or frame is None:
            return
        with self._lock:
            self._frame = frame

    def set_regions(self, regions: dict[str, Any] | None) -> None:
        with self._lock:
            self._regions = regions

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            values = dict(self._latest_values)
            ts = self._latest_ts
        values.setdefault("debug_dumped", False)
        values.setdefault("hud_fail_streak", 0)
        values["hud_ts"] = ts if ts > 0.0 else None
        return values

    def refresh(self, frame, *, now: float | None = None) -> dict[str, Any]:
        with self._lock:
            regions = self._regions
        values = dict(self._read_hud_telemetry(frame, regions=regions))
        completed_ts = time.time() if now is None else float(now)
        with self._lock:
            self._latest_values = values
            self._latest_ts = completed_ts
            self._last_poll_ts = completed_ts
        return dict(values)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            with self._lock:
                frame = self._frame
                last_poll_ts = self._last_poll_ts
            if frame is None:
                self._stop.wait(0.01)
                continue
            wait_s = self._interval_s - (now - last_poll_ts)
            if wait_s > 0.0:
                self._stop.wait(min(wait_s, 0.05))
                continue
            try:
                self.refresh(frame)
            except Exception:
                LOGGER.warning("HUD telemetry cache refresh failed", exc_info=True)
                self._stop.wait(min(self._interval_s or 0.1, 0.1))
