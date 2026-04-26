from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from megabonk_bot.asset_catalog import OcrLexicon
from megabonk_bot.ui_ocr import UiTextDetection

LOGGER = logging.getLogger(__name__)


class ObjectiveUiCache:
    def __init__(
        self,
        *,
        read_objective_ui: Callable[..., UiTextDetection],
        lexicon: OcrLexicon | None,
        interval_s: float,
    ) -> None:
        self._read_objective_ui = read_objective_ui
        self._lexicon = lexicon
        self._interval_s = max(0.0, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = None
        self._latest_value = UiTextDetection()
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
            name="runtime-objective-ocr",
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

    def snapshot(self) -> UiTextDetection:
        with self._lock:
            return self._latest_value

    def refresh(self, frame, *, now: float | None = None) -> UiTextDetection:
        value = self._read_objective_ui(frame, lexicon=self._lexicon)
        completed_ts = time.time() if now is None else float(now)
        with self._lock:
            self._latest_value = value
            self._latest_ts = completed_ts
            self._last_poll_ts = completed_ts
        return value

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
                LOGGER.warning("Objective UI cache refresh failed", exc_info=True)
                self._stop.wait(min(self._interval_s or 0.1, 0.1))
