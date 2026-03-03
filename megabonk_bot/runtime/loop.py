from __future__ import annotations

import time
from dataclasses import dataclass


class RuntimeBotRunner:
    """Минимальный оркестратор цикла runtime."""

    def __init__(self, tick_fn):
        self._tick_fn = tick_fn

    def run(self) -> None:
        while True:
            should_continue = bool(self._tick_fn())
            if not should_continue:
                break


@dataclass
class RateLimiter:
    interval_s: float
    _last_ts: float = 0.0

    def allow(self, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        if (current - self._last_ts) < max(0.0, float(self.interval_s)):
            return False
        self._last_ts = current
        return True


def maybe_warn_capture_error(
    *,
    capture_last_error: str | None,
    capture_bad_grab_count: int,
    limiter: RateLimiter,
    enabled: bool,
    log_fn=print,
) -> None:
    if not enabled or not capture_last_error:
        return
    if limiter.allow():
        log_fn(
            "[WARN] capture failures="
            f"{int(capture_bad_grab_count)} last_error={capture_last_error}"
        )
