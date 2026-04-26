import logging

from megabonk_bot.memory_probe import ExternalProcessProbe
from megabonk_bot.runtime.event_logger import JsonlEventLogger
from megabonk_bot.runtime.input_controller import release_all_keys


def test_event_logger_close_logs_failure(tmp_path, caplog):
    logger = JsonlEventLogger(tmp_path / "runtime.jsonl")

    class BrokenFile:
        def close(self):
            raise OSError("disk is gone")

    logger._fp = BrokenFile()
    caplog.set_level(logging.WARNING)

    logger.close()

    assert any("Failed to close runtime event log" in record.message for record in caplog.records)


def test_release_all_keys_logs_failures(caplog):
    class BrokenInput:
        def keyUp(self, key):
            if key in {"w", "space"}:
                raise RuntimeError(f"cannot release {key}")

    caplog.set_level(logging.WARNING)

    release_all_keys(BrokenInput())

    assert any("Failed to release keys cleanly" in record.message for record in caplog.records)
    assert any("cannot release w" in record.message for record in caplog.records)


def test_memory_probe_logs_degraded_sampling(monkeypatch, caplog):
    probe = ExternalProcessProbe(window_title="Megabonk", signatures={"player_world_pos": {}})
    probe._kernel32 = object()
    probe._user32 = object()
    probe._ctypes = object()

    def _raise():
        raise RuntimeError("probe read failed")

    monkeypatch.setattr(probe, "_ensure_handle", _raise)
    caplog.set_level(logging.WARNING)

    result = probe.sample(now_ts=1.0)

    assert result.status == "degraded_error"
    assert result.details["error"] == "probe read failed"
    assert any("Memory probe sampling degraded" in record.message for record in caplog.records)
