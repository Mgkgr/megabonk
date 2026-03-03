import pytest

try:
    import window_capture as wc
except Exception as exc:  # pragma: no cover - на не-Windows импорт ожидаемо падает
    pytest.skip(f"window_capture unavailable: {exc}", allow_module_level=True)


def test_grab_tracks_last_error_and_bad_grab_count(monkeypatch):
    cap = wc.WindowCapture(window_title="Megabonk", hwnd=1, sct=None, capture_backend="printwindow")
    monkeypatch.setattr(cap, "get_bbox", lambda: {"left": 0, "top": 0, "width": 10, "height": 10})
    monkeypatch.setattr(cap, "focus_if_needed", lambda topmost=False, min_interval_s=0.05: None)
    monkeypatch.setattr(wc.time, "sleep", lambda _: None)

    def _fail(*_args, **_kwargs):
        raise RuntimeError("printwindow failed")

    monkeypatch.setattr(wc, "_grab_with_printwindow", _fail)

    frame = cap.grab()
    assert frame is None
    assert cap._bad_grab_count == 3
    assert cap._last_grab_error == "RuntimeError: printwindow failed"

    diagnostics = cap.get_capture_diagnostics()
    assert diagnostics["bad_grab_count"] == 3
    assert diagnostics["last_error"] == "RuntimeError: printwindow failed"
