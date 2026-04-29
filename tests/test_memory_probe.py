from megabonk_bot.memory_probe import ExternalProcessProbe, NullProbe
from megabonk_bot.window_lookup import select_window_title_match

import run_runtime_bot as runtime_bot


def test_external_probe_without_signatures_does_not_open_process(monkeypatch):
    probe = ExternalProcessProbe(window_title="Megabonk", signatures={})
    probe._kernel32 = object()
    probe._user32 = object()
    probe._ctypes = object()

    def fail_if_called():
        raise AssertionError("empty-signature probe must not open the process")

    monkeypatch.setattr(probe, "_ensure_handle", fail_if_called)

    result = probe.sample(now_ts=1.0)

    assert result.status == "disabled"
    assert probe._pid == 0
    assert probe._handle is None


def test_runtime_uses_null_probe_for_empty_signature_file(tmp_path):
    signatures_path = tmp_path / "memory_signatures.json"
    signatures_path.write_text("{}", encoding="utf-8")

    probe = runtime_bot._build_world_probe(
        enabled=True,
        window_title="Megabonk",
        poll_interval_s=0.25,
        signatures_path=signatures_path,
    )

    assert isinstance(probe, NullProbe)
    assert probe.sample(now_ts=1.0).status == "disabled"


def test_runtime_builds_external_probe_for_non_empty_signature_file(tmp_path, monkeypatch):
    signatures_path = tmp_path / "memory_signatures.json"
    signatures_path.write_text('{"player_world_pos": {"absolute_address": 4096}}', encoding="utf-8")
    captured = {}

    class DummyProbe:
        def __init__(self, *, window_title, poll_interval_s, signatures):
            captured["window_title"] = window_title
            captured["poll_interval_s"] = poll_interval_s
            captured["signatures"] = signatures

    monkeypatch.setattr(runtime_bot, "ExternalProcessProbe", DummyProbe)

    probe = runtime_bot._build_world_probe(
        enabled=True,
        window_title="Megabonk",
        poll_interval_s=0.5,
        signatures_path=signatures_path,
    )

    assert isinstance(probe, DummyProbe)
    assert captured == {
        "window_title": "Megabonk",
        "poll_interval_s": 0.5,
        "signatures": {"player_world_pos": {"absolute_address": 4096}},
    }


def test_window_title_match_prefers_exact_then_shortest_substring():
    assert (
        select_window_title_match(
            "Megabonk",
            [
                (10, "Megabonk - debug"),
                (20, "MEGABONK"),
                (30, "Other"),
            ],
        )
        == 20
    )
    assert (
        select_window_title_match(
            "mega",
            [
                (10, "Megabonk - debug"),
                (20, "Megabonk"),
                (30, "Very long Megabonk capture"),
            ],
        )
        == 20
    )


def test_external_probe_uses_windowcapture_substring_strategy_for_pid():
    class FakeDWORD:
        def __init__(self):
            self.value = 0

    class FakeWintypes:
        BOOL = bool
        HWND = int
        LPARAM = int
        DWORD = FakeDWORD

    class FakeBuffer:
        def __init__(self, _size):
            self.value = ""

    class FakeCtypes:
        def WINFUNCTYPE(self, *_args):
            return lambda callback: callback

        def create_unicode_buffer(self, size):
            return FakeBuffer(size)

        def byref(self, value):
            return value

    class FakeUser32:
        titles = {
            101: "Megabonk - debug",
            202: "MEGABONK",
            303: "Other window",
        }
        pids = {
            101: 1001,
            202: 2002,
            303: 3003,
        }

        def EnumWindows(self, callback, lparam):
            for hwnd in self.titles:
                if not callback(hwnd, lparam):
                    break
            return True

        def IsWindowVisible(self, hwnd):
            return hwnd != 303

        def GetWindowTextLengthW(self, hwnd):
            return len(self.titles[hwnd])

        def GetWindowTextW(self, hwnd, buf, _size):
            buf.value = self.titles[hwnd]
            return len(buf.value)

        def GetWindowThreadProcessId(self, hwnd, pid):
            pid.value = self.pids[hwnd]
            return True

    probe = ExternalProcessProbe(window_title="mega", signatures={"player_world_pos": {}})
    probe._ctypes = FakeCtypes()
    probe._wintypes = FakeWintypes()
    probe._user32 = FakeUser32()

    assert probe._find_pid_by_window_title("mega") == 2002
