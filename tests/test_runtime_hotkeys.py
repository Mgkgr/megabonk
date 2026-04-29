from types import SimpleNamespace

import run_runtime_bot
from run_runtime_bot import WinHotkeyPoller


class _FakeUser32:
    def __init__(self):
        self.states: dict[int, list[int]] = {}

    def set_states(self, vk: int, states: list[int]) -> None:
        self.states[int(vk)] = list(states)

    def GetAsyncKeyState(self, vk: int) -> int:
        queue = self.states.setdefault(int(vk), [0])
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]


def test_hotkey_poller_detects_short_press_between_polls(monkeypatch):
    user32 = _FakeUser32()
    monkeypatch.setattr(
        run_runtime_bot.ctypes,
        "windll",
        SimpleNamespace(user32=user32),
        raising=False,
    )
    user32.set_states(0x77, [0x0001])
    user32.set_states(0x7B, [0])

    poller = WinHotkeyPoller(enabled=True, toggle_vk=0x77, panic_vk=0x7B)

    assert poller.poll() == (True, False)


def test_hotkey_poller_detects_repeated_short_presses(monkeypatch):
    user32 = _FakeUser32()
    monkeypatch.setattr(
        run_runtime_bot.ctypes,
        "windll",
        SimpleNamespace(user32=user32),
        raising=False,
    )
    user32.set_states(0x77, [0x0001, 0x0001])
    user32.set_states(0x7B, [0])

    poller = WinHotkeyPoller(enabled=True, toggle_vk=0x77, panic_vk=0x7B)

    assert poller.poll() == (True, False)
    assert poller.poll() == (True, False)
