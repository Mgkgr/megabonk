from __future__ import annotations

from dataclasses import dataclass

from megabonk_bot.runtime_logic import BotMode


@dataclass
class RuntimeStateMachine:
    mode: BotMode = BotMode.OFF

    def on_events(
        self,
        *,
        toggle: bool = False,
        panic: bool = False,
        dead_detected: bool = False,
        running_restored: bool = False,
    ) -> BotMode:
        if panic:
            self.mode = BotMode.PANIC
            return self.mode
        if toggle:
            if self.mode in {BotMode.OFF, BotMode.PANIC}:
                self.mode = BotMode.ACTIVE
            else:
                self.mode = BotMode.OFF
            return self.mode
        if self.mode == BotMode.ACTIVE and dead_detected:
            self.mode = BotMode.RECOVERY
            return self.mode
        if self.mode == BotMode.RECOVERY and running_restored:
            self.mode = BotMode.ACTIVE
            return self.mode
        return self.mode
