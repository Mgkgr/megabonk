from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardComponents:
    danger: float = 0.0
    stuck: float = 0.0
    loot: float = 0.0

    @property
    def total(self) -> float:
        return float(self.danger + self.stuck + self.loot)


def compute_reward_components(
    *,
    danger_center,
    danger_area: float,
    danger_frac: float,
    loot_center,
    loot_area: float,
    loot_frac: float,
    is_stuck: bool,
    reward_danger_k: float,
    reward_stuck_k: float,
    reward_loot_k: float,
) -> RewardComponents:
    danger = 0.0
    loot = 0.0
    stuck = 0.0
    if danger_center is not None and float(danger_area) > 0:
        danger = -float(reward_danger_k) * float(danger_frac)
    if loot_center is not None and float(loot_area) > 0:
        loot = float(reward_loot_k) * float(loot_frac)
    if is_stuck:
        stuck = -float(reward_stuck_k)
    return RewardComponents(danger=danger, stuck=stuck, loot=loot)
