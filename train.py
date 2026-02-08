# --- DPI AWARE (must be first imports) ---
from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()
# --- end DPI AWARE ---

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from megabonk_env import MegabonkEnv  # noqa: E402
from megabonk_bot.regions import build_regions  # noqa: E402

WINDOW = "Megabonk"  # подстрой под реальный заголовок окна


class PrintCallback(BaseCallback):
    def __init__(self, every=200):
        super().__init__()
        self.every = every

    @staticmethod
    def _fmt(value) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return "n/a"

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0:
            infos = self.locals.get("infos")
            if infos:
                i = infos[0]
                print(
                    f"[{self.n_calls}] screen={i.get('screen')} "
                    f"xp={self._fmt(i.get('xp_fill'))} hp={self._fmt(i.get('hp_fill'))} "
                    f"r=({self._fmt(i.get('r_alive'))},{self._fmt(i.get('r_xp'))},{self._fmt(i.get('r_dmg'))}) "
                    f"auto={i.get('autopilot')} safety={i.get('safety_override')} "
                    f"safety_strength={self._fmt(i.get('safety_strength'))} "
                    f"recognized_time={i.get('time')}"
                )
        return True

def make_env():
    env = MegabonkEnv(
        window_title=WINDOW,
        step_hz=12,
        templates_dir="templates",
        regions_builder=build_regions,
        safety_enabled=True,
        safety_strength=1.0,
        safety_anneal_steps=200_000,
        safety_min_strength=0.0,
        safety_danger_frac_threshold=0.02,
        safety_stuck_enabled=True,
    )
    return Monitor(env)

env = DummyVecEnv([make_env])
# SB3 ожидает (C,H,W), а у нас (H,W,C) — транспонируем
env = VecTransposeImage(env)
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна: проверьте установку драйвера и torch+CUDA.")
device = "cuda"
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
print("torch.cuda.device_name:", torch.cuda.get_device_name(0))
print("device:", device)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    device=device,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    learning_rate=2.5e-4,
    tensorboard_log="tb",
)

model.learn(total_timesteps=2_000_000, callback=PrintCallback(every=200))
model.save("megabonk_ppo_cnn")
