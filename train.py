# --- DPI AWARE (must be first imports) ---
import ctypes


def enable_dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


enable_dpi_awareness()
# --- end DPI AWARE ---

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from megabonk_env import MegabonkEnv
from megabonk_bot.regions import build_regions

WINDOW = "Megabonk"  # подстрой под реальный заголовок окна


class PrintCallback(BaseCallback):
    def __init__(self, every=200):
        super().__init__()
        self.every = every

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0:
            infos = self.locals.get("infos")
            if infos:
                i = infos[0]
                print(
                    f"[{self.n_calls}] screen={i.get('screen')} "
                    f"xp={i.get('xp_fill'):.3f} hp={i.get('hp_fill'):.3f} "
                    f"r=({i.get('r_alive'):.3f},{i.get('r_xp'):.3f},{i.get('r_dmg'):.3f}) "
                    f"auto={i.get('autopilot')}"
                )
        return True

def make_env():
    env = MegabonkEnv(window_title=WINDOW, step_hz=12, templates_dir="templates", regions_builder=build_regions)
    return Monitor(env)

env = DummyVecEnv([make_env])
# SB3 ожидает (C,H,W), а у нас (H,W,C) — транспонируем
env = VecTransposeImage(env)
device = "cuda" if torch.cuda.is_available() else "cpu"
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
