import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from window_capture import get_window_region
from megabonk_env import MegabonkEnv

WINDOW = "Megabonk"  # подстрой под реальный заголовок окна

def make_env():
    region = get_window_region(WINDOW)
    env = MegabonkEnv(region=region, step_hz=12)
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
)

model.learn(total_timesteps=2_000_000)
model.save("megabonk_ppo_cnn")
