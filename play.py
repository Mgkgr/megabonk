import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from window_capture import get_window_region
from megabonk_env import MegabonkEnv
from regions import build_regions

WINDOW = "Megabonk"

def make_env():
    region = get_window_region(WINDOW)
    return MegabonkEnv(region=region, step_hz=12, templates_dir="templates", regions_builder=build_regions)

env = DummyVecEnv([make_env])
env = VecTransposeImage(env)

model = PPO.load("megabonk_ppo_cnn", device="cuda")

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
        time.sleep(0.2)
