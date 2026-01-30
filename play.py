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

import time  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage  # noqa: E402
from megabonk_env import MegabonkEnv  # noqa: E402
from megabonk_bot.regions import build_regions  # noqa: E402

WINDOW = "Megabonk"

def make_env():
    return MegabonkEnv(window_title=WINDOW, step_hz=12, templates_dir="templates", regions_builder=build_regions)

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
