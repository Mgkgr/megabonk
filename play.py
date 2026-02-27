from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()

import argparse  # noqa: E402
import time  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage  # noqa: E402
from megabonk_env import MegabonkEnv  # noqa: E402
from megabonk_bot.regions import build_regions  # noqa: E402

WINDOW = "Megabonk"

def parse_args():
    parser = argparse.ArgumentParser(description="Run trained PPO policy on MegabonkEnv.")
    parser.add_argument("--model-path", type=str, default="megabonk_ppo_cnn", help="Path to saved model.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for inference.")
    parser.add_argument("--window-title", type=str, default=WINDOW, help="Game window title.")
    parser.add_argument("--step-hz", type=int, default=12, help="Environment loop frequency.")
    parser.add_argument("--sleep-on-done", type=float, default=0.2, help="Pause between episodes.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling (default: deterministic).",
    )
    return parser.parse_args()

def make_env(window_title: str, step_hz: int):
    return MegabonkEnv(
        window_title=window_title,
        step_hz=step_hz,
        templates_dir="templates",
        regions_builder=build_regions,
    )

def main():
    args = parse_args()
    env = DummyVecEnv([lambda: make_env(args.window_title, args.step_hz)])
    env = VecTransposeImage(env)
    model = PPO.load(args.model_path, device=args.device)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
            time.sleep(args.sleep_on_done)


if __name__ == "__main__":
    main()
