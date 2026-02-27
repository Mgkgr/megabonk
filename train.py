# --- DPI AWARE (must be first imports) ---
from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()
# --- end DPI AWARE ---

import argparse  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from megabonk_env import MegabonkEnv  # noqa: E402
from megabonk_bot.regions import build_regions  # noqa: E402

WINDOW = "Megabonk"


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
                telemetry_keys = [k for k in ("hp_ratio", "lvl", "kills", "time") if k in i]
                telemetry_repr = " ".join(f"{k}={self._fmt(i.get(k))}" for k in telemetry_keys)
                print(
                    f"[{self.n_calls}] screen={i.get('screen')} "
                    f"r=({self._fmt(i.get('r_alive'))},{self._fmt(i.get('r_xp'))},{self._fmt(i.get('r_dmg'))}) "
                    f"auto={i.get('autopilot')} safety={i.get('safety_override')} "
                    f"safety_strength={self._fmt(i.get('safety_strength'))} "
                    f"{telemetry_repr}"
                )
        return True

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on MegabonkEnv.")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total PPO timesteps.")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout length per env.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument("--step-hz", type=int, default=12, help="Environment loop frequency.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device: cpu | cuda | cuda:0 ...",
    )
    parser.add_argument("--window-title", type=str, default=WINDOW, help="Game window title.")
    parser.add_argument("--model-out", type=str, default="megabonk_ppo_cnn", help="Output model path.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print callback frequency in environment steps.",
    )
    return parser.parse_args()


def make_env(window_title: str, step_hz: int):
    env = MegabonkEnv(
        window_title=window_title,
        step_hz=step_hz,
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

def main():
    args = parse_args()
    env = DummyVecEnv([lambda: make_env(args.window_title, args.step_hz)])
    # SB3 ожидает (C,H,W), а у нас (H,W,C) — транспонируем
    env = VecTransposeImage(env)

    torch.backends.cudnn.benchmark = True
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна: передайте --device cpu или установите torch с CUDA.")

    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("torch.cuda.device_name:", torch.cuda.get_device_name(0))
    print("device:", device)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.lr,
        tensorboard_log="tb",
    )

    model.learn(total_timesteps=args.timesteps, callback=PrintCallback(every=args.log_every))
    model.save(args.model_out)


if __name__ == "__main__":
    main()
