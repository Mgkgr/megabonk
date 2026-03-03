#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[smoke] checking merge-conflict markers"
if rg -n --glob '*.py' '^(<<<<<<< |=======|>>>>>>> )' "$ROOT_DIR"; then
  echo "[smoke] conflict markers detected"
  exit 1
fi

echo "[smoke] checking python syntax"
python -m py_compile \
  megabonk_env.py \
  train.py \
  play.py \
  run_runtime_bot.py \
  autopilot.py \
  megabonk_bot/hud.py \
  megabonk_bot/runtime_logic.py \
  megabonk_bot/runtime_state.py \
  megabonk_bot/max_model.py \
  megabonk_bot/runtime/input_controller.py \
  megabonk_bot/runtime/event_logger.py \
  megabonk_bot/runtime/overlay.py \
  megabonk_bot/runtime/recovery.py \
  megabonk_bot/runtime/loop.py \
  megabonk_bot/env/hud_worker.py \
  megabonk_bot/env/reward_engine.py \
  megabonk_bot/env/safety_policy.py \
  megabonk_bot/env/restart_flow.py

echo "[smoke] OK"
