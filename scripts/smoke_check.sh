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
  megabonk_bot/max_model.py

echo "[smoke] OK"
