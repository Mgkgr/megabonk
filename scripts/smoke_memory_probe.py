#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megabonk_bot.memory_probe import ExternalProcessProbe


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-check memory probe signatures against a live game window.")
    parser.add_argument("--window", default="Megabonk", help="Window title substring")
    parser.add_argument(
        "--signatures",
        default="config/memory_signatures.json",
        help="Path to memory signatures JSON",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=0.1,
        help="Probe poll interval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signatures_path = Path(args.signatures).resolve()
    if not signatures_path.exists():
        raise FileNotFoundError(f"Signatures file not found: {signatures_path}")
    raw = json.loads(signatures_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Signatures file must contain a JSON object")

    probe = ExternalProcessProbe(
        window_title=str(args.window),
        poll_interval_s=float(args.poll_interval_s),
        signatures=raw,
    )
    try:
        result = probe.sample()
    finally:
        probe.close()
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
