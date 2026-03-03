#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any


TELEMETRY_ONLY_COLUMNS = [
    "time",
    "gold",
    "lvl",
    "kills",
    "hp_ratio",
    "latency_ms",
]


FULL_COLUMNS = [
    "schema_version",
    "ts",
    "mode",
    "frame_id",
    "screen",
    "step_hz",
    "dt_ms",
    "window_title",
    "frame_width",
    "frame_height",
    "telemetry_time",
    "telemetry_gold",
    "telemetry_lvl",
    "telemetry_kills",
    "telemetry_hp_ratio",
    "latency_ms",
    "is_dead",
    "is_upgrade",
    "safe_sector",
    "reason",
    "restart",
    "boss_prep",
    "boss_name",
    "preferred_direction",
    "capture_bad_grab_count",
    "capture_last_error",
    "hud_debug_dumped",
    "hud_fail_streak",
]


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        for lineno, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
            if not isinstance(event, dict):
                continue
            yield event


def _extract_frame_size(raw_value: Any) -> tuple[Any, Any]:
    if isinstance(raw_value, dict):
        return raw_value.get("width"), raw_value.get("height")
    if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
        return raw_value[0], raw_value[1]
    return None, None


def _event_to_full_row(event: dict[str, Any]) -> dict[str, Any]:
    telemetry = event.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
    capture = event.get("capture")
    if not isinstance(capture, dict):
        capture = {}
    hud = event.get("hud")
    if not isinstance(hud, dict):
        hud = {}
    frame_w, frame_h = _extract_frame_size(event.get("frame_size"))
    return {
        "schema_version": event.get("schema_version", "legacy"),
        "ts": event.get("ts"),
        "mode": event.get("mode"),
        "frame_id": event.get("frame_id"),
        "screen": event.get("screen"),
        "step_hz": event.get("step_hz"),
        "dt_ms": event.get("dt_ms"),
        "window_title": event.get("window_title"),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "telemetry_time": telemetry.get("time"),
        "telemetry_gold": telemetry.get("gold"),
        "telemetry_lvl": telemetry.get("lvl"),
        "telemetry_kills": telemetry.get("kills"),
        "telemetry_hp_ratio": telemetry.get("hp_ratio"),
        "latency_ms": event.get("latency_ms"),
        "is_dead": event.get("is_dead"),
        "is_upgrade": event.get("is_upgrade"),
        "safe_sector": event.get("safe_sector"),
        "reason": event.get("reason"),
        "restart": event.get("restart"),
        "boss_prep": event.get("boss_prep"),
        "boss_name": event.get("boss_name"),
        "preferred_direction": event.get("preferred_direction"),
        "capture_bad_grab_count": capture.get("bad_grab_count"),
        "capture_last_error": capture.get("last_error"),
        "hud_debug_dumped": hud.get("debug_dumped"),
        "hud_fail_streak": hud.get("fail_streak"),
    }


def _event_to_telemetry_row(event: dict[str, Any]) -> dict[str, Any]:
    telemetry = event.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
    return {
        "time": telemetry.get("time"),
        "gold": telemetry.get("gold"),
        "lvl": telemetry.get("lvl"),
        "kills": telemetry.get("kills"),
        "hp_ratio": telemetry.get("hp_ratio"),
        "latency_ms": event.get("latency_ms"),
    }


def _write_csv_stdlib(rows: list[dict[str, Any]], out_path: Path, columns: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_pandas(rows: list[dict[str, Any]], out_path: Path, columns: list[str]) -> None:
    import pandas as pd  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(out_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Export runtime JSONL events to CSV.")
    parser.add_argument("input", help="Path to JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    parser.add_argument(
        "--telemetry-only",
        action="store_true",
        help="Export only telemetry columns: time/gold/lvl/kills/hp_ratio/latency_ms",
    )
    parser.add_argument(
        "--use-pandas",
        action="store_true",
        help="Use pandas for CSV export (optional dependency).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    out_path = (
        Path(args.output).resolve()
        if args.output
        else in_path.with_suffix(".telemetry.csv" if args.telemetry_only else ".csv")
    )

    rows: list[dict[str, Any]] = []
    for event in _iter_jsonl(in_path):
        if args.telemetry_only:
            rows.append(_event_to_telemetry_row(event))
        else:
            rows.append(_event_to_full_row(event))

    columns = TELEMETRY_ONLY_COLUMNS if args.telemetry_only else FULL_COLUMNS
    if args.use_pandas:
        _write_csv_pandas(rows, out_path, columns)
    else:
        _write_csv_stdlib(rows, out_path, columns)
    print(f"Exported {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
