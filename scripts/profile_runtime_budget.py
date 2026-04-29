#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megabonk_bot.asset_catalog import load_curated_catalogs
from megabonk_bot.config import load_config
from megabonk_bot.recognition import analyze_scene
from megabonk_bot.regions import build_regions
from megabonk_bot.runtime.overlay import draw_runtime_overlay
from megabonk_bot.runtime.perf_budget import (
    build_performance_sample,
    normalize_performance_budget_ms,
    summarize_performance_samples,
)
from megabonk_bot.runtime_logic import BotMode, build_scene_snapshot
from megabonk_bot.templates import load_templates
from megabonk_bot.ui_ocr import UiTextDetection
from run_runtime_bot import _resolve_optional_path, _resolve_project_base_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile runtime capture/HUD/scene/overlay stages on a sample frame.",
    )
    parser.add_argument("--config", default="config/bot_profile.yaml", help="Runtime config path")
    parser.add_argument("--sample-frame", default="screen.png", help="Sample BGR frame path")
    parser.add_argument("--iterations", type=int, default=20, help="Number of profiling ticks")
    parser.add_argument(
        "--live-window",
        default=None,
        help="Window title substring for real WindowCapture profiling",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "printwindow", "mss"),
        default=None,
        help="Capture backend for --live-window",
    )
    parser.add_argument(
        "--output",
        default="logs/runtime_performance_baseline.json",
        help="Where to write the baseline JSON summary",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay rendering stage")
    return parser.parse_args()


def _load_frame(cv2, frame_path: Path):
    frame = cv2.imread(str(frame_path))
    if frame is None or frame.size == 0:
        raise RuntimeError(f"Failed to read sample frame: {frame_path}")
    return frame


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")

    import cv2

    config_path = Path(args.config).resolve()
    resource_base_dir = _resolve_project_base_dir(config_path)
    config = load_config(config_path)
    runtime_cfg = config["runtime"]
    detect_cfg = config["detection"]
    step_hz = max(1, int(runtime_cfg["step_hz"]))
    budget_ms = normalize_performance_budget_ms(
        runtime_cfg.get("performance_budget_ms"),
        tick_budget_ms=(1000.0 / step_hz),
    )

    live_capture = None
    frame_path = _resolve_optional_path(args.sample_frame, base_dir=resource_base_dir)
    if args.live_window:
        from window_capture import WindowCapture

        live_capture = WindowCapture.create(
            str(args.live_window),
            capture_backend=str(args.capture_backend or runtime_cfg.get("capture_backend", "auto")),
        )
        base_frame = live_capture.grab()
        if base_frame is None or base_frame.size == 0:
            raise RuntimeError(f"Failed to capture live window: {args.live_window}")
    else:
        if frame_path is None:
            raise ValueError("--sample-frame must not be empty")
        base_frame = _load_frame(cv2, frame_path)
    h, w = base_frame.shape[:2]
    regions = build_regions(w, h)
    templates_dir = _resolve_optional_path(
        runtime_cfg.get("templates_dir"),
        base_dir=resource_base_dir,
    )
    templates = load_templates(str(templates_dir))
    catalogs = load_curated_catalogs(
        asset_refs_dir=_resolve_optional_path(
            detect_cfg.get("asset_refs_dir"),
            base_dir=resource_base_dir,
        ),
        enemy_catalog_path=_resolve_optional_path(
            detect_cfg.get("enemy_catalog_path"),
            base_dir=resource_base_dir,
        ),
        world_catalog_path=_resolve_optional_path(
            detect_cfg.get("world_catalog_path"),
            base_dir=resource_base_dir,
        ),
        projectile_catalog_path=_resolve_optional_path(
            detect_cfg.get("projectile_catalog_path"),
            base_dir=resource_base_dir,
        ),
        ocr_lexicon_path=_resolve_optional_path(
            detect_cfg.get("ocr_lexicon_path"),
            base_dir=resource_base_dir,
        ),
    )

    hud_values = {
        "debug_dumped": False,
        "hud_fail_streak": 0,
        "time": None,
        "kills": None,
        "lvl": None,
        "gold": None,
        "hp_ratio": None,
    }
    overlay_redraw_interval_ticks = max(1, int(runtime_cfg.get("overlay_redraw_interval_ticks", 1)))
    last_overlay_frame = None
    samples = []
    for frame_id in range(1, int(args.iterations) + 1):
        loop_start = time.perf_counter()
        stages_ms: dict[str, float] = {}

        stage_start = time.perf_counter()
        if live_capture is not None:
            frame = live_capture.grab()
            if frame is None or frame.size == 0:
                raise RuntimeError(f"Failed to capture live window: {args.live_window}")
        else:
            frame = base_frame.copy()
        stages_ms["capture"] = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        objective_ui = UiTextDetection()
        cached_hud_values = dict(hud_values)
        stages_ms["hud"] = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        analysis = analyze_scene(
            frame,
            templates=templates,
            catalogs=catalogs,
            regions=regions,
            objective_ui=objective_ui,
            grid_rows=int(detect_cfg["grid_rows"]),
            grid_cols=int(detect_cfg["grid_cols"]),
            enemy_hsv_lower=tuple(detect_cfg["enemy_hsv_lower"]),
            enemy_hsv_upper=tuple(detect_cfg["enemy_hsv_upper"]),
            enemy_min_area=float(detect_cfg["enemy_min_area"]),
            interact_threshold=float(detect_cfg["interact_threshold"]),
            enemy_classifier_mode=str(detect_cfg.get("enemy_classifier_mode", "hybrid")),
            minimap_enabled=bool(detect_cfg.get("minimap_enabled", False)),
            projectiles_enabled=bool(detect_cfg.get("projectiles_enabled", False)),
            world_objects_enabled=bool(detect_cfg.get("world_objects_enabled", False)),
            world_object_families=tuple(detect_cfg.get("world_object_families", [])),
            analysis_scale=float(detect_cfg.get("analysis_scale", 1.0)),
        )
        stages_ms["scene_analysis"] = (time.perf_counter() - stage_start) * 1000.0
        snapshot = build_scene_snapshot(
            frame_id=frame_id,
            ts=time.time(),
            frame_width=w,
            frame_height=h,
            analysis=analysis,
            hud_values=cached_hud_values,
            is_dead=False,
            is_upgrade=False,
        )

        stage_start = time.perf_counter()
        if not args.no_overlay:
            redraw_overlay = (
                last_overlay_frame is None
                or (frame_id % overlay_redraw_interval_ticks) == 0
            )
            if redraw_overlay:
                last_overlay_frame, _ = draw_runtime_overlay(
                    cv2,
                    frame,
                    analysis,
                    snapshot,
                    mode=BotMode.OFF,
                    action_reason="profile_runtime_budget",
                    hud_values=cached_hud_values,
                    hud_regions=regions,
                    transparent_canvas=bool(runtime_cfg.get("overlay_transparent", False)),
                )
        stages_ms["overlay"] = (time.perf_counter() - stage_start) * 1000.0

        samples.append(
            build_performance_sample(
                stages_ms,
                budget_ms=budget_ms,
                tick_ms=(time.perf_counter() - loop_start) * 1000.0,
            )
        )

    summary = summarize_performance_samples(samples)
    summary["source"] = {
        "config": str(config_path),
        "sample_frame": str(frame_path) if live_capture is None else None,
        "live_window": str(args.live_window) if live_capture is not None else None,
        "capture_backend": str(args.capture_backend or runtime_cfg.get("capture_backend", "auto")),
        "iterations": int(args.iterations),
        "step_hz": step_hz,
        "overlay_profiled": not args.no_overlay,
        "overlay_redraw_interval_ticks": overlay_redraw_interval_ticks,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
