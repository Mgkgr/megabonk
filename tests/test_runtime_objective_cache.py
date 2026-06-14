import time
from pathlib import Path

import ast

import pytest

from megabonk_bot.config import load_config
from run_runtime_bot import (
    NullHudTelemetryCache,
    NullObjectiveUiCache,
    _build_hud_cache,
    _build_objective_cache,
    _build_screen_detector,
    _build_world_probe,
    _resolve_allow_map_scan,
    _resolve_runtime_detection_flags,
    _resolve_runtime_movement_flags,
    _should_auto_pick_upgrade,
)
from megabonk_bot.ui_ocr import UiTextDetection


def test_objective_cache_refresh_updates_snapshot():
    runtime_objective_cache = pytest.importorskip("megabonk_bot.runtime.objective_cache")
    cache = runtime_objective_cache.ObjectiveUiCache(
        read_objective_ui=lambda frame, lexicon=None: UiTextDetection(
            text=str(frame),
            normalized=f"objective_{frame}",
            confidence=77.0,
            region=(1, 2, 3, 4),
            source="objective_cache",
        ),
        lexicon=None,
        interval_s=1.2,
    )

    value = cache.refresh(321, now=12.5)
    snapshot = cache.snapshot()

    assert value == snapshot
    assert snapshot.normalized == "objective_321"
    assert snapshot.source == "objective_cache"


def test_objective_cache_background_poll_uses_latest_submitted_frame():
    runtime_objective_cache = pytest.importorskip("megabonk_bot.runtime.objective_cache")
    seen = []

    def _fake_read(frame, lexicon=None):
        seen.append(frame)
        return UiTextDetection(
            text=str(frame),
            normalized=f"objective_{frame}",
            confidence=70.0,
            source="objective_cache",
        )

    cache = runtime_objective_cache.ObjectiveUiCache(
        read_objective_ui=_fake_read,
        lexicon=None,
        interval_s=0.01,
    )
    cache.start()
    try:
        cache.submit(100)
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if cache.snapshot().normalized == "objective_100":
                break
            time.sleep(0.01)

        cache.submit(200)
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if cache.snapshot().normalized == "objective_200":
                break
            time.sleep(0.01)
    finally:
        cache.close()

    assert seen
    assert cache.snapshot().normalized == "objective_200"


def test_survival_only_runtime_uses_null_objective_cache():
    seen = []

    def _fake_read(frame, lexicon=None):
        seen.append((frame, lexicon))
        return UiTextDetection(normalized="should_not_run", source="objective_cache")

    cache = _build_objective_cache(
        survival_only=True,
        read_objective_ui=_fake_read,
        lexicon=None,
        interval_s=0.01,
    )

    assert isinstance(cache, NullObjectiveUiCache)
    cache.start()
    cache.submit(123)

    snapshot = cache.snapshot()

    assert snapshot == UiTextDetection()
    assert seen == []


def test_survival_only_runtime_uses_null_hud_cache():
    seen = []

    def _fake_read(frame, regions=None):
        seen.append((frame, regions))
        return {"time": 123}

    cache = _build_hud_cache(
        survival_only=True,
        read_hud_telemetry=_fake_read,
        regions={"REG_HUD_TIME": (1, 2, 3, 4)},
        interval_s=0.01,
    )

    assert isinstance(cache, NullHudTelemetryCache)
    cache.start()
    cache.submit(123)
    cache.set_regions({"REG_HUD_TIME": (10, 20, 30, 40)})

    snapshot = cache.snapshot()

    assert snapshot == {
        "debug_dumped": False,
        "hud_fail_streak": 0,
        "hud_ts": None,
    }
    assert seen == []


def test_survival_only_runtime_forces_map_scan_off():
    assert _resolve_allow_map_scan(
        survival_only=True,
        allow_map_scan_tab=True,
        max_enabled=True,
        explore_with_tab=True,
        current_scene_id="room_01",
    ) is False


def test_map_scan_is_disabled_in_final_boss_scene():
    assert _resolve_allow_map_scan(
        survival_only=False,
        allow_map_scan_tab=True,
        max_enabled=False,
        explore_with_tab=False,
        current_scene_id="FinalBossMap",
    ) is False


def test_survival_only_runtime_forces_upgrade_autopick_off():
    assert _should_auto_pick_upgrade(
        survival_only=True,
        auto_pick_upgrade_with_space=True,
    ) is False


def test_survival_only_runtime_forces_detection_flags_off():
    flags = _resolve_runtime_detection_flags(
        survival_only=True,
        minimap_enabled=True,
        projectiles_enabled=True,
        world_objects_enabled=True,
        memory_probe_enabled=True,
    )

    assert flags == {
        "minimap_enabled": False,
        "projectiles_enabled": False,
        "world_objects_enabled": False,
        "memory_probe_enabled": False,
    }


def test_non_survival_runtime_keeps_detection_flags():
    flags = _resolve_runtime_detection_flags(
        survival_only=False,
        minimap_enabled=True,
        projectiles_enabled=False,
        world_objects_enabled=True,
        memory_probe_enabled=False,
    )

    assert flags == {
        "minimap_enabled": True,
        "projectiles_enabled": False,
        "world_objects_enabled": True,
        "memory_probe_enabled": False,
    }


def test_survival_only_runtime_forces_movement_flags_off():
    flags = _resolve_runtime_movement_flags(
        survival_only=True,
        bunny_hop_enabled=True,
        sliding_enabled=True,
    )

    assert flags == {
        "bunny_hop_enabled": False,
        "sliding_enabled": False,
    }


def test_non_survival_runtime_keeps_movement_flags():
    flags = _resolve_runtime_movement_flags(
        survival_only=False,
        bunny_hop_enabled=True,
        sliding_enabled=False,
    )

    assert flags == {
        "bunny_hop_enabled": True,
        "sliding_enabled": False,
    }


def test_survival_only_runtime_uses_null_world_probe_even_when_enabled():
    probe = _build_world_probe(
        survival_only=True,
        enabled=True,
        window_title="Megabonk",
        poll_interval_s=0.25,
        signatures_path=Path("config/memory_signatures.json"),
    )

    snapshot = probe.sample(now_ts=123.0)

    assert snapshot.status == "disabled"
    assert snapshot.source == "external_memory"


def test_survival_mvp_profile_resolves_to_survival_only_runtime_guards():
    config = load_config(Path("config/survival_mvp.yaml"))

    detection_flags = _resolve_runtime_detection_flags(
        survival_only=bool(config["mvp_policy"]["survival_only"]),
        minimap_enabled=bool(config["detection"]["minimap_enabled"]),
        projectiles_enabled=bool(config["detection"]["projectiles_enabled"]),
        world_objects_enabled=bool(config["detection"]["world_objects_enabled"]),
        memory_probe_enabled=bool(config["detection"]["memory_probe_enabled"]),
    )
    movement_flags = _resolve_runtime_movement_flags(
        survival_only=bool(config["mvp_policy"]["survival_only"]),
        bunny_hop_enabled=bool(config["max_policy"]["bunny_hop_enabled"]),
        sliding_enabled=bool(config["max_policy"]["sliding_enabled"]),
    )

    assert _resolve_allow_map_scan(
        survival_only=bool(config["mvp_policy"]["survival_only"]),
        allow_map_scan_tab=bool(config["mvp_policy"]["allow_map_scan_tab"]),
        max_enabled=bool(config["max_policy"]["enabled"]),
        explore_with_tab=bool(config["max_policy"]["explore_with_tab"]),
        current_scene_id="room_01",
    ) is False
    assert _should_auto_pick_upgrade(
        survival_only=bool(config["mvp_policy"]["survival_only"]),
        auto_pick_upgrade_with_space=bool(config["mvp_policy"]["auto_pick_upgrade_with_space"]),
    ) is False
    assert detection_flags == {
        "minimap_enabled": False,
        "projectiles_enabled": False,
        "world_objects_enabled": False,
        "memory_probe_enabled": False,
    }
    assert movement_flags == {
        "bunny_hop_enabled": False,
        "sliding_enabled": False,
    }


def test_non_survival_runtime_respects_upgrade_autopick_flag():
    assert _should_auto_pick_upgrade(
        survival_only=False,
        auto_pick_upgrade_with_space=True,
    ) is True


def test_build_screen_detector_applies_runtime_template_threshold_overrides():
    detector = _build_screen_detector(
        templates={"tpl_dead": "dead"},
        regions={"REG_DEAD": (0, 0, 10, 10)},
        autopilot_cfg={"template_thresholds": {"dead_hard": 0.77}},
    )

    assert detector.template_thresholds["dead_hard"] == 0.77
    assert detector.template_thresholds["running_minimap"] == 0.55


def test_run_runtime_bot_no_longer_imports_legacy_autopilot():
    module = ast.parse(Path("run_runtime_bot.py").read_text(encoding="utf-8"))

    has_legacy_import = any(
        isinstance(node, ast.ImportFrom) and node.module == "autopilot"
        for node in ast.walk(module)
    )
    has_legacy_constructor = any(
        isinstance(node, ast.Name) and node.id == "AutoPilot"
        for node in ast.walk(module)
    )

    assert has_legacy_import is False
    assert has_legacy_constructor is False
