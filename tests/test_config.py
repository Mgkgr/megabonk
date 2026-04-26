import pytest

from megabonk_bot.config import load_config
from run_runtime_bot import _resolve_optional_path, _resolve_project_base_dir


def test_load_config_defaults():
    config = load_config(None)
    assert config["runtime"]["step_hz"] == 12
    assert config["runtime"]["capture_backend"] == "auto"
    assert config["runtime"]["capture_log_errors"] is True
    assert config["runtime"]["hud_ocr_every_s"] == 0.8
    assert config["runtime"]["objective_ocr_every_s"] == 1.2
    assert config["runtime"]["hud_debug_save_policy"] == "on_fail_change"
    assert config["runtime"]["hud_debug_min_interval_s"] == 15.0
    assert config["runtime"]["event_schema_version"] == "runtime_events_v4"
    assert config["runtime"]["window_focus_interval_s"] == 0.25
    assert config["runtime"]["event_log_interval_s"] == 0.2
    assert config["mvp_policy"]["map_scan_interval_ticks"] == 180
    assert config["detection"]["asset_refs_dir"] == "art_refs/megabonk_unity_extracts"
    assert config["detection"]["enemy_catalog_path"] == "config/enemy_catalog.json"
    assert config["detection"]["world_catalog_path"] == "config/world_catalog.json"
    assert config["detection"]["projectile_catalog_path"] == "config/projectile_catalog.json"
    assert config["detection"]["ocr_lexicon_path"] == "config/ocr_lexicon.json"
    assert config["detection"]["memory_signatures_path"] == ""
    assert config["detection"]["minimap_enabled"] is True
    assert config["detection"]["memory_probe_enabled"] is True
    assert config["detection"]["memory_poll_interval_s"] == 0.25
    assert config["detection"]["enemy_classifier_mode"] == "hybrid"
    assert config["navigation"]["profile"] == "cautious"
    assert config["navigation"]["lane_count"] == 5
    assert config["navigation"]["memory_required_for_slide"] is True
    assert config["autopilot"]["click_cooldown_s"] == 0.5
    assert config["autopilot"]["heuristic"]["stuck_frames_required"] == 6


def test_load_config_yaml_deep_merge(tmp_path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "profile.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  step_hz: 20",
                "detection:",
                "  interact_threshold: 0.9",
                "autopilot:",
                "  heuristic:",
                "    jump_cooldown: 44",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    assert config["runtime"]["step_hz"] == 20
    assert config["runtime"]["event_log_interval_s"] == 0.2
    assert config["detection"]["interact_threshold"] == 0.9
    assert config["detection"]["grid_rows"] == 12
    assert config["autopilot"]["heuristic"]["jump_cooldown"] == 44


def test_load_config_yaml_navigation_override(tmp_path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "navigation.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "navigation:",
                "  profile: balanced",
                "  lane_count: 7",
                "  memory_required_for_slide: false",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    assert config["navigation"]["profile"] == "balanced"
    assert config["navigation"]["lane_count"] == 7
    assert config["navigation"]["memory_required_for_slide"] is False


def test_load_config_rejects_invalid_types(tmp_path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "invalid.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  step_hz: fast",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime.step_hz"):
        load_config(cfg_path)


def test_load_config_rejects_invalid_capture_backend(tmp_path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "invalid_capture_backend.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  capture_backend: desktop_duplication",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime.capture_backend"):
        load_config(cfg_path)


def test_load_config_rejects_invalid_navigation_profile(tmp_path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "invalid_navigation.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "navigation:",
                "  profile: unsafe",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="navigation.profile"):
        load_config(cfg_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stuck_frames_required", 9),
        ("jump_cooldown", 42),
        ("scan_interval", 75),
    ],
)
def test_load_config_autopilot_heuristic_critical_fields(tmp_path, field, value):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "autopilot.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "autopilot:",
                "  heuristic:",
                f"    {field}: {value}",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    assert config["autopilot"]["heuristic"][field] == value


def test_runtime_resource_paths_resolve_from_project_root_with_explicit_config(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "bot_profile.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  templates_dir: templates",
                "detection:",
                "  asset_refs_dir: art_refs/megabonk_unity_extracts",
                "  enemy_catalog_path: config/enemy_catalog.json",
                "  world_catalog_path: config/world_catalog.json",
                "  projectile_catalog_path: config/projectile_catalog.json",
                "  ocr_lexicon_path: config/ocr_lexicon.json",
                "  memory_signatures_path: config/memory_signatures.json",
            ]
        ),
        encoding="utf-8",
    )
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    config = load_config(config_path)
    base_dir = _resolve_project_base_dir(config_path)

    assert base_dir == repo_root.resolve()
    assert _resolve_optional_path(config["runtime"]["templates_dir"], base_dir=base_dir) == (
        repo_root / "templates"
    ).resolve()
    assert _resolve_optional_path(config["detection"]["asset_refs_dir"], base_dir=base_dir) == (
        repo_root / "art_refs" / "megabonk_unity_extracts"
    ).resolve()
    assert _resolve_optional_path(config["detection"]["enemy_catalog_path"], base_dir=base_dir) == (
        repo_root / "config" / "enemy_catalog.json"
    ).resolve()
    assert _resolve_optional_path(config["detection"]["world_catalog_path"], base_dir=base_dir) == (
        repo_root / "config" / "world_catalog.json"
    ).resolve()
    assert _resolve_optional_path(
        config["detection"]["projectile_catalog_path"],
        base_dir=base_dir,
    ) == (repo_root / "config" / "projectile_catalog.json").resolve()
    assert _resolve_optional_path(config["detection"]["ocr_lexicon_path"], base_dir=base_dir) == (
        repo_root / "config" / "ocr_lexicon.json"
    ).resolve()
    assert _resolve_optional_path(
        config["detection"]["memory_signatures_path"],
        base_dir=base_dir,
    ) == (repo_root / "config" / "memory_signatures.json").resolve()


def test_runtime_cli_template_override_resolves_from_project_root_with_explicit_config(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "bot_profile.yaml"
    config_path.write_text("runtime:\n  step_hz: 12\n", encoding="utf-8")
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    base_dir = _resolve_project_base_dir(config_path)

    assert _resolve_optional_path("templates/custom", base_dir=base_dir) == (
        repo_root / "templates" / "custom"
    ).resolve()
