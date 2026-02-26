import pytest

from megabonk_bot.config import load_config


def test_load_config_defaults():
    config = load_config(None)
    assert config["runtime"]["step_hz"] == 12
    assert config["runtime"]["event_log_interval_s"] == 0.2
    assert config["mvp_policy"]["map_scan_interval_ticks"] == 180


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
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    assert config["runtime"]["step_hz"] == 20
    assert config["runtime"]["event_log_interval_s"] == 0.2
    assert config["detection"]["interact_threshold"] == 0.9
    assert config["detection"]["grid_rows"] == 12


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
