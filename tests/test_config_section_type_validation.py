from copy import deepcopy

import pytest

from megabonk_bot.config import _validate_config, default_config_dict, load_config


def test_load_config_rejects_runtime_non_object(tmp_path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "runtime_bad.yaml"
    cfg_path.write_text("runtime: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime must be object"):
        load_config(cfg_path)


def test_load_config_rejects_autopilot_non_object(tmp_path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "autopilot_bad.yaml"
    cfg_path.write_text("autopilot: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="autopilot must be object"):
        load_config(cfg_path)


def test_validate_config_reports_missing_required_path():
    data = deepcopy(default_config_dict())
    del data["runtime"]["step_hz"]
    with pytest.raises(ValueError, match="Missing config key: runtime.step_hz"):
        _validate_config(data)
