import ast
from pathlib import Path

import numpy as np

from megabonk_bot.runtime.upgrade_dialog import is_upgrade_dialog


def test_is_upgrade_dialog_returns_true_when_any_upgrade_template_matches(monkeypatch):
    calls = []

    def _fake_find_in_region(frame, template, region, threshold):
        calls.append((frame, template, region, threshold))
        return (template == "katana_template", None, None)

    import megabonk_bot.vision as vision

    monkeypatch.setattr(vision, "find_in_region", _fake_find_in_region)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    templates = {"tpl_katana": "katana_template", "tpl_blood_tome": "blood_tome_template"}
    regions = {"REG_CHEST": (1, 2, 3, 4)}

    assert is_upgrade_dialog(frame, templates, regions) is True
    assert calls[0][1] == "katana_template"


def test_is_upgrade_dialog_returns_false_without_chest_region():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    assert is_upgrade_dialog(frame, {"tpl_katana": "katana_template"}, {}) is False


def test_run_runtime_bot_uses_runtime_upgrade_dialog_helper():
    module = ast.parse(Path("run_runtime_bot.py").read_text(encoding="utf-8"))

    has_local_helper = any(
        isinstance(node, ast.FunctionDef) and node.name == "_is_upgrade_dialog"
        for node in ast.walk(module)
    )
    uses_runtime_helper = any(
        isinstance(node, ast.Name) and node.id == "_runtime_is_upgrade_dialog"
        for node in ast.walk(module)
    )

    assert has_local_helper is False
    assert uses_runtime_helper is True
