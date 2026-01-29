from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from megabonk_bot.templates import load_templates  # noqa: E402


def test_load_templates_loads_png(tmp_path: Path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    path = tmp_path / "tpl_test.png"
    cv2.imwrite(str(path), img)

    templates = load_templates(tmp_path)

    assert "tpl_test" in templates
    assert templates["tpl_test"].shape == (10, 10, 3)
