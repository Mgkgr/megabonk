import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from vision import find_in_region, load_templates, match_template, normalize_region  # noqa: E402


def _make_gradient_image(width, height):
    base = np.arange(width * height, dtype=np.uint8).reshape((height, width))
    return cv2.merge([base, base, base])


def test_match_template_returns_best_score_and_location():
    frame = _make_gradient_image(8, 8)
    tpl = frame[2:5, 3:6]
    score, (x, y) = match_template(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY),
    )

    assert score == 1.0
    assert (x, y) == (3, 2)


def test_find_in_region_hits_template_center():
    frame = _make_gradient_image(10, 10)
    tpl = frame[4:7, 1:4]
    found, (cx, cy), score = find_in_region(frame, tpl, (0, 0, 10, 10), threshold=0.99)

    assert found is True
    assert score == 1.0
    assert (cx, cy) == (2, 5)


def test_find_in_region_handles_empty_roi():
    frame = _make_gradient_image(5, 5)
    tpl = frame[0:2, 0:2]
    found, center, score = find_in_region(frame, tpl, (10, 10, 0, 0), threshold=0.5)

    assert found is False
    assert center == (0, 0)
    assert score == 0.0


def test_normalize_region_scales_relative_values():
    assert normalize_region(200, 100, 0.1, 0.2, 0.3, 0.4) == (20, 20, 60, 40)


def test_load_templates_reads_png_files(tmp_path):
    img = _make_gradient_image(4, 4)
    file_path = tmp_path / "sample.png"
    cv2.imwrite(str(file_path), img)

    templates = load_templates(tmp_path)

    assert "sample" in templates
    assert templates["sample"].shape == img.shape
