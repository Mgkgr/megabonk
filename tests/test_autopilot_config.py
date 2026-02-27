import numpy as np
import pytest

from run_runtime_bot import _prepare_heuristic_config

pytest.importorskip("cv2")
_autopilot = pytest.importorskip("autopilot")
AutoPilot = _autopilot.AutoPilot
HeuristicAutoPilot = _autopilot.HeuristicAutoPilot


def test_autopilot_accepts_click_cooldown_and_threshold_overrides():
    pilot = AutoPilot(
        templates={},
        regions={},
        click_cooldown_s=1.25,
        template_thresholds={"main_play_detect": 0.73},
    )
    assert pilot.click_cooldown == 1.25
    assert pilot._thr("main_play_detect") == 0.73


def test_autopilot_supports_custom_click_handler():
    clicks = []

    def _click(x, y, delay):
        clicks.append((x, y, delay))

    pilot = AutoPilot(
        templates={},
        regions={},
        click_fn=_click,
    )
    ok = pilot.safe_click_if_found(True, (10, 20), score=0.9, thr=0.5)
    assert ok is True
    assert clicks == [(10, 20, 0.05)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stuck_frames_required", 11),
        ("jump_cooldown", 77),
        ("scan_interval", 123),
    ],
)
def test_heuristic_autopilot_critical_fields_are_overridable(field, value):
    pilot = HeuristicAutoPilot(config={field: value})
    assert getattr(pilot, field) == value


def test_prepare_heuristic_config_uses_detect_defaults_and_allows_overrides():
    detect_cfg = {
        "enemy_hsv_lower": [10, 20, 30],
        "enemy_hsv_upper": [40, 50, 60],
        "enemy_min_area": 777.0,
    }
    autopilot_cfg = {"heuristic": {"scan_interval": 88, "enemy_area_threshold": 999.0}}
    merged = _prepare_heuristic_config(detect_cfg, autopilot_cfg)
    assert merged["scan_interval"] == 88
    assert merged["enemy_area_threshold"] == 999.0
    assert merged["enemy_hsv_lower"] == (10, 20, 30)
    assert merged["enemy_hsv_upper"] == (40, 50, 60)


def test_heuristic_autopilot_config_hsv_applies():
    pilot = HeuristicAutoPilot(config={"enemy_hsv_lower": [1, 2, 3]})
    assert np.array_equal(pilot.enemy_hsv_lower, np.array([1, 2, 3], dtype=np.uint8))
