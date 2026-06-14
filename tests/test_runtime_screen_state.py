import numpy as np

from megabonk_bot.runtime.screen_state import RuntimeScreenDetector


def test_runtime_screen_detector_uses_hard_dead_template_without_brightness_check():
    calls = []

    def _finder(_frame, template, _region, threshold):
        calls.append((template, threshold))
        if template == "dead":
            return True, (10, 20), threshold
        return False, (0, 0), 0.0

    detector = RuntimeScreenDetector(
        templates={"tpl_dead": "dead"},
        regions={"REG_DEAD": (0, 0, 10, 10)},
        finder=_finder,
        death_like_fn=lambda _frame: False,
    )

    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == "DEAD"
    assert calls == [("dead", 0.55)]


def test_runtime_screen_detector_requires_death_like_frame_for_soft_dead_match():
    def _finder(_frame, template, _region, threshold):
        if template == "dead" and threshold == 0.35:
            return True, (10, 20), threshold
        return False, (0, 0), 0.0

    detector = RuntimeScreenDetector(
        templates={"tpl_dead": "dead"},
        regions={"REG_DEAD": (0, 0, 10, 10)},
        finder=_finder,
        death_like_fn=lambda _frame: False,
    )

    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == "UNKNOWN"


def test_runtime_screen_detector_falls_back_to_running_minimap():
    def _finder(_frame, template, _region, threshold):
        if template == "minimap":
            return True, (10, 20), threshold
        return False, (0, 0), 0.0

    detector = RuntimeScreenDetector(
        templates={"tpl_minimap": "minimap"},
        regions={"REG_MINIMAP": (0, 0, 10, 10)},
        finder=_finder,
    )

    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == "RUNNING"
