import pytest


class _FakeCap:
    def focus(self, topmost=True):
        return None

    def get_bbox(self):
        return {"left": 0, "top": 0, "width": 64, "height": 64}


def test_megabonk_env_passes_heuristic_config(monkeypatch):
    megabonk_env = pytest.importorskip("megabonk_env")
    captured = {}

    class _StubHeuristic:
        def __init__(self, config=None):
            captured["config"] = dict(config or {})

    monkeypatch.setattr(megabonk_env, "HeuristicAutoPilot", _StubHeuristic)
    env = megabonk_env.MegabonkEnv(
        cap=_FakeCap(),
        templates_dir=None,
        use_heuristic_autopilot=True,
        autopilot_config={"heuristic": {"scan_interval": 91, "jump_cooldown": 33}},
        hud_ocr_every_s=0.0,
    )
    try:
        assert env.heuristic_pilot is not None
        assert captured["config"]["scan_interval"] == 91
        assert captured["config"]["jump_cooldown"] == 33
    finally:
        env.close()
