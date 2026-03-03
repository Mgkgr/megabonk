from collections import deque

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")


def test_step_reward_uses_latest_frame(monkeypatch):
    megabonk_env = pytest.importorskip("megabonk_env")
    env = megabonk_env.MegabonkEnv.__new__(megabonk_env.MegabonkEnv)

    frames = [
        np.full((4, 4, 3), 1, dtype=np.uint8),  # initial frame
        np.full((4, 4, 3), 2, dtype=np.uint8),  # frame_skip #1
        np.full((4, 4, 3), 3, dtype=np.uint8),  # frame_skip #2 (должен быть финальным)
    ]
    seen = {"loot_frame": None, "stuck_frame": None}

    env._grab_frame = lambda: frames.pop(0)
    env._prof_add = lambda *_args, **_kwargs: None
    env._finish_step_profile = lambda: None
    env._submit_hud_frame = lambda frame: None
    env._get_cached_hud_time = lambda: None
    env._get_cached_hp_ratio = lambda: None
    env._get_cached_lvl = lambda: None
    env._get_cached_kills = lambda: None
    env._get_cached_hud_debug = lambda: {}
    env._focus_game_window = lambda: None
    env._apply_cam_yaw = lambda yaw: None
    env._apply_cam_pitch = lambda pitch: None
    env._current_safety_strength = lambda: 0.0
    env._apply_safety_override = (
        lambda dir_id, yaw, pitch, jump, slide, danger_now, stuck_now: (
            dir_id,
            yaw,
            pitch,
            jump,
            slide,
            None,
            0.0,
        )
    )
    env._to_gray84 = lambda _frame: np.zeros((84, 84), dtype=np.uint8)
    env._get_obs = lambda: np.zeros((84, 84, 4), dtype=np.uint8)
    env._restart_after_death = lambda **_kwargs: None
    env._log_event = lambda *_args, **_kwargs: None
    env._is_safety_stuck = lambda _frame, _forwardish: False

    enemy_lower = object()
    enemy_upper = object()
    coin_lower = object()
    coin_upper = object()
    env.reward_enemy_hsv_lower = enemy_lower
    env.reward_enemy_hsv_upper = enemy_upper
    env.reward_coin_hsv_lower = coin_lower
    env.reward_coin_hsv_upper = coin_upper

    def _find_center_area_in_roi(frame, lower, upper, roi_rel):
        value = int(frame[0, 0, 0])
        if lower is coin_lower:
            seen["loot_frame"] = value
            return (1, 1), 1.0, value / 10.0
        return None, 0.0, 0.0

    env._find_center_area_in_roi = _find_center_area_in_roi

    def _is_reward_stuck(frame, _forwardish):
        seen["stuck_frame"] = int(frame[0, 0, 0])
        return False

    env._is_reward_stuck = _is_reward_stuck

    env.debug_recognition = False
    env.include_cam_yaw = False
    env.include_cam_pitch = False
    env.use_heuristic_autopilot = False
    env.heuristic_pilot = None
    env.safety_danger_frac_threshold = 0.02
    env._safety_step_count = 0
    env._sticky_left = 0
    env._sticky_dir = 0
    env.sticky_steps_range = (1, 1)
    env.frame_skip_range = (2, 2)
    env.dt = 0.0
    env.jump_key = "space"
    env.slide_key = "shift"
    env._death_check_idx = 0
    env._last_dead_r_time = 0.0
    env.restart_cooldown_s = 9999.0
    env._death_like_streak = 0
    env.death_hysteresis_frames = 3
    env.frames = deque(maxlen=4)
    env._last_obs = None
    env.reward_danger_k = 1.0
    env.reward_loot_k = 1.0
    env.reward_stuck_k = 1.0
    env.reward_enemy_roi = (0.0, 0.0, 1.0, 1.0)
    env.reward_loot_roi = (0.0, 0.0, 1.0, 1.0)
    env.cap = None
    env._last_capture_error = None

    monkeypatch.setattr(megabonk_env, "set_move", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(megabonk_env, "tap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(megabonk_env.time, "sleep", lambda _dt: None)
    monkeypatch.setattr(megabonk_env, "_fast_death_check", lambda _gray: False)

    obs, reward, terminated, truncated, info = env.step((1, 0, 0))

    assert terminated is False
    assert truncated is False
    assert obs.shape == (84, 84, 4)
    assert seen["loot_frame"] == 3
    assert seen["stuck_frame"] == 3
    assert reward > 0.3
