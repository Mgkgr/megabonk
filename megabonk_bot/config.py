from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HotkeysConfig:
    enabled: bool = True
    toggle_vk: int = 0x77
    panic_vk: int = 0x7B


@dataclass(frozen=True)
class RuntimeConfig:
    state: str = "OFF"
    step_hz: int = 12
    window_title: str = "Megabonk"
    capture_backend: str = "auto"
    window_focus_interval_s: float = 0.25
    templates_dir: str = "templates"
    overlay_enabled: bool = True
    overlay_window: str = "Megabonk Runtime Bot"
    overlay_topmost: bool = True
    overlay_transparent: bool = True
    event_log_path: str = "logs/runtime_events.jsonl"
    event_log_interval_s: float = 0.2
    upgrade_space_cooldown_s: float = 0.3
    cam_yaw_pixels: int = 160
    restart_cooldown_s: float = 3.5
    restart_hold_s: float = 3.5
    restart_wait_timeout_s: float = 8.0
    restart_max_attempts: int = 2


@dataclass(frozen=True)
class DetectionConfig:
    grid_rows: int = 12
    grid_cols: int = 20
    enemy_hsv_lower: list[int] = field(default_factory=lambda: [45, 80, 40])
    enemy_hsv_upper: list[int] = field(default_factory=lambda: [85, 255, 255])
    enemy_min_area: float = 1200.0
    interact_threshold: float = 0.65
    use_onnx: bool = False
    onnx_model_path: str = ""
    scene_memory_ttl_s: float = 2.0


@dataclass(frozen=True)
class MvpPolicyConfig:
    chest_policy: str = "never_open"
    auto_pick_upgrade_with_space: bool = True
    user_picks_character_manually: bool = True
    allow_map_scan_tab: bool = False
    map_scan_interval_ticks: int = 180


@dataclass(frozen=True)
class MaxPolicyConfig:
    enabled: bool = False
    explore_with_tab: bool = True
    threat_scoring: bool = True
    bunny_hop_enabled: bool = True
    sliding_enabled: bool = True
    collect_shrines_and_statues: bool = True


@dataclass(frozen=True)
class HeuristicAutoPilotConfig:
    enemy_hsv_lower: list[int] = field(default_factory=lambda: [45, 80, 40])
    enemy_hsv_upper: list[int] = field(default_factory=lambda: [85, 255, 255])
    coin_hsv_lower: list[int] = field(default_factory=lambda: [18, 120, 80])
    coin_hsv_upper: list[int] = field(default_factory=lambda: [35, 255, 255])
    enemy_area_threshold: float = 1400.0
    coin_area_threshold: float = 900.0
    center_roi: list[float] = field(default_factory=lambda: [0.35, 0.38, 0.30, 0.36])
    center_lower_roi: list[float] = field(default_factory=lambda: [0.35, 0.55, 0.30, 0.35])
    stuck_diff_threshold: float = 3.0
    stuck_frames_required: int = 6
    stuck_escape_ticks: int = 16
    jump_cooldown: int = 30
    slide_cooldown: int = 24
    enemy_close_multiplier: float = 1.6
    scan_interval: int = 60
    scan_duration: int = 8
    scan_decision_ticks: int = 10


@dataclass(frozen=True)
class AutoPilotConfig:
    click_cooldown_s: float = 0.5
    template_thresholds: dict[str, float] = field(default_factory=dict)
    heuristic: HeuristicAutoPilotConfig = field(default_factory=HeuristicAutoPilotConfig)


@dataclass(frozen=True)
class BotConfig:
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    mvp_policy: MvpPolicyConfig = field(default_factory=MvpPolicyConfig)
    max_policy: MaxPolicyConfig = field(default_factory=MaxPolicyConfig)
    autopilot: AutoPilotConfig = field(default_factory=AutoPilotConfig)
    item_priorities: list[str] = field(default_factory=lambda: ["blood_tome", "katana"])
    boss_schedule: list[dict[str, Any]] = field(default_factory=list)


DEFAULT_CONFIG = BotConfig()


def default_config_dict() -> dict[str, Any]:
    return asdict(DEFAULT_CONFIG)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _must_be_type(name: str, value: Any, expected_type: type | tuple[type, ...]) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} must be {expected_type}, got {type(value)}")


def _must_be_non_negative(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _must_be_positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_hsv_triplet(name: str, value: Any) -> None:
    _must_be_type(name, value, list)
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 items")
    for idx, component in enumerate(value):
        _must_be_type(f"{name}[{idx}]", component, int)
        if component < 0 or component > 255:
            raise ValueError(f"{name}[{idx}] must be in [0, 255]")


def _validate_float_quad(name: str, value: Any) -> None:
    _must_be_type(name, value, list)
    if len(value) != 4:
        raise ValueError(f"{name} must contain exactly 4 items")
    for idx, component in enumerate(value):
        _must_be_type(f"{name}[{idx}]", component, (int, float))
        if not 0.0 <= float(component) <= 1.0:
            raise ValueError(f"{name}[{idx}] must be in [0.0, 1.0]")


def _validate_config(data: dict[str, Any]) -> None:
    allowed_root = {
        "hotkeys",
        "runtime",
        "detection",
        "mvp_policy",
        "max_policy",
        "autopilot",
        "item_priorities",
        "boss_schedule",
    }
    unknown_root = set(data.keys()) - allowed_root
    if unknown_root:
        raise ValueError(f"Unknown config keys: {sorted(unknown_root)}")

    runtime = data["runtime"]
    detection = data["detection"]
    mvp_policy = data["mvp_policy"]
    hotkeys = data["hotkeys"]
    autopilot = data["autopilot"]
    heuristic = autopilot["heuristic"]

    _must_be_type("runtime.step_hz", runtime["step_hz"], int)
    _must_be_positive("runtime.step_hz", runtime["step_hz"])
    _must_be_type("runtime.cam_yaw_pixels", runtime["cam_yaw_pixels"], int)
    _must_be_non_negative("runtime.cam_yaw_pixels", runtime["cam_yaw_pixels"])
    _must_be_type("runtime.restart_max_attempts", runtime["restart_max_attempts"], int)
    _must_be_positive("runtime.restart_max_attempts", runtime["restart_max_attempts"])

    for seconds_key in (
        "event_log_interval_s",
        "window_focus_interval_s",
        "upgrade_space_cooldown_s",
        "restart_cooldown_s",
        "restart_hold_s",
        "restart_wait_timeout_s",
    ):
        _must_be_type(f"runtime.{seconds_key}", runtime[seconds_key], (int, float))
        _must_be_non_negative(f"runtime.{seconds_key}", float(runtime[seconds_key]))

    _must_be_type("runtime.state", runtime["state"], str)
    if runtime["state"].upper() not in {"OFF", "ACTIVE", "PANIC", "RECOVERY"}:
        raise ValueError("runtime.state must be one of OFF|ACTIVE|PANIC|RECOVERY")
    _must_be_type("runtime.capture_backend", runtime["capture_backend"], str)
    if str(runtime["capture_backend"]).lower() not in {"auto", "printwindow", "mss"}:
        raise ValueError("runtime.capture_backend must be one of auto|printwindow|mss")

    _must_be_type("detection.grid_rows", detection["grid_rows"], int)
    _must_be_positive("detection.grid_rows", detection["grid_rows"])
    _must_be_type("detection.grid_cols", detection["grid_cols"], int)
    _must_be_positive("detection.grid_cols", detection["grid_cols"])
    _must_be_type("detection.enemy_min_area", detection["enemy_min_area"], (int, float))
    _must_be_non_negative("detection.enemy_min_area", float(detection["enemy_min_area"]))
    _must_be_type("detection.interact_threshold", detection["interact_threshold"], (int, float))
    if not 0.0 <= float(detection["interact_threshold"]) <= 1.0:
        raise ValueError("detection.interact_threshold must be in [0.0, 1.0]")
    _must_be_type("detection.scene_memory_ttl_s", detection["scene_memory_ttl_s"], (int, float))
    _must_be_non_negative("detection.scene_memory_ttl_s", float(detection["scene_memory_ttl_s"]))
    _validate_hsv_triplet("detection.enemy_hsv_lower", detection["enemy_hsv_lower"])
    _validate_hsv_triplet("detection.enemy_hsv_upper", detection["enemy_hsv_upper"])

    _must_be_type("mvp_policy.map_scan_interval_ticks", mvp_policy["map_scan_interval_ticks"], int)
    _must_be_positive("mvp_policy.map_scan_interval_ticks", mvp_policy["map_scan_interval_ticks"])

    _must_be_type("autopilot.click_cooldown_s", autopilot["click_cooldown_s"], (int, float))
    _must_be_non_negative("autopilot.click_cooldown_s", float(autopilot["click_cooldown_s"]))
    _must_be_type("autopilot.template_thresholds", autopilot["template_thresholds"], dict)
    for key, value in autopilot["template_thresholds"].items():
        _must_be_type(f"autopilot.template_thresholds.{key}", value, (int, float))
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"autopilot.template_thresholds.{key} must be in [0.0, 1.0]")

    _validate_hsv_triplet("autopilot.heuristic.enemy_hsv_lower", heuristic["enemy_hsv_lower"])
    _validate_hsv_triplet("autopilot.heuristic.enemy_hsv_upper", heuristic["enemy_hsv_upper"])
    _validate_hsv_triplet("autopilot.heuristic.coin_hsv_lower", heuristic["coin_hsv_lower"])
    _validate_hsv_triplet("autopilot.heuristic.coin_hsv_upper", heuristic["coin_hsv_upper"])
    _validate_float_quad("autopilot.heuristic.center_roi", heuristic["center_roi"])
    _validate_float_quad("autopilot.heuristic.center_lower_roi", heuristic["center_lower_roi"])
    for name in ("enemy_area_threshold", "coin_area_threshold", "stuck_diff_threshold", "enemy_close_multiplier"):
        _must_be_type(f"autopilot.heuristic.{name}", heuristic[name], (int, float))
        _must_be_non_negative(f"autopilot.heuristic.{name}", float(heuristic[name]))
    for name in (
        "stuck_frames_required",
        "stuck_escape_ticks",
        "jump_cooldown",
        "slide_cooldown",
        "scan_interval",
        "scan_duration",
        "scan_decision_ticks",
    ):
        _must_be_type(f"autopilot.heuristic.{name}", heuristic[name], int)
        _must_be_non_negative(f"autopilot.heuristic.{name}", heuristic[name])

    _must_be_type("hotkeys.toggle_vk", hotkeys["toggle_vk"], int)
    _must_be_type("hotkeys.panic_vk", hotkeys["panic_vk"], int)
    if not 0 <= hotkeys["toggle_vk"] <= 255:
        raise ValueError("hotkeys.toggle_vk must be in [0, 255]")
    if not 0 <= hotkeys["panic_vk"] <= 255:
        raise ValueError("hotkeys.panic_vk must be in [0, 255]")

    _must_be_type("item_priorities", data["item_priorities"], list)
    for idx, value in enumerate(data["item_priorities"]):
        _must_be_type(f"item_priorities[{idx}]", value, str)

    _must_be_type("boss_schedule", data["boss_schedule"], list)
    for idx, item in enumerate(data["boss_schedule"]):
        _must_be_type(f"boss_schedule[{idx}]", item, dict)
        if "spawn_s" in item:
            _must_be_type(f"boss_schedule[{idx}].spawn_s", item["spawn_s"], int)
            _must_be_non_negative(f"boss_schedule[{idx}].spawn_s", item["spawn_s"])
        if "prep_s" in item:
            _must_be_type(f"boss_schedule[{idx}].prep_s", item["prep_s"], int)
            _must_be_non_negative(f"boss_schedule[{idx}].prep_s", item["prep_s"])


def dump_default_config_yaml() -> str:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("Для печати YAML нужен пакет pyyaml (`pip install pyyaml`).") from exc
    return yaml.safe_dump(default_config_dict(), sort_keys=False, allow_unicode=True)


def load_config(config_path: Path | None) -> dict[str, Any]:
    profile = default_config_dict()
    if config_path is None:
        _validate_config(profile)
        return profile
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    raw = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("Для YAML-конфига нужен пакет pyyaml (`pip install pyyaml`).") from exc
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Config root must be object")
    merged = _deep_merge(profile, data)
    _validate_config(merged)
    return merged
