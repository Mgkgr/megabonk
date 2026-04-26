from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from typing import Any

import cv2
import numpy as np

from megabonk_bot.runtime_logic import BotAction, BotMode, SceneSnapshot


RISKY_TERRAIN_KINDS = {"pit", "edge", "blocked"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _coerce_lane_count(value: int) -> int:
    lane_count = max(3, int(value))
    if lane_count % 2 == 0:
        lane_count += 1
    return lane_count


def _lane_labels(lane_count: int) -> tuple[str, ...]:
    if lane_count == 3:
        return ("left", "center", "right")
    if lane_count == 5:
        return ("left_far", "left", "center", "right", "right_far")
    center = lane_count // 2
    labels = []
    for index in range(lane_count):
        if index == center:
            labels.append("center")
            continue
        offset = abs(index - center)
        side = "left" if index < center else "right"
        labels.append(side if offset == 1 else f"{side}_{offset}")
    return tuple(labels)


def _lane_side(label: str) -> str:
    lowered = str(label).lower()
    if lowered.startswith("left"):
        return "left"
    if lowered.startswith("right"):
        return "right"
    return "center"


@dataclass(frozen=True)
class NavigationLane:
    index: int
    label: str
    x0: int
    x1: int
    threat_score: float
    obstacle_cost: float
    drop_risk: float
    clearance: float
    landing_clearance: float
    terrain_kind: str
    total_cost: float


@dataclass(frozen=True)
class NavigationContext:
    lanes: tuple[NavigationLane, ...] = ()
    drop_risk: float = 0.0
    obstacle_cost: float = 0.0
    clearance: float = 0.0
    terrain_kind: str = "unknown"
    nav_confidence: float = 0.0
    source: str = "cv"
    slope_source: str = "none"
    escape_lane: str | None = None
    jump_gate: str = "not_evaluated"
    slide_gate: str = "not_evaluated"
    slope_delta_z: float | None = None


class StatefulNavigationPlanner:
    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        allow_bunny_hop: bool = True,
        sliding_enabled: bool = True,
        jump_cooldown: int = 30,
        slide_cooldown: int = 24,
        stuck_diff_threshold: float = 3.0,
        stuck_frames_required: int = 6,
        stuck_escape_ticks: int = 16,
    ):
        cfg = dict(config or {})
        self.profile = str(cfg.get("profile", "cautious")).strip().lower() or "cautious"
        self.lane_count = _coerce_lane_count(int(cfg.get("lane_count", 5)))
        self.drop_risk_threshold = _clamp(float(cfg.get("drop_risk_threshold", 0.58)))
        self.downhill_z_threshold = max(0.001, float(cfg.get("downhill_z_threshold", 0.18)))
        self.memory_required_for_slide = bool(cfg.get("memory_required_for_slide", True))
        self.allow_bunny_hop = bool(allow_bunny_hop)
        self.sliding_enabled = bool(sliding_enabled)
        self.jump_cooldown = max(1, int(jump_cooldown))
        self.slide_cooldown = max(1, int(slide_cooldown))
        self.stuck_diff_threshold = float(stuck_diff_threshold)
        self.stuck_frames_required = max(1, int(stuck_frames_required))
        self.stuck_escape_ticks = max(1, int(stuck_escape_ticks))
        self._lane_labels = _lane_labels(self.lane_count)
        self._tick = 0
        self._last_jump_tick = -9999
        self._last_bunny_tick = -9999
        self._last_slide_tick = -9999
        self._z_history: deque[float] = deque(maxlen=6)
        self._last_room_id: str | None = None
        self._last_gray = None
        self._last_forwardish = False
        self._stuck_frames = 0
        self._stuck_escape_timer = 0
        self._stuck_side = "left"

    def reset(self) -> None:
        self._tick = 0
        self._last_jump_tick = -9999
        self._last_bunny_tick = -9999
        self._last_slide_tick = -9999
        self._z_history.clear()
        self._last_room_id = None
        self._last_gray = None
        self._last_forwardish = False
        self._stuck_frames = 0
        self._stuck_escape_timer = 0
        self._stuck_side = "left"

    def evaluate(
        self,
        frame_bgr,
        snapshot: SceneSnapshot,
        *,
        mode: BotMode,
        threats: list[Any] | None = None,
        allow_map_scan: bool = False,
        map_scan_now: bool = False,
    ) -> tuple[BotAction, NavigationContext]:
        self._tick += 1
        navigation = self.build_context(frame_bgr, snapshot, threats=threats)
        press_tab = bool(allow_map_scan and map_scan_now)
        action = self._base_action(snapshot, mode=mode, press_tab=press_tab)
        if action is None:
            action, navigation = self._plan_active_action(
                snapshot,
                navigation,
                press_tab=press_tab,
                frame_bgr=frame_bgr,
            )
        self._last_forwardish = action.dir_id in {1, 5, 6}
        if action.jump:
            self._last_jump_tick = self._tick
        if action.slide:
            self._last_slide_tick = self._tick
        if action.reason == "bunny_hop_safe":
            self._last_bunny_tick = self._tick
        return action, navigation

    def build_context(
        self,
        frame_bgr,
        snapshot: SceneSnapshot,
        *,
        threats: list[Any] | None = None,
    ) -> NavigationContext:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return NavigationContext(source="memory" if snapshot.player_pose.world_pos is not None else "cv")
        frame_h, frame_w = frame_bgr.shape[:2]
        nav_y0 = int(frame_h * 0.55)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        nav_edges = cv2.Canny(gray[nav_y0:frame_h, :], 60, 180) if nav_y0 < frame_h else gray[0:0, 0:0]
        slope_kind, slope_delta_z, slope_source = self._estimate_slope(snapshot)
        room_edge_risk = self._estimate_room_edge_risk(snapshot)
        lane_bounds = self._lane_bounds(frame_w)
        grid_cells = tuple(self._iter_grid_cells(snapshot, nav_y0))
        lanes: list[NavigationLane] = []
        for index, (x0, x1) in enumerate(lane_bounds):
            metrics = self._lane_metrics(
                gray,
                nav_edges,
                grid_cells=grid_cells,
                x0=x0,
                x1=x1,
                nav_y0=nav_y0,
                frame_h=frame_h,
                room_edge_risk=room_edge_risk,
                threat_items=threats,
                snapshot=snapshot,
            )
            lane_terrain = slope_kind if slope_kind in {"flat", "uphill", "downhill"} else "unknown"
            if metrics["drop_risk"] >= max(self.drop_risk_threshold, 0.72):
                lane_terrain = "pit" if metrics["landing_clearance"] < 0.40 else "edge"
            elif metrics["obstacle_cost"] >= 0.72 and metrics["clearance"] < 0.42:
                lane_terrain = "blocked"
            total_cost = (
                metrics["threat_score"]
                + metrics["obstacle_cost"]
                + metrics["drop_risk"]
                - metrics["clearance"]
            )
            lanes.append(
                NavigationLane(
                    index=index,
                    label=self._lane_labels[index],
                    x0=x0,
                    x1=x1,
                    threat_score=metrics["threat_score"],
                    obstacle_cost=metrics["obstacle_cost"],
                    drop_risk=metrics["drop_risk"],
                    clearance=metrics["clearance"],
                    landing_clearance=metrics["landing_clearance"],
                    terrain_kind=lane_terrain,
                    total_cost=float(total_cost),
                )
            )
        center_lane = lanes[len(lanes) // 2]
        best_lane = min(lanes, key=lambda lane: (lane.total_cost, -lane.clearance))
        nav_confidence = _clamp(
            np.mean([max(0.0, lane.clearance - lane.drop_risk * 0.5) for lane in lanes]).item()
            if lanes
            else 0.0
        )
        if slope_source == "memory":
            nav_confidence = _clamp(nav_confidence + 0.15)
        source = "hybrid" if slope_source == "memory" else "cv"
        return NavigationContext(
            lanes=tuple(lanes),
            drop_risk=center_lane.drop_risk,
            obstacle_cost=center_lane.obstacle_cost,
            clearance=center_lane.clearance,
            terrain_kind=center_lane.terrain_kind,
            nav_confidence=nav_confidence,
            source=source,
            slope_source=slope_source,
            escape_lane=best_lane.label,
            slope_delta_z=slope_delta_z,
        )

    def _base_action(
        self,
        snapshot: SceneSnapshot,
        *,
        mode: BotMode,
        press_tab: bool,
    ) -> BotAction | None:
        if mode == BotMode.PANIC:
            return BotAction(press_tab=press_tab, reason="panic")
        if mode == BotMode.OFF:
            return BotAction(press_tab=press_tab, reason="off")
        if snapshot.is_dead or mode == BotMode.RECOVERY:
            return BotAction(press_r=True, press_tab=press_tab, reason="recovery_restart")
        if snapshot.is_upgrade:
            return BotAction(press_space=True, press_tab=press_tab, reason="upgrade_random_space")
        return None

    def _plan_active_action(
        self,
        snapshot: SceneSnapshot,
        navigation: NavigationContext,
        *,
        press_tab: bool,
        frame_bgr,
    ) -> tuple[BotAction, NavigationContext]:
        lanes = list(navigation.lanes)
        if not lanes:
            return BotAction(press_tab=press_tab, reason="advance_safe"), navigation
        center_idx = len(lanes) // 2
        center_lane = lanes[center_idx]
        safe_lanes = [
            lane
            for lane in lanes
            if lane.drop_risk < self.drop_risk_threshold
            and lane.clearance >= 0.35
            and lane.terrain_kind not in RISKY_TERRAIN_KINDS
        ]
        best_lane = min(
            safe_lanes or lanes,
            key=lambda lane: (lane.total_cost, lane.drop_risk, -lane.clearance),
        )
        stuck = self._is_stuck(frame_bgr)
        jump_allowed, jump_gate = self._jump_gate(center_lane, navigation)
        slide_allowed, slide_gate = self._slide_gate(best_lane, navigation)
        action: BotAction
        if center_lane.drop_risk >= self.drop_risk_threshold and best_lane.index != center_idx:
            action = self._lane_move_action(best_lane, reason="hold_drop_risk", press_tab=press_tab, force_strafe=True)
        elif stuck:
            action = self._unstuck_action(
                center_lane,
                best_lane,
                press_tab=press_tab,
                jump_allowed=jump_allowed,
            )
        elif self._escape_required(center_lane, best_lane):
            reason = f"escape_lane_{_lane_side(best_lane.label)}"
            action = self._lane_move_action(best_lane, reason=reason, press_tab=press_tab, force_strafe=True)
        elif (
            slide_allowed
            and center_lane.drop_risk < self.drop_risk_threshold * 0.75
            and center_lane.landing_clearance >= 0.60
        ):
            action = BotAction(dir_id=1, yaw=1, jump=0, slide=1, press_tab=press_tab, reason="slide_downhill")
            slide_gate = "allowed_slide_downhill"
        elif jump_allowed:
            action = BotAction(dir_id=1, yaw=1, jump=1, slide=0, press_tab=press_tab, reason="jump_obstacle")
            jump_gate = "allowed_jump_obstacle"
        elif self._bunny_hop_allowed(center_lane, navigation):
            action = BotAction(dir_id=1, yaw=1, jump=1, slide=0, press_tab=press_tab, reason="bunny_hop_safe")
            jump_gate = "allowed_bunny_hop"
        else:
            action = self._lane_move_action(best_lane, reason="advance_safe", press_tab=press_tab)
        return action, replace(
            navigation,
            escape_lane=best_lane.label,
            jump_gate=jump_gate,
            slide_gate=slide_gate,
        )

    def _escape_required(self, center_lane: NavigationLane, best_lane: NavigationLane) -> bool:
        if best_lane.index == center_lane.index:
            return False
        return (
            center_lane.threat_score >= self._profile_limits()["threat_escape"]
            or center_lane.obstacle_cost >= 0.62
            or center_lane.drop_risk >= self.drop_risk_threshold * 0.9
        )

    def _jump_gate(self, center_lane: NavigationLane, navigation: NavigationContext) -> tuple[bool, str]:
        if center_lane.drop_risk >= self.drop_risk_threshold * 0.85:
            return False, "blocked_by_drop_risk"
        if center_lane.landing_clearance < 0.55:
            return False, "blocked_by_landing_clearance"
        if center_lane.obstacle_cost < 0.22:
            return False, "blocked_by_no_obstacle"
        if center_lane.obstacle_cost > 0.92 and center_lane.clearance < 0.25:
            return False, "blocked_by_wall"
        if center_lane.threat_score > self._profile_limits()["jump_threat_max"]:
            return False, "blocked_by_threat_pressure"
        if (self._tick - self._last_jump_tick) < self.jump_cooldown:
            return False, "blocked_by_cooldown"
        if navigation.terrain_kind in {"pit", "edge"}:
            return False, "blocked_by_drop_risk"
        return True, "allowed_jump_obstacle"

    def _slide_gate(self, best_lane: NavigationLane, navigation: NavigationContext) -> tuple[bool, str]:
        if not self.sliding_enabled:
            return False, "blocked_by_feature_flag"
        if self.memory_required_for_slide and navigation.slope_source != "memory":
            return False, "blocked_by_memory_required"
        if navigation.terrain_kind == "unknown":
            return False, "blocked_by_slope_unknown"
        if navigation.terrain_kind == "uphill":
            return False, "blocked_by_uphill"
        if navigation.terrain_kind != "downhill":
            return False, "blocked_by_non_downhill"
        if best_lane.drop_risk >= self.drop_risk_threshold * 0.75:
            return False, "blocked_by_drop_risk"
        if best_lane.landing_clearance < 0.60:
            return False, "blocked_by_landing_clearance"
        if best_lane.threat_score > self._profile_limits()["slide_threat_max"]:
            return False, "blocked_by_threat_pressure"
        if (self._tick - self._last_slide_tick) < self.slide_cooldown:
            return False, "blocked_by_cooldown"
        return True, "allowed_slide_downhill"

    def _bunny_hop_allowed(self, center_lane: NavigationLane, navigation: NavigationContext) -> bool:
        if not self.allow_bunny_hop:
            return False
        if navigation.terrain_kind != "flat":
            return False
        if center_lane.drop_risk >= self.drop_risk_threshold * 0.55:
            return False
        if center_lane.clearance < 0.75 or center_lane.landing_clearance < 0.72:
            return False
        if center_lane.obstacle_cost > 0.25:
            return False
        if center_lane.threat_score > self._profile_limits()["bunny_threat_max"]:
            return False
        min_interval = self.jump_cooldown * self._profile_limits()["bunny_interval_mult"]
        if (self._tick - self._last_bunny_tick) < min_interval:
            return False
        return True

    def _unstuck_action(
        self,
        center_lane: NavigationLane,
        best_lane: NavigationLane,
        *,
        press_tab: bool,
        jump_allowed: bool,
    ) -> BotAction:
        if self._stuck_escape_timer <= 0:
            self._stuck_side = _lane_side(best_lane.label)
            if self._stuck_side == "center":
                self._stuck_side = "left"
            self._stuck_escape_timer = self.stuck_escape_ticks
        else:
            self._stuck_escape_timer -= 1
        if (
            self._stuck_frames >= (self.stuck_frames_required * 2)
            and jump_allowed
            and center_lane.drop_risk < self.drop_risk_threshold * 0.8
        ):
            return BotAction(dir_id=1, yaw=1, jump=1, slide=0, press_tab=press_tab, reason="unstuck_blocked")
        if self._stuck_side == "left":
            return BotAction(dir_id=3, yaw=0, jump=0, slide=0, press_tab=press_tab, reason="unstuck_blocked")
        return BotAction(dir_id=4, yaw=2, jump=0, slide=0, press_tab=press_tab, reason="unstuck_blocked")

    def _lane_move_action(
        self,
        lane: NavigationLane,
        *,
        reason: str,
        press_tab: bool,
        force_strafe: bool = False,
    ) -> BotAction:
        center = self.lane_count // 2
        if lane.index < center:
            if force_strafe or lane.index < (center - 1):
                return BotAction(dir_id=3, yaw=0, press_tab=press_tab, reason=reason)
            return BotAction(dir_id=5, yaw=0, press_tab=press_tab, reason=reason)
        if lane.index > center:
            if force_strafe or lane.index > (center + 1):
                return BotAction(dir_id=4, yaw=2, press_tab=press_tab, reason=reason)
            return BotAction(dir_id=6, yaw=2, press_tab=press_tab, reason=reason)
        return BotAction(dir_id=1, yaw=1, press_tab=press_tab, reason=reason)

    def _estimate_slope(self, snapshot: SceneSnapshot) -> tuple[str, float | None, str]:
        room_id = getattr(snapshot.map_state, "active_room_id", None)
        if room_id != self._last_room_id:
            self._z_history.clear()
            self._last_room_id = room_id
        world_pos = getattr(snapshot.player_pose, "world_pos", None)
        if world_pos is not None and len(world_pos) >= 3:
            self._z_history.append(float(world_pos[2]))
        if len(self._z_history) < 3:
            return "unknown", None, "none"
        delta = (self._z_history[-1] - self._z_history[0]) / max(1, len(self._z_history) - 1)
        if abs(delta) < self.downhill_z_threshold:
            return "flat", delta, "memory"
        if delta <= -self.downhill_z_threshold:
            return "downhill", delta, "memory"
        return "uphill", delta, "memory"

    def _estimate_room_edge_risk(self, snapshot: SceneSnapshot) -> float:
        pose = getattr(snapshot, "player_pose", None)
        state = getattr(snapshot, "map_state", None)
        if pose is None or state is None:
            return 0.0
        world_pos = getattr(pose, "world_pos", None)
        room_start = getattr(state, "room_start", None)
        room_end = getattr(state, "room_end", None)
        if world_pos is None or room_start is None or room_end is None:
            return 0.0
        x, y = float(world_pos[0]), float(world_pos[1])
        min_x, max_x = sorted((float(room_start[0]), float(room_end[0])))
        min_y, max_y = sorted((float(room_start[1]), float(room_end[1])))
        span_x = max(1e-3, max_x - min_x)
        span_y = max(1e-3, max_y - min_y)
        margin = min(
            (x - min_x) / span_x,
            (max_x - x) / span_x,
            (y - min_y) / span_y,
            (max_y - y) / span_y,
        )
        base_risk = _clamp(1.0 - (margin / 0.18))
        heading_deg = getattr(pose, "heading_deg", None)
        if heading_deg is None:
            return base_risk
        rad = math.radians(float(heading_deg))
        dx = math.cos(rad)
        dy = math.sin(rad)
        distances = []
        if abs(dx) > 1e-3:
            distances.append(((max_x - x) / dx) if dx > 0 else ((min_x - x) / dx))
        if abs(dy) > 1e-3:
            distances.append(((max_y - y) / dy) if dy > 0 else ((min_y - y) / dy))
        positive = [distance for distance in distances if distance > 0]
        if not positive:
            return base_risk
        forward_margin = min(positive) / max(span_x, span_y)
        return max(base_risk, _clamp(1.0 - (forward_margin / 0.25)))

    def _lane_bounds(self, frame_w: int) -> list[tuple[int, int]]:
        bounds = []
        step = float(frame_w) / float(self.lane_count)
        for index in range(self.lane_count):
            x0 = int(round(index * step))
            x1 = int(round((index + 1) * step))
            if index == self.lane_count - 1:
                x1 = frame_w
            bounds.append((max(0, x0), min(frame_w, max(x0 + 1, x1))))
        return bounds

    def _iter_grid_cells(self, snapshot: SceneSnapshot, nav_y0: int):
        for item in snapshot.analysis.get("grid", []):
            rect = getattr(item, "rect", None)
            if rect is None or len(rect) < 4:
                continue
            _, y, _, h = rect
            if int(y + h) <= int(nav_y0):
                continue
            yield item

    def _lane_metrics(
        self,
        gray: np.ndarray,
        nav_edges: np.ndarray,
        *,
        grid_cells,
        x0: int,
        x1: int,
        nav_y0: int,
        frame_h: int,
        room_edge_risk: float,
        threat_items: list[Any] | None,
        snapshot: SceneSnapshot,
    ) -> dict[str, float]:
        lane_gray = gray[nav_y0:frame_h, x0:x1]
        if lane_gray.size == 0:
            return {
                "threat_score": 0.0,
                "obstacle_cost": 1.0,
                "drop_risk": 1.0,
                "clearance": 0.0,
                "landing_clearance": 0.0,
            }
        nav_h = max(1, lane_gray.shape[0])
        near_y0 = int(nav_h * 0.68)
        mid_y0 = int(nav_h * 0.36)
        near_patch = lane_gray[near_y0:, :]
        far_patch = lane_gray[:mid_y0, :]
        dark_ratio = float((near_patch < 42).mean()) if near_patch.size else 0.0
        far_dark_ratio = float((far_patch < 42).mean()) if far_patch.size else 0.0
        edge_density = 0.0
        if nav_edges.size:
            lane_edges = nav_edges[:, x0:x1]
            edge_density = float((lane_edges > 0).mean()) if lane_edges.size else 0.0

        support_sum = 0.0
        support_weight = 0.0
        obstacle_sum = 0.0
        obstacle_weight = 0.0
        landing_sum = 0.0
        landing_weight = 0.0
        for cell in grid_cells:
            cx, cy = self._rect_center(getattr(cell, "rect", None))
            if cx is None or cy is None or not (x0 <= cx < x1):
                continue
            rel_y = _clamp((cy - nav_y0) / max(1.0, frame_h - nav_y0))
            weight = 0.8 + (rel_y * 1.4)
            label = str(getattr(cell, "label", "unknown")).lower()
            surface_score = 1.0 if label == "surface" else 0.20 if label == "unknown" else 0.0
            obstacle_score = 1.0 if label == "obstacle" else 0.45 if label == "unknown" else 0.0
            support_sum += surface_score * weight
            support_weight += weight
            obstacle_focus = 1.15 - (rel_y * 0.45)
            obstacle_sum += obstacle_score * obstacle_focus
            obstacle_weight += obstacle_focus
            if rel_y <= 0.45:
                landing_sum += surface_score
                landing_weight += 1.0

        support_ratio = support_sum / support_weight if support_weight > 0 else 0.0
        obstacle_cost = obstacle_sum / obstacle_weight if obstacle_weight > 0 else 0.65
        landing_clearance = landing_sum / landing_weight if landing_weight > 0 else 0.0
        drop_risk = _clamp(
            ((1.0 - support_ratio) * 0.62)
            + (dark_ratio * 0.18)
            + (far_dark_ratio * 0.08)
            + (room_edge_risk * 0.22)
            + max(0.0, edge_density - 0.08) * 1.8
        )
        if support_weight <= 0:
            drop_risk = max(drop_risk, 0.85)
        clearance = _clamp(((1.0 - obstacle_cost) * 0.65) + (landing_clearance * 0.35))
        threat_score = min(2.5, self._lane_threat_score(x0, x1, frame_h, threat_items, snapshot))
        return {
            "threat_score": float(threat_score),
            "obstacle_cost": _clamp(obstacle_cost),
            "drop_risk": float(drop_risk),
            "clearance": float(clearance),
            "landing_clearance": _clamp(landing_clearance),
        }

    def _lane_threat_score(
        self,
        x0: int,
        x1: int,
        frame_h: int,
        threat_items: list[Any] | None,
        snapshot: SceneSnapshot,
    ) -> float:
        items = list(threat_items or [])
        if not items:
            items.extend(getattr(snapshot, "enemy_classes", []))
            items.extend(getattr(snapshot, "projectile_classes", []))
            items.extend(getattr(snapshot, "hazards", []))
            items.extend(getattr(snapshot, "enemies", []))
            items.extend(getattr(snapshot, "projectiles", []))
        total = 0.0
        for item in items:
            cx, cy = self._rect_center(getattr(item, "rect", None))
            if cx is None or cy is None or not (x0 <= cx < x1):
                continue
            base = getattr(item, "priority", None)
            if base is None:
                base = max(
                    0.1,
                    float(getattr(item, "score", 0.0) or 0.0)
                    * max(1.0, float(getattr(item, "threat_tier", 1.0) or 1.0)),
                )
            proximity = 0.6 + (0.6 * _clamp(cy / max(1.0, float(frame_h))))
            total += float(base) * proximity
        return total

    @staticmethod
    def _rect_center(rect: Any) -> tuple[float | None, float | None]:
        if rect is None or len(rect) < 4:
            return None, None
        x, y, w, h = rect
        return float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)

    def _is_stuck(self, frame_bgr) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._last_gray is None:
            self._last_gray = gray
            return False
        diff = cv2.absdiff(gray, self._last_gray)
        self._last_gray = gray
        if not self._last_forwardish:
            self._stuck_frames = 0
            return False
        if float(diff.mean()) < self.stuck_diff_threshold:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
            self._stuck_escape_timer = 0
        return self._stuck_frames >= self.stuck_frames_required

    def _profile_limits(self) -> dict[str, float]:
        if self.profile == "aggressive":
            return {
                "threat_escape": 0.95,
                "jump_threat_max": 1.35,
                "slide_threat_max": 0.95,
                "bunny_threat_max": 0.40,
                "bunny_interval_mult": 2,
            }
        if self.profile == "balanced":
            return {
                "threat_escape": 0.75,
                "jump_threat_max": 1.05,
                "slide_threat_max": 0.78,
                "bunny_threat_max": 0.25,
                "bunny_interval_mult": 3,
            }
        return {
            "threat_escape": 0.55,
            "jump_threat_max": 0.82,
            "slide_threat_max": 0.58,
            "bunny_threat_max": 0.12,
            "bunny_interval_mult": 4,
        }
