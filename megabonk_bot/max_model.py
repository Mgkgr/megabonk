from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    rect: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class ThreatScore:
    label: str
    rect: tuple[int, int, int, int]
    score: float
    distance_norm: float
    priority: float


@dataclass(frozen=True)
class BossWindow:
    name: str
    spawn_s: int
    prep_s: int = 10


class OnnxObjectDetector:
    """Lightweight ONNX Runtime adapter with a permissive parser."""

    def __init__(self, model_path: str, *, score_threshold: float = 0.25):
        self.model_path = model_path
        self.score_threshold = float(score_threshold)
        self._session = None
        self._enabled = False
        self._input_name = None
        self._class_labels: dict[int, str] = {}
        try:
            import onnxruntime as ort  # type: ignore

            self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            inputs = self._session.get_inputs()
            if inputs:
                self._input_name = inputs[0].name
            self._enabled = True
        except Exception:
            self._session = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(self, frame_bgr) -> list[ObjectDetection]:
        if not self._enabled or frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return []
        if self._input_name is None:
            return []
        import cv2
        import numpy as np

        input_meta = self._session.get_inputs()[0]
        shape = list(input_meta.shape)
        target_h = int(shape[2]) if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0 else 640
        target_w = int(shape[3]) if len(shape) >= 4 and isinstance(shape[3], int) and shape[3] > 0 else 640
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype("float32") / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        outputs = self._session.run(None, {self._input_name: tensor})
        return self._parse_outputs(outputs, frame_w=frame_bgr.shape[1], frame_h=frame_bgr.shape[0])

    def _parse_outputs(self, outputs, *, frame_w: int, frame_h: int) -> list[ObjectDetection]:
        import numpy as np

        if not outputs:
            return []
        first = outputs[0]
        if not isinstance(first, np.ndarray):
            return []
        arr = np.squeeze(first)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        if arr.ndim != 2 or arr.shape[1] < 6:
            return []

        detections: list[ObjectDetection] = []
        for row in arr:
            score = float(row[4])
            if score < self.score_threshold:
                continue
            x0, y0, x1, y1 = [int(round(float(value))) for value in row[:4]]
            if x1 <= x0 or y1 <= y0:
                cx = float(row[0]) * frame_w
                cy = float(row[1]) * frame_h
                bw = float(row[2]) * frame_w
                bh = float(row[3]) * frame_h
                x0 = int(round(cx - bw / 2.0))
                y0 = int(round(cy - bh / 2.0))
                x1 = int(round(cx + bw / 2.0))
                y1 = int(round(cy + bh / 2.0))
            class_id = int(row[5]) if arr.shape[1] > 5 else 0
            detections.append(
                ObjectDetection(
                    label=self._class_labels.get(class_id, f"class_{class_id}"),
                    rect=(max(0, x0), max(0, y0), max(1, x1 - x0), max(1, y1 - y0)),
                    score=score,
                )
            )
        return detections


class SceneMemory360:
    """Short-lived object memory with TTL decay."""

    def __init__(self, ttl_s: float = 2.0):
        self.ttl_s = float(ttl_s)
        self._memory: list[tuple[float, ObjectDetection]] = []

    def update(self, detections: list[ObjectDetection], now_ts: float | None = None) -> None:
        now_ts = time.time() if now_ts is None else float(now_ts)
        self._memory.extend((now_ts, det) for det in detections)
        self._memory = [
            (ts, det)
            for ts, det in self._memory
            if (now_ts - ts) <= self.ttl_s
        ]

    def get(self, now_ts: float | None = None) -> list[ObjectDetection]:
        now_ts = time.time() if now_ts is None else float(now_ts)
        alive = []
        for ts, det in self._memory:
            if (now_ts - ts) <= self.ttl_s:
                alive.append(det)
        return alive

    def clear(self) -> None:
        self._memory = []


def score_enemy_threats(
    enemies: list[ObjectDetection],
    *,
    frame_w: int,
    frame_h: int,
) -> list[ThreatScore]:
    if frame_w <= 0 or frame_h <= 0:
        return []
    center_x = frame_w / 2.0
    center_y = frame_h / 2.0
    frame_diag = math.sqrt(frame_w ** 2 + frame_h ** 2)
    scores: list[ThreatScore] = []
    for enemy in enemies:
        x, y, w, h = enemy.rect
        ex = x + w / 2.0
        ey = y + h / 2.0
        dist = math.sqrt((ex - center_x) ** 2 + (ey - center_y) ** 2)
        dist_norm = min(1.0, dist / max(1.0, frame_diag))
        area = max(1.0, float(w * h))
        area_norm = min(1.0, area / max(1.0, frame_w * frame_h * 0.2))
        threat_tier = float(getattr(enemy, "threat_tier", 1.0) or 1.0)
        label = str(getattr(enemy, "label", "enemy")).lower()
        family = str(getattr(enemy, "family", "") or "").lower()
        hazard_kind = str(getattr(enemy, "hazard_kind", "") or "").lower()
        metadata = getattr(enemy, "metadata", {}) or {}
        if isinstance(metadata, dict):
            hazard_kind = hazard_kind or str(metadata.get("hazard_kind", "") or "").lower()
            family = family or str(metadata.get("family", "") or "").lower()
        if "projectile" in label:
            threat_tier = max(threat_tier, 1.4)
        if "boss" in label:
            threat_tier = max(threat_tier, 2.0)
        if family in {"bossorb", "bosspoison", "ghostboss", "projectilebloodmagic"}:
            threat_tier = max(threat_tier, 1.8)
        if hazard_kind in {"spike", "poison", "root", "stone", "explosion", "hazard"}:
            threat_tier = max(threat_tier, 1.7)
        priority = ((1.0 - dist_norm) * 0.7 + area_norm * 0.3) * threat_tier
        priority *= max(0.0, min(1.0, float(enemy.score)))
        scores.append(
            ThreatScore(
                label=enemy.label,
                rect=enemy.rect,
                score=float(enemy.score),
                distance_norm=dist_norm,
                priority=priority,
            )
        )
    scores.sort(key=lambda item: item.priority, reverse=True)
    return scores


def build_occupancy_cost_map(
    *,
    frame_w: int,
    frame_h: int,
    obstacles: list[ObjectDetection],
    rows: int = 12,
    cols: int = 20,
) -> list[list[float]]:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    grid = [[0.0 for _ in range(cols)] for _ in range(rows)]
    if frame_w <= 0 or frame_h <= 0:
        return grid
    for obstacle in obstacles:
        x, y, w, h = obstacle.rect
        cx = x + w / 2.0
        cy = y + h / 2.0
        col = min(cols - 1, max(0, int((cx / frame_w) * cols)))
        row = min(rows - 1, max(0, int((cy / frame_h) * rows)))
        grid[row][col] += 1.0 + float(obstacle.score)
    return grid


def pick_low_cost_direction(cost_map: list[list[float]]) -> str:
    if not cost_map or not cost_map[0]:
        return "center"
    rows = len(cost_map)
    cols = len(cost_map[0])
    left_cols = range(0, max(1, cols // 3))
    center_cols = range(max(0, cols // 3), max(1, (2 * cols) // 3))
    right_cols = range(max(0, (2 * cols) // 3), cols)

    def _sum_cols(col_range):
        total = 0.0
        for row in range(rows):
            for col in col_range:
                total += cost_map[row][col]
        return total

    costs = {
        "left": _sum_cols(left_cols),
        "center": _sum_cols(center_cols),
        "right": _sum_cols(right_cols),
    }
    return min(costs, key=costs.get)


def should_enter_boss_prep(
    elapsed_s: int,
    schedule: list[BossWindow],
) -> tuple[bool, str | None]:
    now = int(elapsed_s)
    for window in schedule:
        if window.spawn_s - window.prep_s <= now < window.spawn_s:
            return True, window.name
    return False, None
