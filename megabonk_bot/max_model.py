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
    """Тонкий адаптер для ONNX Runtime.

    В MVP может не использоваться, но интерфейс готов для Max-этапа.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None
        self._enabled = False
        try:
            import onnxruntime as ort  # type: ignore

            self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self._enabled = True
        except Exception:
            self._session = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(self, _frame_bgr) -> list[ObjectDetection]:
        # Интеграция реального препроцессинга/постпроцессинга модели
        # будет добавляться в Max-этапе.
        if not self._enabled:
            return []
        return []


class SceneMemory360:
    """Краткосрочная память объектов с затуханием."""

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
        priority = (1.0 - dist_norm) * 0.7 + area_norm * 0.3
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
