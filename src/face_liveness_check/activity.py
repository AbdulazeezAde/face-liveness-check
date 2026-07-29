"""Landmark-based active-liveness signals for Face Mesh compatible landmarks.

Landmarks are normalized ``(x, y[, z])`` coordinates. The defaults follow the
MediaPipe Face Mesh index convention; applications using another model can pass
their own ``LandmarkIndices``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Challenge


@dataclass(frozen=True, slots=True)
class LandmarkIndices:
    left_eye: tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
    right_eye: tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)
    nose_tip: int = 1
    left_cheek: int = 234
    right_cheek: int = 454


@dataclass(frozen=True, slots=True)
class ActivityConfig:
    eye_closed_threshold: float = 0.19
    turn_threshold: float = 0.20
    nod_threshold: float = 0.045
    mirrored_input: bool = True
    indices: LandmarkIndices = LandmarkIndices()


class LandmarkActivityDetector:
    """Converts a stream of face landmarks into one-shot active challenges.

    For a mirrored selfie preview, ``mirrored_input=True`` reports directions as
    the person sees them. Calibrate thresholds against the camera/model pair.
    """

    def __init__(self, config: ActivityConfig | None = None) -> None:
        self.config = config or ActivityConfig()
        self._eyes_were_closed = False
        self._baseline_nose_y: float | None = None
        self._nod_down = False
        self._turn_latched: Challenge | None = None

    def observe(self, landmarks: np.ndarray) -> Challenge | None:
        points = np.asarray(landmarks, dtype=np.float32)
        required = max(*self.config.indices.left_eye, *self.config.indices.right_eye,
                       self.config.indices.nose_tip, self.config.indices.left_cheek,
                       self.config.indices.right_cheek)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) <= required:
            raise ValueError("landmarks do not contain the configured Face Mesh indices")

        blink = self._blink(points)
        if blink:
            return Challenge.BLINK
        turn = self._turn(points)
        if turn:
            return turn
        return self._nod(points)

    def eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        points = np.asarray(landmarks, dtype=np.float32)
        left = self._ear(points, self.config.indices.left_eye)
        right = self._ear(points, self.config.indices.right_eye)
        return float((left + right) / 2.0)

    def _blink(self, points: np.ndarray) -> bool:
        closed = self.eye_aspect_ratio(points) < self.config.eye_closed_threshold
        blinked = self._eyes_were_closed and not closed
        self._eyes_were_closed = closed
        return blinked

    def _turn(self, points: np.ndarray) -> Challenge | None:
        indices = self.config.indices
        nose_x = points[indices.nose_tip, 0]
        left_x, right_x = points[indices.left_cheek, 0], points[indices.right_cheek, 0]
        centre, half_width = (left_x + right_x) / 2.0, abs(right_x - left_x) / 2.0
        if half_width <= 1e-6:
            return None
        offset = (nose_x - centre) / half_width
        if self.config.mirrored_input:
            offset *= -1
        challenge: Challenge | None = None
        if offset >= self.config.turn_threshold:
            challenge = Challenge.TURN_RIGHT
        elif offset <= -self.config.turn_threshold:
            challenge = Challenge.TURN_LEFT
        if challenge != self._turn_latched:
            self._turn_latched = challenge
            return challenge
        if challenge is None:
            self._turn_latched = None
        return None

    def _nod(self, points: np.ndarray) -> Challenge | None:
        indices = self.config.indices
        nose_y = float(points[indices.nose_tip, 1])
        cheek_y = float((points[indices.left_cheek, 1] + points[indices.right_cheek, 1]) / 2.0)
        scale = abs(points[indices.right_cheek, 0] - points[indices.left_cheek, 0])
        if scale <= 1e-6:
            return None
        relative_y = (nose_y - cheek_y) / scale
        if self._baseline_nose_y is None:
            self._baseline_nose_y = relative_y
            return None
        delta = relative_y - self._baseline_nose_y
        if delta >= self.config.nod_threshold:
            self._nod_down = True
            return None
        if self._nod_down and delta <= self.config.nod_threshold / 3:
            self._nod_down = False
            self._baseline_nose_y = relative_y
            return Challenge.NOD
        if not self._nod_down:
            self._baseline_nose_y = 0.95 * self._baseline_nose_y + 0.05 * relative_y
        return None

    @staticmethod
    def _ear(points: np.ndarray, indices: tuple[int, int, int, int, int, int]) -> float:
        outer, upper_a, upper_b, inner, lower_a, lower_b = (points[index, :2] for index in indices)
        horizontal = np.linalg.norm(outer - inner)
        if horizontal <= 1e-6:
            return 0.0
        return float((np.linalg.norm(upper_a - lower_a) + np.linalg.norm(upper_b - lower_b)) / (2.0 * horizontal))
