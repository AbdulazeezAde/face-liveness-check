"""Licensed-model adapters for ONNX face verification deployments.

The package deliberately does not ship model weights. Callers provide models they
are licensed to use and are responsible for calibrating thresholds on their data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """A detected face, using pixel coordinates in left, top, right, bottom order."""

    bbox: tuple[float, float, float, float]
    confidence: float
    landmarks: np.ndarray | None = None

    def crop(self, image: np.ndarray, padding: float = 0.0) -> np.ndarray:
        """Return a clipped BGR crop, optionally expanded by a fraction of face size."""
        height, width = image.shape[:2]
        left, top, right, bottom = self.bbox
        pad_x, pad_y = (right - left) * padding, (bottom - top) * padding
        x1, y1 = max(0, int(left - pad_x)), max(0, int(top - pad_y))
        x2, y2 = min(width, int(right + pad_x)), min(height, int(bottom + pad_y))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("face bounding box has no visible pixels")
        return image[y1:y2, x1:x2].copy()


class FaceDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> Sequence[FaceDetection]: ...


class FaceEmbedder(Protocol):
    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray: ...


class PassiveAntiSpoof(Protocol):
    def score(self, face_bgr: np.ndarray) -> float: ...


class FaceAligner(Protocol):
    def align(self, image_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray: ...


class LandmarkEstimator(Protocol):
    def estimate(self, aligned_face_bgr: np.ndarray) -> np.ndarray: ...


class OnnxArcFaceEmbedder:
    """ArcFace-style ONNX embedding adapter for already-aligned BGR face crops.

    Most ArcFace exports accept 112x112 RGB tensors normalized with
    ``(pixel - 127.5) / 128``. Override defaults if your licensed model differs.
    """

    def __init__(
        self, model_path: str | Path, *, providers: Sequence[str] | None = None,
        input_size: tuple[int, int] = (112, 112), session: object | None = None,
    ) -> None:
        self.input_size = input_size
        self._session = session or _load_session(model_path, providers)
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        cv2 = _opencv()
        resized = cv2.resize(aligned_face_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.transpose((rgb - 127.5) / 128.0, (2, 0, 1))[None, ...]
        output = self._session.run(None, {self._input_name: tensor})[0]
        return _normalize(np.asarray(output, dtype=np.float32).reshape(-1))


class OnnxPassiveAntiSpoof:
    """Binary ONNX PAD adapter with explicit, model-specific output configuration."""

    def __init__(
        self, model_path: str | Path, *, providers: Sequence[str] | None = None,
        input_size: tuple[int, int] = (80, 80), positive_index: int = 1,
        output_is_logits: bool = True, session: object | None = None,
    ) -> None:
        self.input_size, self.positive_index, self.output_is_logits = input_size, positive_index, output_is_logits
        self._session = session or _load_session(model_path, providers)
        self._input_name = self._session.get_inputs()[0].name

    def score(self, face_bgr: np.ndarray) -> float:
        cv2 = _opencv()
        resized = cv2.resize(face_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        output = np.asarray(self._session.run(None, {self._input_name: tensor})[0]).reshape(-1)
        return self.score_output(output, self.positive_index, self.output_is_logits)

    @staticmethod
    def score_output(output: np.ndarray, positive_index: int, output_is_logits: bool) -> float:
        values = np.asarray(output, dtype=np.float64).reshape(-1)
        if not 0 <= positive_index < values.size:
            raise ValueError("positive_index is outside the PAD model output")
        if output_is_logits:
            values = _softmax(values)
        score = float(values[positive_index])
        if not 0.0 <= score <= 1.0:
            raise ValueError("PAD model output must be a probability or logits")
        return score


def _load_session(model_path: str | Path, providers: Sequence[str] | None) -> object:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("Install ONNX support: pip install 'face-liveness-check[onnx]'") from error
    return ort.InferenceSession(str(model_path), providers=list(providers) if providers else None)


def _opencv() -> object:
    try:
        import cv2
    except ImportError as error:
        raise ImportError("Install ONNX support: pip install 'face-liveness-check[onnx]'") from error
    return cv2


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("model returned a zero embedding")
    return vector / norm


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
