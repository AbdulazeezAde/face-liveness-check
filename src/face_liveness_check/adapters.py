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
    opencv_row: np.ndarray | None = None

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
        output_is_logits: bool = True, color_order: str = "RGB", session: object | None = None,
    ) -> None:
        if color_order not in {"BGR", "RGB"}:
            raise ValueError("color_order must be BGR or RGB")
        self.input_size, self.positive_index, self.output_is_logits = input_size, positive_index, output_is_logits
        self.color_order = color_order
        self._session = session or _load_session(model_path, providers)
        self._input_name = self._session.get_inputs()[0].name

    def score(self, face_bgr: np.ndarray) -> float:
        cv2 = _opencv()
        resized = cv2.resize(face_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        pixels = resized if self.color_order == "BGR" else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(pixels.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
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


class OpenCVYuNetDetector:
    """YuNet face detector using OpenCV's FaceDetectorYN API."""

    def __init__(self, model_path: str | Path, *, score_threshold: float = 0.9,
                 nms_threshold: float = 0.3, top_k: int = 5000) -> None:
        cv2 = _opencv()
        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, nms_threshold, top_k)

    def detect(self, image_bgr: np.ndarray) -> Sequence[FaceDetection]:
        height, width = image_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image_bgr)
        if faces is None:
            return ()
        detections = []
        for row in faces:
            x, y, box_width, box_height = row[:4]
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            detections.append(FaceDetection((float(x), float(y), float(x + box_width), float(y + box_height)), float(row[14]), landmarks, np.asarray(row, dtype=np.float32)))
        return tuple(sorted(detections, key=lambda item: item.confidence, reverse=True))


class OpenCVSFaceAligner:
    """Five-point SFace alignment for detections produced by ``OpenCVYuNetDetector``."""

    def __init__(self, model_path: str | Path) -> None:
        self._recognizer = _opencv().FaceRecognizerSF.create(str(model_path), "")

    def align(self, image_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        if detection.opencv_row is None:
            raise ValueError("OpenCVSFaceAligner requires a detection produced by OpenCVYuNetDetector")
        return self._recognizer.alignCrop(image_bgr, detection.opencv_row.reshape(1, -1))


class OpenCVSFaceEmbedder:
    """SFace embedding adapter using OpenCV's FaceRecognizerSF API."""

    def __init__(self, model_path: str | Path) -> None:
        self._recognizer = _opencv().FaceRecognizerSF.create(str(model_path), "")

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        return _normalize(np.asarray(self._recognizer.feature(aligned_face_bgr), dtype=np.float32).reshape(-1))


class MediaPipeLandmarkEstimator:
    """Optional dense-landmark adapter for reliable blink and nod challenges."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise ImportError("Install landmarks support: pip install 'face-liveness-check[mediapipe]'") from error
        self._mp = mp
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def estimate(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        cv2 = _opencv()
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if len(result.face_landmarks) != 1:
            raise ValueError("landmark model must detect exactly one face")
        return np.asarray([[point.x, point.y, point.z] for point in result.face_landmarks[0]], dtype=np.float32)
