"""Bridge model adapters to liveness evidence and reference embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .activity import LandmarkActivityDetector
from .adapters import FaceAligner, FaceDetector, FaceEmbedder, LandmarkEstimator, PassiveAntiSpoof
from .models import FrameEvidence
from .quality import blur_score, lighting_score, perceptual_fingerprint


@dataclass(frozen=True, slots=True)
class ReferenceFace:
    embedding: np.ndarray
    detection_confidence: float


class ReferenceExtractor:
    """Extract a single enrolled identity embedding from an ID portrait or image."""

    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder, aligner: FaceAligner | None = None) -> None:
        self.detector, self.embedder, self.aligner = detector, embedder, aligner

    def extract(self, image_bgr: np.ndarray) -> ReferenceFace:
        detections = list(self.detector.detect(image_bgr))
        if len(detections) != 1:
            raise ValueError(f"reference image must contain exactly one face; found {len(detections)}")
        detection = detections[0]
        face = self.aligner.align(image_bgr, detection) if self.aligner else detection.crop(image_bgr, padding=0.10)
        return ReferenceFace(self.embedder.embed(face), detection.confidence)


class FrameEvidenceBuilder:
    """Produce one ``FrameEvidence`` item from a video frame and supplied models."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        anti_spoof: PassiveAntiSpoof,
        *,
        aligner: FaceAligner | None = None,
        landmarks: LandmarkEstimator | None = None,
        activity: LandmarkActivityDetector | None = None,
        pad_crop_scale: float = 1.2,
        tracking_id_factory: Callable[[np.ndarray], str | None] | None = None,
    ) -> None:
        if (landmarks is None) != (activity is None):
            raise ValueError("landmarks and activity detector must be configured together")
        if pad_crop_scale < 1.0:
            raise ValueError("pad_crop_scale must be at least 1.0")
        self.detector, self.embedder, self.anti_spoof = detector, embedder, anti_spoof
        self.aligner, self.landmarks, self.activity = aligner, landmarks, activity
        self.pad_crop_scale = pad_crop_scale
        self.tracking_id_factory = tracking_id_factory

    def build(self, frame_bgr: np.ndarray, timestamp_s: float) -> FrameEvidence:
        detections = list(self.detector.detect(frame_bgr))
        fingerprint = perceptual_fingerprint(frame_bgr)
        if len(detections) != 1:
            return FrameEvidence(timestamp_s, len(detections), None, 0.0, 0.0, None, frame_fingerprint=fingerprint)
        detection = detections[0]
        face = self.aligner.align(frame_bgr, detection) if self.aligner else detection.crop(frame_bgr, padding=0.10)
        pad_face = detection.crop(frame_bgr, padding=(self.pad_crop_scale - 1.0) / 2.0)
        light, sharpness = lighting_score(face), blur_score(face)
        motion = self.activity.observe(self.landmarks.estimate(face)) if self.activity and self.landmarks else None
        return FrameEvidence(
            timestamp_s=timestamp_s,
            face_count=1,
            tracking_id=self.tracking_id_factory(face) if self.tracking_id_factory else None,
            quality_score=min(light, sharpness),
            lighting_score=light,
            passive_antispoof_score=self.anti_spoof.score(pad_face),
            motion=motion,
            embedding=self.embedder.embed(face),
            frame_fingerprint=fingerprint,
        )

    def extract_face_crop(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return one live face crop for opt-in evidence capture, or ``None``."""
        detections = list(self.detector.detect(frame_bgr))
        if len(detections) != 1:
            return None
        return detections[0].crop(frame_bgr, padding=0.10)
