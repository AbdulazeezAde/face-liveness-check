"""Opinionated, lean liveness stack composed from OpenCV and ONNX models."""

from __future__ import annotations

from pathlib import Path

from .activity import LandmarkActivityDetector
from .adapters import (
    MediaPipeLandmarkEstimator,
    OnnxPassiveAntiSpoof,
    OpenCVSFaceAligner,
    OpenCVSFaceEmbedder,
    OpenCVYuNetDetector,
)
from .evidence import EvidencePolicy, EvidenceSink
from .models import LivenessPolicy
from .model_packs import InstalledModelPack
from .pipeline import FrameEvidenceBuilder, ReferenceExtractor
from .verifier import LivenessVerifier


def create_opencv_verifier(
    *,
    yunet_model: str | Path,
    sface_model: str | Path,
    pad_model: str | Path,
    face_landmarker_model: str | Path,
    policy: LivenessPolicy | None = None,
    evidence_policy: EvidencePolicy | None = None,
    evidence_sink: EvidenceSink | None = None,
) -> LivenessVerifier:
    """Create the supported full liveness stack.

    YuNet supplies five landmarks for SFace alignment. The MediaPipe landmark model
    is deliberately required here because the default policy includes randomized
    blink and nod challenges; silently downgrading those checks would weaken liveness.
    """
    detector = OpenCVYuNetDetector(yunet_model)
    aligner = OpenCVSFaceAligner(sface_model)
    embedder = OpenCVSFaceEmbedder(sface_model)
    pad = OnnxPassiveAntiSpoof(
        pad_model,
        input_size=(80, 80),
        positive_index=0,
        output_is_logits=True,
        color_order="BGR",
    )
    activity = LandmarkActivityDetector()
    landmarks = MediaPipeLandmarkEstimator(face_landmarker_model)
    reference_extractor = ReferenceExtractor(detector, embedder, aligner)
    evidence_builder = FrameEvidenceBuilder(
        detector,
        embedder,
        pad,
        aligner=aligner,
        landmarks=landmarks,
        activity=activity,
        pad_crop_scale=2.7,
    )
    return LivenessVerifier(
        reference_extractor,
        evidence_builder,
        policy,
        evidence_policy=evidence_policy,
        evidence_sink=evidence_sink,
    )


def create_opencv_verifier_from_pack(
    installed: InstalledModelPack, *, policy: LivenessPolicy | None = None,
    evidence_policy: EvidencePolicy | None = None,
    evidence_sink: EvidenceSink | None = None,
) -> LivenessVerifier:
    """Build the supported full stack from a verified ``opencv-default`` cache."""
    required = {"yunet", "sface", "pad_minifasnet_v2", "face_landmarker"}
    missing = required.difference(installed.models)
    if missing:
        raise ValueError(f"installed model pack lacks OpenCV default artifacts: {', '.join(sorted(missing))}")
    return create_opencv_verifier(
        yunet_model=installed.models["yunet"],
        sface_model=installed.models["sface"],
        pad_model=installed.models["pad_minifasnet_v2"],
        face_landmarker_model=installed.models["face_landmarker"],
        policy=policy,
        evidence_policy=evidence_policy,
        evidence_sink=evidence_sink,
    )
