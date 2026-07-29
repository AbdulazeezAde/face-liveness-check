"""Composable liveness and face-verification orchestration."""

from .models import Challenge, FrameEvidence, LivenessPolicy, LivenessResult, VerificationResult
from .session import LivenessSession
from .adapters import FaceDetection, OnnxArcFaceEmbedder, OnnxPassiveAntiSpoof
from .activity import ActivityConfig, LandmarkActivityDetector
from .pipeline import FrameEvidenceBuilder, ReferenceExtractor, ReferenceFace
from .verifier import LivenessVerifier, VerificationRun

__all__ = [
    "Challenge",
    "ActivityConfig",
    "FaceDetection",
    "FrameEvidenceBuilder",
    "FrameEvidence",
    "LivenessPolicy",
    "LivenessResult",
    "LivenessSession",
    "LivenessVerifier",
    "LandmarkActivityDetector",
    "OnnxArcFaceEmbedder",
    "OnnxPassiveAntiSpoof",
    "ReferenceExtractor",
    "ReferenceFace",
    "VerificationResult",
    "VerificationRun",
]

__version__ = "0.1.0"
