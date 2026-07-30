"""Composable liveness and face-verification orchestration."""

from .models import Challenge, FrameEvidence, LivenessPolicy, LivenessResult, VerificationResult
from .session import LivenessSession
from .adapters import FaceDetection, MediaPipeLandmarkEstimator, OnnxArcFaceEmbedder, OnnxPassiveAntiSpoof, OpenCVSFaceAligner, OpenCVSFaceEmbedder, OpenCVYuNetDetector
from .activity import ActivityConfig, LandmarkActivityDetector
from .pipeline import FrameEvidenceBuilder, ReferenceExtractor, ReferenceFace
from .verifier import LiveVerification, LivenessVerifier, VerificationRun
from .model_packs import ModelArtifact, ModelPack, ModelPackError, ModelPackManager, ModelPackRegistry, default_registry, opencv_default_pack, research_default_pack
from .presets import create_opencv_verifier, create_opencv_verifier_from_pack
from .reference_io import extract_reference_face_crop, extract_reference_face_crop_file, load_reference_image_bgr
from .webcam import verify_webcam

__all__ = [
    "Challenge",
    "ActivityConfig",
    "FaceDetection",
    "extract_reference_face_crop",
    "extract_reference_face_crop_file",
    "FrameEvidenceBuilder",
    "FrameEvidence",
    "LivenessPolicy",
    "LivenessResult",
    "LivenessSession",
    "LivenessVerifier",
    "LiveVerification",
    "MediaPipeLandmarkEstimator",
    "ModelArtifact",
    "ModelPack",
    "ModelPackError",
    "ModelPackManager",
    "ModelPackRegistry",
    "default_registry",
    "create_opencv_verifier",
    "create_opencv_verifier_from_pack",
    "LandmarkActivityDetector",
    "OnnxArcFaceEmbedder",
    "OnnxPassiveAntiSpoof",
    "OpenCVSFaceAligner",
    "OpenCVSFaceEmbedder",
    "OpenCVYuNetDetector",
    "ReferenceExtractor",
    "load_reference_image_bgr",
    "research_default_pack",
    "opencv_default_pack",
    "ReferenceFace",
    "VerificationResult",
    "VerificationRun",
    "verify_webcam",
]

__version__ = "0.1.0"
