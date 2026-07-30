"""Composable liveness and face-verification orchestration."""

from .models import Challenge, FrameEvidence, LivenessPolicy, LivenessResult, PassiveAntiSpoofMode, VerificationResult
from .session import LivenessSession
from .adapters import FaceDetection, MediaPipeLandmarkEstimator, OnnxArcFaceEmbedder, OnnxPassiveAntiSpoof, OpenCVSFaceAligner, OpenCVSFaceEmbedder, OpenCVYuNetDetector
from .activity import ActivityConfig, LandmarkActivityDetector
from .pipeline import FrameEvidenceBuilder, ReferenceExtractor, ReferenceFace
from .verifier import LiveVerification, LivenessVerifier, VerificationRun
from .model_packs import ModelArtifact, ModelPack, ModelPackError, ModelPackManager, ModelPackRegistry, default_registry, facenox_pad_experimental_pack, opencv_default_pack, research_default_pack
from .presets import create_opencv_verifier, create_opencv_verifier_from_pack
from .reference_io import extract_reference_face_crop, extract_reference_face_crop_file, load_reference_image_bgr
from .webcam import verify_webcam
from .evaluation import PadCandidate, PadEvaluator, PadLabel, PadObservation, append_observation, summarize_observations
from .evidence import EvidenceArtifact, EvidenceEvent, EvidencePolicy, EvidenceSink, LocalEncryptedEvidenceSink, S3EvidenceSink

__all__ = [
    "Challenge",
    "ActivityConfig",
    "FaceDetection",
    "EvidenceArtifact",
    "EvidenceEvent",
    "EvidencePolicy",
    "EvidenceSink",
    "facenox_pad_experimental_pack",
    "extract_reference_face_crop",
    "extract_reference_face_crop_file",
    "FrameEvidenceBuilder",
    "FrameEvidence",
    "LivenessPolicy",
    "LivenessResult",
    "LivenessSession",
    "LivenessVerifier",
    "LocalEncryptedEvidenceSink",
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
    "PassiveAntiSpoofMode",
    "PadCandidate",
    "PadEvaluator",
    "PadLabel",
    "PadObservation",
    "OpenCVSFaceAligner",
    "OpenCVSFaceEmbedder",
    "OpenCVYuNetDetector",
    "ReferenceExtractor",
    "append_observation",
    "load_reference_image_bgr",
    "research_default_pack",
    "summarize_observations",
    "opencv_default_pack",
    "ReferenceFace",
    "S3EvidenceSink",
    "VerificationResult",
    "VerificationRun",
    "verify_webcam",
]

__version__ = "0.1.0"
