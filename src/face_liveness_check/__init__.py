"""Composable liveness and face-verification orchestration."""

from .models import Challenge, FrameEvidence, LivenessPolicy, LivenessResult, PassiveAntiSpoofMode, VerificationProfile, VerificationProfileName, VerificationResult, active_first_policy, active_first_profile, evaluation_profile, strict_profile
from .session import LivenessSession
from .adapters import FaceDetection, MediaPipeLandmarkEstimator, OnnxArcFaceEmbedder, OnnxPassiveAntiSpoof, OpenCVSFaceAligner, OpenCVSFaceEmbedder, OpenCVYuNetDetector
from .activity import ActivityConfig, LandmarkActivityDetector
from .pipeline import FrameEvidenceBuilder, ReferenceExtractor, ReferenceFace
from .verifier import LiveVerification, LivenessVerifier, VerificationRun
from .model_packs import ModelArtifact, ModelPack, ModelPackError, ModelPackManager, ModelPackRegistry, default_registry, facenox_pad_experimental_pack, opencv_default_pack, research_default_pack
from .presets import create_opencv_verifier, create_opencv_verifier_from_pack
from .reference_io import extract_reference_face_crop, extract_reference_face_crop_file, load_reference_image_bgr
from .webcam import verify_webcam
from .evaluation import PadCalibrationResult, PadCandidate, PadEvaluator, PadLabel, PadObservation, PadThresholdMetrics, append_observation, calibrate_thresholds, summarize_observations
from .evidence import EvidenceArtifact, EvidenceEvent, EvidencePolicy, EvidenceRetentionPlan, EvidenceRetentionResult, EvidenceSink, LocalEncryptedEvidenceSink, S3EvidenceSink
from .review import ReviewEvent, ReviewPolicy, ReviewSink, WebhookReviewSink

__all__ = [
    "Challenge",
    "ActivityConfig",
    "active_first_policy",
    "active_first_profile",
    "FaceDetection",
    "EvidenceArtifact",
    "EvidenceEvent",
    "EvidencePolicy",
    "EvidenceRetentionPlan",
    "EvidenceRetentionResult",
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
    "PadCalibrationResult",
    "PadEvaluator",
    "PadLabel",
    "PadObservation",
    "PadThresholdMetrics",
    "OpenCVSFaceAligner",
    "OpenCVSFaceEmbedder",
    "OpenCVYuNetDetector",
    "ReferenceExtractor",
    "ReviewEvent",
    "ReviewPolicy",
    "ReviewSink",
    "append_observation",
    "calibrate_thresholds",
    "load_reference_image_bgr",
    "research_default_pack",
    "summarize_observations",
    "opencv_default_pack",
    "ReferenceFace",
    "S3EvidenceSink",
    "strict_profile",
    "evaluation_profile",
    "VerificationProfile",
    "VerificationProfileName",
    "VerificationResult",
    "VerificationRun",
    "WebhookReviewSink",
    "verify_webcam",
]

__version__ = "0.1.0"
