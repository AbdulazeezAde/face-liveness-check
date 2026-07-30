import hashlib

import numpy as np
import pytest

from face_liveness_check import (
    Challenge,
    FrameEvidence,
    LivenessPolicy,
    LivenessSession,
    ModelArtifact,
    ModelPack,
    ModelPackError,
    ModelPackManager,
    ModelPackRegistry,
    default_registry,
    extract_reference_face_crop,
)
from face_liveness_check.adapters import FaceDetection, OnnxPassiveAntiSpoof
from face_liveness_check.activity import ActivityConfig, LandmarkActivityDetector, LandmarkIndices
from face_liveness_check.cli import build_parser
from face_liveness_check.pipeline import ReferenceExtractor
from face_liveness_check.verifier import LivenessVerifier


def _evidence(timestamp, motion=None, embedding=None):
    return FrameEvidence(
        timestamp_s=timestamp,
        face_count=1,
        tracking_id="person-1",
        quality_score=0.9,
        lighting_score=0.9,
        passive_antispoof_score=0.95,
        motion=motion,
        embedding=embedding,
        frame_fingerprint=f"frame-{timestamp}",
    )


def test_session_requires_ordered_challenges_and_matches_embedding():
    policy = LivenessPolicy(challenge_count=1, min_face_frames=3, minimum_live_embeddings=3, min_match_similarity=0.9)
    session = LivenessSession(policy)
    challenge = session.challenges[0]
    vector = np.array([1.0, 0.0, 0.0])
    session.observe(_evidence(1, embedding=vector))
    session.observe(_evidence(2, motion=challenge, embedding=vector))
    session.observe(_evidence(3, embedding=vector))

    result = session.compare(vector)

    assert result.liveness.passed
    assert result.matched
    assert result.similarity == 1.0


def test_session_fails_without_passive_antispoof_evidence():
    policy = LivenessPolicy(challenge_count=1, min_face_frames=1)
    session = LivenessSession(policy)
    session.observe(
        FrameEvidence(1, 1, "person-1", 0.9, 0.9, None, motion=session.challenges[0])
    )

    result = session.result()

    assert not result.passed
    assert "passive anti-spoof evidence is missing" in result.reasons


def test_face_detection_crop_and_pad_logits_are_normalized():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    crop = FaceDetection((2, 3, 8, 9), 0.99).crop(image)

    score = OnnxPassiveAntiSpoof.score_output(np.array([-2.0, 2.0]), 1, True)

    assert crop.shape == (6, 6, 3)
    assert 0.98 < score < 0.99


def test_landmark_detector_emits_blink_after_close_then_open():
    indices = LandmarkIndices(left_eye=(0, 1, 2, 3, 4, 5), right_eye=(6, 7, 8, 9, 10, 11), nose_tip=12, left_cheek=13, right_cheek=14)
    detector = LandmarkActivityDetector(ActivityConfig(indices=indices, eye_closed_threshold=0.2))
    open_face = np.array([[0, 0], [.25, .2], [.75, .2], [1, 0], [.25, -.2], [.75, -.2]] * 2 + [[.5, .5], [0, .5], [1, .5]], dtype=np.float32)
    closed_face = open_face.copy()
    closed_face[[1, 2, 4, 5, 7, 8, 10, 11], 1] = 0
    assert detector.observe(open_face) is None
    assert detector.observe(closed_face) is None
    assert detector.observe(open_face) == Challenge.BLINK


class _OneFaceDetector:
    def detect(self, image):
        return [FaceDetection((0, 0, 4, 4), 0.99)]


class _Embedder:
    def embed(self, face):
        return np.array([1.0, 0.0])


def test_reference_extractor_rejects_ambiguous_images_and_verifier_exposes_challenges():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    extractor = ReferenceExtractor(_OneFaceDetector(), _Embedder())
    reference = extractor.extract(image)
    assert reference.detection_confidence == 0.99
    assert np.array_equal(reference.embedding, np.array([1.0, 0.0]))

    class Builder:
        def build(self, frame, timestamp):
            return _evidence(timestamp, embedding=np.array([1.0, 0.0]))

    verifier = LivenessVerifier(extractor, Builder(), LivenessPolicy(challenge_count=1, min_face_frames=3, minimum_live_embeddings=3))
    run = verifier.verify(image, [(1, image), (2, image), (3, image)])
    assert len(run.challenges) == 1
    assert not run.result.liveness.passed  # no active challenge was supplied


def test_live_verification_exposes_prompts_before_frames_are_captured():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    extractor = ReferenceExtractor(_OneFaceDetector(), _Embedder())

    class Builder:
        def build(self, frame, timestamp):
            return _evidence(timestamp, embedding=np.array([1.0, 0.0]))

    verifier = LivenessVerifier(extractor, Builder(), LivenessPolicy(challenge_count=1))
    live = verifier.start(image)

    assert len(live.challenges) == 1
    assert live.challenges[0] in {challenge.value for challenge in Challenge}


def test_model_pack_install_records_license_and_offline_resolution_rechecks_checksums(tmp_path):
    payload = tmp_path / "source.onnx"
    payload.write_bytes(b"trusted-model-bytes")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    pack = ModelPack(
        name="test-pack",
        version="1",
        license_notice="test licence",
        artifacts=(ModelArtifact("model", payload.as_uri(), digest, "model.onnx", "test", "https://example.test/license"),),
    )
    registry = ModelPackRegistry()
    registry.register(pack)
    manager = ModelPackManager(registry, cache_dir=tmp_path / "cache")

    with pytest.raises(ModelPackError, match="accept_model_license"):
        manager.install("test-pack")
    installed = manager.install("test-pack", accept_model_license=True)
    assert manager.resolve("test-pack").models == installed.models

    installed.models["model"].write_bytes(b"tampered")
    with pytest.raises(ModelPackError, match="invalid checksums"):
        manager.resolve("test-pack")


def test_opencv_default_pack_is_complete_and_checksum_pinned():
    pack = default_registry().get("opencv-default")
    assert {artifact.name for artifact in pack.artifacts} == {"yunet", "sface", "pad_minifasnet_v2", "face_landmarker"}
    assert all(len(artifact.sha256) == 64 for artifact in pack.artifacts)
    assert all("latest" not in artifact.url for artifact in pack.artifacts)


def test_reference_crop_requires_one_face_without_opencv_runtime():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    crop = extract_reference_face_crop(image, _OneFaceDetector())
    assert crop.shape == (4, 4, 3)

    class NoFace:
        def detect(self, image):
            return []

    with pytest.raises(ValueError, match="exactly one face"):
        extract_reference_face_crop(image, NoFace())


def test_cli_parses_model_install_and_webcam_commands():
    parser = build_parser()
    install = parser.parse_args(["models", "install", "opencv-default", "--accept-model-license"])
    webcam = parser.parse_args(["webcam", "id.png", "--download", "--duration", "12"])

    assert install.name == "opencv-default"
    assert install.accept_model_license
    assert webcam.reference.name == "id.png"
    assert webcam.duration == 12
