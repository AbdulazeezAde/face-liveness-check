import hashlib

import numpy as np
import pytest

from face_liveness_check import (
    Challenge,
    EvidenceArtifact,
    EvidenceEvent,
    EvidencePolicy,
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
    PadCandidate,
    PadEvaluator,
    PadLabel,
    PassiveAntiSpoofMode,
    S3EvidenceSink,
    append_observation,
    summarize_observations,
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


def test_advisory_passive_pad_warns_without_failing_active_liveness():
    policy = LivenessPolicy(
        challenge_count=1,
        min_face_frames=1,
        minimum_live_embeddings=1,
        passive_antispoof_mode=PassiveAntiSpoofMode.ADVISORY,
    )
    session = LivenessSession(policy)
    session.observe(FrameEvidence(1, 1, "person-1", .9, .9, .1, motion=session.challenges[0], embedding=np.array([1.0, 0.0])))

    result = session.result()

    assert result.passed
    assert result.warnings == ("passive anti-spoof check is suspicious",)


def test_face_detection_crop_and_pad_logits_are_normalized():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    crop = FaceDetection((2, 3, 8, 9), 0.99).crop(image)

    score = OnnxPassiveAntiSpoof.score_output(np.array([-2.0, 2.0]), 1, True)

    assert crop.shape == (6, 6, 3)
    assert 0.98 < score < 0.99


def test_pad_adapter_feeds_resized_tensor(monkeypatch):
    class FakeCv:
        INTER_LINEAR = 1
        INTER_LANCZOS4 = 2
        INTER_AREA = 3
        COLOR_BGR2RGB = 4
        BORDER_REFLECT_101 = 5

        @staticmethod
        def resize(image, size, interpolation):
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        @staticmethod
        def cvtColor(image, code):
            return image

    class Input:
        name = "input"

    class Session:
        def __init__(self):
            self.tensor = None

        @staticmethod
        def get_inputs():
            return [Input()]

        def run(self, _, values):
            self.tensor = values["input"]
            return [np.array([[2.0, -2.0]])]

    import face_liveness_check.adapters as adapters

    monkeypatch.setattr(adapters, "_opencv", lambda: FakeCv)
    session = Session()
    adapter = OnnxPassiveAntiSpoof("unused.onnx", session=session, input_size=(80, 80), positive_index=0)
    adapter.score(np.zeros((9, 7, 3), dtype=np.uint8))

    assert session.tensor.shape == (1, 3, 80, 80)


def test_landmark_detector_emits_blink_after_close_then_open():
    indices = LandmarkIndices(left_eye=(0, 1, 2, 3, 4, 5), right_eye=(6, 7, 8, 9, 10, 11), nose_tip=12, left_cheek=13, right_cheek=14)
    detector = LandmarkActivityDetector(ActivityConfig(indices=indices, eye_closed_threshold=0.2))
    open_face = np.array([[0, 0], [.25, .2], [.75, .2], [1, 0], [.25, -.2], [.75, -.2]] * 2 + [[.5, .5], [0, .5], [1, .5]], dtype=np.float32)
    closed_face = open_face.copy()
    closed_face[[1, 2, 4, 5, 7, 8, 10, 11], 1] = 0
    assert detector.observe(open_face) is None
    assert detector.observe(closed_face) is None
    assert detector.observe(open_face) == Challenge.BLINK


def test_landmark_detector_requires_neutral_pose_before_turn():
    indices = LandmarkIndices(left_eye=(0, 1, 2, 3, 4, 5), right_eye=(6, 7, 8, 9, 10, 11), nose_tip=12, left_cheek=13, right_cheek=14)
    detector = LandmarkActivityDetector(ActivityConfig(indices=indices, turn_threshold=0.2, mirrored_input=False))
    landmarks = np.array([[0, 0], [.25, .2], [.75, .2], [1, 0], [.25, -.2], [.75, -.2]] * 2 + [[.8, .5], [0, .5], [1, .5]], dtype=np.float32)
    neutral = landmarks.copy()
    neutral[12, 0] = .5

    assert detector.observe(landmarks) is None  # a pre-turned static image is not activity
    assert detector.observe(neutral) is None
    assert detector.observe(landmarks) == Challenge.TURN_RIGHT


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


def test_suspicious_session_stores_opt_in_evidence_without_embeddings():
    class Sink:
        def __init__(self):
            self.event = None
            self.artifacts = None

        def store(self, event, artifacts):
            self.event, self.artifacts = event, artifacts

    class Builder:
        motion = None

        def build(self, frame, timestamp):
            return FrameEvidence(timestamp, 1, "person-1", .9, .9, .1, motion=self.motion, embedding=np.array([1.0, 0.0]))

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    sink, builder = Sink(), Builder()
    verifier = LivenessVerifier(
        ReferenceExtractor(_OneFaceDetector(), _Embedder()),
        builder,
        LivenessPolicy(challenge_count=1, min_face_frames=1, minimum_live_embeddings=1, passive_antispoof_mode=PassiveAntiSpoofMode.ADVISORY),
        evidence_policy=EvidencePolicy(enabled=True, capture_on={"suspicious"}, max_frames=1),
        evidence_sink=sink,
    )
    with pytest.raises(PermissionError, match="evidence_consent"):
        verifier.start(image)

    live = verifier.start(image, evidence_consent=True, session_id="session_001")
    builder.motion = live.session.challenges[0]
    live.observe(image, 1)
    run = live.finish()

    assert run.result.matched
    assert sink.event.session_id == "session_001"
    assert sink.event.categories == ("suspicious",)
    assert len(sink.artifacts) == 1
    assert b"embedding" not in sink.artifacts[0].data


def test_s3_sink_uses_kms_for_metadata_and_artifacts():
    class Client:
        def __init__(self):
            self.calls = []

        def put_object(self, **kwargs):
            self.calls.append(kwargs)

    client = Client()
    sink = S3EvidenceSink("evidence-bucket", kms_key_id="key-123", prefix="liveness", client=client)
    event = EvidenceEvent.create(
        session_id="session_001", categories=("suspicious",), matched=True, liveness_passed=True,
        similarity=.9, liveness_reasons=(), liveness_warnings=("passive anti-spoof check is suspicious",), retention_days=30,
    )
    sink.store(event, [EvidenceArtifact("frame_000", "application/x-npy", b"frame")])

    assert [call["Key"] for call in client.calls] == ["liveness/session_001/event.json", "liveness/session_001/frame_000.npy"]
    assert all(call["ServerSideEncryption"] == "aws:kms" and call["SSEKMSKeyId"] == "key-123" for call in client.calls)


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


def test_cli_parses_model_install_webcam_and_pad_evaluation_commands():
    parser = build_parser()
    install = parser.parse_args(["models", "install", "opencv-default", "--accept-model-license"])
    webcam = parser.parse_args(["webcam", "id.png", "--download", "--duration", "12"])
    evaluation = parser.parse_args(["evaluate-webcam", "--label", "replay", "--output", "scores.jsonl", "--candidate", "facenox_experimental"])

    assert install.name == "opencv-default"
    assert install.accept_model_license
    assert webcam.reference.name == "id.png"
    assert webcam.duration == 12
    assert evaluation.label == "replay"
    assert evaluation.output.name == "scores.jsonl"
    assert evaluation.candidate == ["facenox_experimental"]


def test_pad_evaluation_records_scores_without_source_paths(tmp_path):
    class Scorer:
        def score(self, crop):
            return 0.9

    evaluator = PadEvaluator(_OneFaceDetector(), [PadCandidate("candidate", Scorer(), 0.1)])
    observation = evaluator.observe(np.zeros((4, 4, 3), dtype=np.uint8), sample_id="genuine-001", label=PadLabel.GENUINE)
    output = append_observation(tmp_path / "scores.jsonl", observation)

    assert observation.scores == {"candidate": 0.9}
    assert "source" not in output.read_text(encoding="utf-8")
    assert summarize_observations(output, {"candidate": 0.8})["candidate"]["genuine_accept_rate"] == 1.0

    with pytest.raises(ValueError, match="not a file path"):
        evaluator.observe(np.zeros((4, 4, 3), dtype=np.uint8), sample_id="C:/id.jpg", label=PadLabel.GENUINE)
