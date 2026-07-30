import hashlib
from importlib.metadata import version
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from face_liveness_check import (
    Challenge,
    BarcodePayload,
    DocumentQuality,
    DocumentType,
    EvidenceArtifact,
    EvidenceEvent,
    EvidencePolicy,
    EvidenceRetentionResult,
    FrameEvidence,
    IdDocumentExtractor,
    JsonBarcodeFieldParser,
    LabelledCardTemplate,
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
    PadCalibrationResult,
    PadEvaluator,
    PadLabel,
    NormalizedDocument,
    OcrTextBlock,
    PassiveAntiSpoofMode,
    ReviewEvent,
    ReviewPolicy,
    S3EvidenceSink,
    WebhookReviewSink,
    active_first_profile,
    append_observation,
    active_first_policy,
    calibrate_thresholds,
    evaluation_profile,
    strict_profile,
    summarize_observations,
)
from face_liveness_check.adapters import FaceDetection, OnnxPassiveAntiSpoof
from face_liveness_check.activity import ActivityConfig, LandmarkActivityDetector, LandmarkIndices
from face_liveness_check.cli import build_parser
from face_liveness_check.pipeline import ReferenceExtractor
from face_liveness_check.verifier import LivenessVerifier


def test_public_version_matches_installed_package_metadata():
    import face_liveness_check

    assert face_liveness_check.__version__ == version("face-liveness-check")


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


def test_active_first_policy_keeps_secure_default_thresholds_but_makes_pad_advisory():
    policy = active_first_policy(challenge_count=2)

    assert policy.challenge_count == 2
    assert policy.passive_antispoof_mode is PassiveAntiSpoofMode.ADVISORY


def test_verification_profiles_make_decision_intent_explicit():
    strict = strict_profile(challenge_count=2)
    active = active_first_profile(challenge_count=2)
    evaluation = evaluation_profile(challenge_count=2)

    assert strict.name.value == "strict" and strict.allows_automatic_decision
    assert strict.policy.passive_antispoof_mode is PassiveAntiSpoofMode.REQUIRED
    assert active.name.value == "active_first" and active.allows_automatic_decision
    assert active.policy.passive_antispoof_mode is PassiveAntiSpoofMode.ADVISORY
    assert evaluation.name.value == "evaluation" and not evaluation.allows_automatic_decision
    assert evaluation.policy.passive_antispoof_mode is PassiveAntiSpoofMode.ADVISORY


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


def test_suspicious_active_first_session_dispatches_metadata_only_review_event():
    class Sink:
        def __init__(self):
            self.events = []

        def publish(self, event):
            self.events.append(event)

    class Builder:
        motion = None

        def build(self, frame, timestamp):
            return FrameEvidence(timestamp, 1, "person-1", .9, .9, .1, motion=self.motion, embedding=np.array([1.0, 0.0]))

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    sink, builder = Sink(), Builder()
    verifier = LivenessVerifier(
        ReferenceExtractor(_OneFaceDetector(), _Embedder()),
        builder,
        profile=active_first_profile(challenge_count=1, min_face_frames=1, minimum_live_embeddings=1),
        review_policy=ReviewPolicy(enabled=True),
        review_sink=sink,
    )
    live = verifier.start(image, session_id="session_002")
    builder.motion = live.session.challenges[0]
    live.observe(image, 1)
    run = live.finish()

    assert run.profile == "active_first"
    assert run.automatic_decision_allowed
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.disposition == "suspicious"
    assert event.session_id == "session_002"
    assert event.profile == "active_first"
    assert b"embedding" not in event.payload()
    assert b"frame" not in event.payload()


def test_evaluation_profile_disables_automatic_decisions_and_webhook_is_signed():
    event = ReviewEvent.create(
        event_id="event_001",
        session_id="session_003",
        profile="evaluation",
        automatic_decision_allowed=False,
        matched=True,
        liveness_passed=True,
        similarity=.9,
        liveness_reasons=(),
        liveness_warnings=(),
    )

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def getcode(self):
            return self.status

    calls = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return Response()

    sink = WebhookReviewSink("https://review.example.test/events", signing_key="review-key", opener=opener)
    sink.publish(event)

    request, timeout = calls[0]
    assert timeout == 5.0
    assert request.data == event.payload()
    assert request.get_header("X-face-liveness-event-id") == "event_001"
    assert request.get_header("X-face-liveness-signature").startswith("sha256=")
    assert event.automatic_decision_allowed is False


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
    webcam = parser.parse_args([
        "webcam", "id.png", "--download", "--duration", "12", "--pad-advisory",
        "--evidence-local-dir", "evidence", "--evidence-consent", "--evidence-capture-face-crops",
    ])
    evaluation = parser.parse_args(["evaluate-webcam", "--label", "replay", "--output", "scores.jsonl", "--candidate", "facenox_experimental"])
    calibration = parser.parse_args(["calibrate-pad", "--input", "scores.jsonl", "--candidate", "facenox_experimental"])
    retention = parser.parse_args(["evidence-retention", "--evidence-local-dir", "evidence"])
    extract_id = parser.parse_args(["extract-id", "passport.png", "--document-type", "passport_td3", "--read-barcodes"])

    assert install.name == "opencv-default"
    assert install.accept_model_license
    assert webcam.reference.name == "id.png"
    assert webcam.duration == 12
    assert webcam.pad_advisory
    assert webcam.evidence_local_dir.name == "evidence"
    assert webcam.evidence_capture_face_crops
    assert evaluation.label == "replay"
    assert evaluation.output.name == "scores.jsonl"
    assert evaluation.candidate == ["facenox_experimental"]
    assert calibration.input.name == "scores.jsonl"
    assert calibration.candidate == ["facenox_experimental"]
    assert retention.evidence_local_dir.name == "evidence"
    assert not retention.apply
    assert extract_id.source.name == "passport.png"
    assert extract_id.document_type == "passport_td3"
    assert extract_id.read_barcodes


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


def test_pad_calibration_uses_distinct_samples_and_withholds_small_datasets(tmp_path):
    records = []
    for label, score in (("genuine", .9), ("print", .1), ("replay", .1), ("mask", .1)):
        for index in range(2):
            records.append({"sample_id": f"{label}-{index}", "label": label, "scores": {"candidate": score}})
    path = tmp_path / "scores.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    withheld = calibrate_thresholds(path, minimum_samples_per_label=3)
    proposed = calibrate_thresholds(path, minimum_samples_per_label=2)

    assert isinstance(withheld["candidate"], PadCalibrationResult)
    assert not withheld["candidate"].eligible
    assert proposed["candidate"].eligible
    assert proposed["candidate"].threshold == .9
    assert proposed["candidate"].metrics.attack_reject_rate == 1.0


def test_local_evidence_retention_is_dry_run_by_default_and_skips_indefinite_records(tmp_path):
    pytest.importorskip("cryptography.fernet")
    from face_liveness_check import LocalEncryptedEvidenceSink

    sink = LocalEncryptedEvidenceSink(tmp_path / "evidence", LocalEncryptedEvidenceSink.generate_key())
    expired = EvidenceEvent(
        session_id="expired_001", created_at="2025-01-01T00:00:00+00:00", categories=("failed",),
        matched=False, liveness_passed=False, similarity=None, liveness_reasons=("failed",),
        liveness_warnings=(), retention_days=1,
    )
    indefinite = EvidenceEvent(
        session_id="indefinite_001", created_at="2025-01-01T00:00:00+00:00", categories=("failed",),
        matched=False, liveness_passed=False, similarity=None, liveness_reasons=("failed",),
        liveness_warnings=(), retention_days=None,
    )
    sink.store(expired, [])
    sink.store(indefinite, [])

    preview = sink.purge_expired(now=datetime(2025, 1, 3, tzinfo=timezone.utc))
    applied = sink.purge_expired(dry_run=False, now=datetime(2025, 1, 3, tzinfo=timezone.utc))

    assert isinstance(preview, EvidenceRetentionResult)
    assert preview.dry_run and preview.plan.eligible_session_ids == ("expired_001",)
    assert not (tmp_path / "evidence" / "expired_001").exists()
    assert applied.removed_session_ids == ("expired_001",)
    assert (tmp_path / "evidence" / "indefinite_001").exists()


def test_id_extractor_validates_passport_mrz_and_keeps_normalized_document_opt_in():
    first = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    second = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"

    class Ocr:
        def read(self, _image):
            return (OcrTextBlock(first, .99), OcrTextBlock(second, .98))

    class Normalizer:
        def normalize(self, image):
            return NormalizedDocument(image.copy(), DocumentQuality(True, .9, .01, ()))

    image = np.zeros((100, 160, 3), dtype=np.uint8)
    result = IdDocumentExtractor(Ocr(), normalizer=Normalizer(), detector=_OneFaceDetector()).extract(image)

    assert result.document_type is DocumentType.PASSPORT_TD3
    assert result.fields["document_number"].value == "L898902C3"
    assert result.fields["document_number"].validated
    assert result.fields["date_of_birth_yymmdd"].validated
    assert result.portrait_crop_bgr.shape == (4, 4, 3)
    assert result.normalized_document_bgr is None
    assert not result.requires_manual_review


def test_paddle_adapter_accepts_documented_prediction_shape_without_paddle_runtime():
    from face_liveness_check.ocr import PaddleOcrEngine

    class Runner:
        def predict(self, _image):
            return [{"res": {"rec_texts": ["FEDERAL REPUBLIC"], "rec_scores": [.96], "rec_polys": [[[1, 2], [3, 2], [3, 4], [1, 4]]]}}]

    blocks = PaddleOcrEngine(Runner()).read(np.zeros((2, 2, 3), dtype=np.uint8))

    assert blocks == (OcrTextBlock("FEDERAL REPUBLIC", .96, ((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),)


def test_labelled_card_template_and_barcode_conflicts_require_manual_review():
    class Ocr:
        def read(self, _image):
            return (OcrTextBlock("NIN: 12345678901", .98), OcrTextBlock("FULL NAME", .95), OcrTextBlock("Ada Example", .96))

    class Normalizer:
        def normalize(self, image):
            return NormalizedDocument(image.copy(), DocumentQuality(True, .9, .01, ()))

    class Barcodes:
        def read(self, _image):
            return (BarcodePayload("QR", '{"nin":"99999999999"}'),)

    template = LabelledCardTemplate(
        {"nin": ("NIN",), "full_name": ("FULL NAME",)},
        markers=("NIN",), required_fields=("nin", "full_name"),
        validators={"nin": lambda value: value.isdigit() and len(value) == 11},
    )
    result = IdDocumentExtractor(
        Ocr(), normalizer=Normalizer(), templates=(template,), barcode_reader=Barcodes(), barcode_field_parser=JsonBarcodeFieldParser(),
    ).extract(np.zeros((20, 30, 3), dtype=np.uint8))

    assert result.document_type is DocumentType.CARD
    assert result.fields["nin"].value == "12345678901"
    assert result.barcodes[0].format == "QR"
    assert "barcode value conflicts with extracted field: nin" in result.warnings
    assert result.requires_manual_review
