"""Opt-in command-line model management, reference extraction, and webcam tools."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from .adapters import OnnxPassiveAntiSpoof, OpenCVYuNetDetector
from .evidence import EvidencePolicy, LocalEncryptedEvidenceSink
from .evaluation import PadCandidate, PadEvaluator, PadLabel, append_observation, calibrate_thresholds, summarize_observations
from .model_packs import ModelPackManager, default_registry
from .models import active_first_profile
from .presets import create_opencv_verifier_from_pack
from .reference_io import extract_reference_face_crop_file, load_reference_image_bgr
from .webcam import verify_webcam


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="face-liveness-check")
    subcommands = parser.add_subparsers(dest="command", required=True)
    models = subcommands.add_parser("models", help="inspect or install checksum-verified model packs")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_commands.add_parser("list", help="list bundled model manifests")
    install = model_commands.add_parser("install", help="download a model pack into the local cache")
    install.add_argument("name")
    _cache_and_license_options(install)

    extract = subcommands.add_parser("extract", help="extract one portrait crop from an image or PDF ID")
    extract.add_argument("source", type=Path)
    extract.add_argument("destination", type=Path)
    extract.add_argument("--page", type=int, default=0, help="zero-indexed PDF page (default: 0)")
    _model_source_options(extract)

    webcam = subcommands.add_parser("webcam", help="run an interactive webcam liveness verification")
    webcam.add_argument("reference", type=Path, help="portrait image or PDF ID")
    webcam.add_argument("--reference-page", type=int, default=0)
    webcam.add_argument("--camera", default="0", help="camera index, file path, or RTSP URL")
    webcam.add_argument("--duration", type=float, default=15.0, help="capture duration in seconds")
    webcam.add_argument("--no-preview", action="store_true", help="do not open an OpenCV preview window")
    webcam.add_argument("--pad-advisory", action="store_true", help="flag low PAD scores as suspicious instead of rejecting them")
    webcam.add_argument("--evidence-local-dir", type=Path, help="encrypted local evidence directory; requires explicit consent and a Fernet key")
    webcam.add_argument("--evidence-key-env", default="FACE_LIVENESS_EVIDENCE_KEY", help="environment variable containing the Fernet key")
    webcam.add_argument("--evidence-consent", action="store_true", help="confirm consent for this session's optional evidence retention")
    webcam.add_argument("--evidence-capture-face-crops", action="store_true", help="also retain up to three detected face crops when evidence is captured")
    webcam.add_argument("--evidence-retention-days", type=int, help="retention deadline recorded with encrypted evidence")
    _model_source_options(webcam)

    evaluate = subcommands.add_parser(
        "evaluate-webcam",
        help="collect score-only, labelled passive anti-spoofing observations from a webcam",
    )
    evaluate.add_argument("--label", required=True, choices=[label.value for label in PadLabel if label is not PadLabel.UNKNOWN])
    evaluate.add_argument("--output", required=True, type=Path, help="JSONL output containing labels and scores only")
    evaluate.add_argument("--camera", default="0", help="camera index, file path, or RTSP URL")
    evaluate.add_argument("--duration", type=float, default=15.0, help="capture duration in seconds")
    evaluate.add_argument(
        "--candidate",
        action="append",
        choices=("minifasnet_v2", "facenox_experimental"),
        help="candidate to score; repeat to compare candidates (default: both)",
    )
    evaluate.add_argument("--no-preview", action="store_true", help="do not open an OpenCV preview window")
    _model_source_options(evaluate)

    calibrate = subcommands.add_parser(
        "calibrate-pad",
        help="propose PAD thresholds from score-only labelled JSONL; never uploads sample data",
    )
    calibrate.add_argument("--input", required=True, type=Path, help="score-only JSONL produced by evaluate-webcam")
    calibrate.add_argument("--candidate", action="append", help="candidate name to calibrate; repeat as needed")
    calibrate.add_argument("--target-genuine-accept-rate", type=float, default=.95)
    calibrate.add_argument("--target-attack-reject-rate", type=float, default=.95)
    calibrate.add_argument("--minimum-samples-per-label", type=int, default=20)

    retention = subcommands.add_parser(
        "evidence-retention",
        help="preview or apply local encrypted-evidence retention deletion",
    )
    retention.add_argument("--evidence-local-dir", required=True, type=Path)
    retention.add_argument("--evidence-key-env", default="FACE_LIVENESS_EVIDENCE_KEY")
    retention.add_argument("--apply", action="store_true", help="delete only sessions identified by the retention preview")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = ModelPackManager(default_registry(), cache_dir=getattr(args, "cache_dir", None))
    if args.command == "models":
        return _run_models(args, manager)
    if args.command == "evaluate-webcam":
        return _run_pad_evaluation(args, manager)
    if args.command == "calibrate-pad":
        result = calibrate_thresholds(
            args.input,
            candidates=args.candidate,
            target_genuine_accept_rate=args.target_genuine_accept_rate,
            target_attack_reject_rate=args.target_attack_reject_rate,
            minimum_samples_per_label=args.minimum_samples_per_label,
        )
        print(json.dumps({name: calibration.to_record() for name, calibration in result.items()}, indent=2))
        return 0 if all(calibration.eligible for calibration in result.values()) else 2
    if args.command == "evidence-retention":
        key = os.environ.get(args.evidence_key_env)
        if not key:
            raise ValueError(f"set the Fernet key in environment variable {args.evidence_key_env}")
        result = LocalEncryptedEvidenceSink(args.evidence_local_dir, key).purge_expired(dry_run=not args.apply)
        print(json.dumps(result.to_record(), indent=2))
        return 0
    installed = _resolve_pack(args, manager)
    if args.command == "extract":
        destination = extract_reference_face_crop_file(
            args.source, args.destination, OpenCVYuNetDetector(installed.models["yunet"]), pdf_page=args.page,
        )
        print(destination)
        return 0
    if args.command == "webcam":
        evidence_policy, evidence_sink = _local_evidence_options(args)
        verifier = create_opencv_verifier_from_pack(
            installed,
            profile=active_first_profile() if args.pad_advisory else None,
            evidence_policy=evidence_policy,
            evidence_sink=evidence_sink,
        )
        reference = load_reference_image_bgr(args.reference, pdf_page=args.reference_page)
        camera: int | str = int(args.camera) if args.camera.isdecimal() else args.camera
        run = verify_webcam(
            verifier, reference, source=camera, duration_s=args.duration, preview=not args.no_preview,
            on_challenges=lambda prompts: print("Complete in order:", " -> ".join(prompts)),
            evidence_consent=args.evidence_consent,
        )
        print(json.dumps(_run_json(run), indent=2))
        return 0 if run.result.matched else 2
    raise AssertionError(f"unhandled command: {args.command}")


def _cache_and_license_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--accept-model-license", action="store_true", help="acknowledge the displayed model-pack licence notice")


def _model_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pack", default="opencv-default")
    parser.add_argument("--download", action="store_true", help="download models if they are not already cached")
    _cache_and_license_options(parser)


def _run_models(args: argparse.Namespace, manager: ModelPackManager) -> int:
    if args.models_command == "list":
        for name in manager.registry.names():
            pack = manager.registry.get(name)
            print(f"{pack.name} {pack.version}: {pack.license_notice}")
        return 0
    installed = manager.install(args.name, accept_model_license=args.accept_model_license)
    print(installed.directory)
    return 0


def _local_evidence_options(args: argparse.Namespace) -> tuple[EvidencePolicy | None, LocalEncryptedEvidenceSink | None]:
    if args.evidence_local_dir is None:
        if args.evidence_consent or args.evidence_capture_face_crops or args.evidence_retention_days is not None:
            raise ValueError("evidence options require --evidence-local-dir")
        return None, None
    if not args.evidence_consent:
        raise ValueError("--evidence-local-dir requires --evidence-consent")
    key = os.environ.get(args.evidence_key_env)
    if not key:
        raise ValueError(f"set the Fernet key in environment variable {args.evidence_key_env}")
    return (
        EvidencePolicy(
            enabled=True,
            capture_face_crops=args.evidence_capture_face_crops,
            retention_days=args.evidence_retention_days,
        ),
        LocalEncryptedEvidenceSink(args.evidence_local_dir, key),
    )


def _run_pad_evaluation(args: argparse.Namespace, manager: ModelPackManager) -> int:
    """Collect only opaque IDs, labels, face counts, timestamps, and PAD scores."""
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    selected = tuple(dict.fromkeys(args.candidate or ("minifasnet_v2", "facenox_experimental")))
    installed = _resolve_pack(args, manager)
    candidates: list[PadCandidate] = []
    if "minifasnet_v2" in selected:
        candidates.append(PadCandidate(
            "minifasnet_v2",
            OnnxPassiveAntiSpoof(installed.models["pad_minifasnet_v2"], input_size=(80, 80), positive_index=0, color_order="BGR"),
            crop_padding=0.85,
        ))
    if "facenox_experimental" in selected:
        facenox = _resolve_named_pack(args, manager, "pad-facenox-experimental")
        candidates.append(PadCandidate(
            "facenox_experimental",
            OnnxPassiveAntiSpoof(
                facenox.models["pad_facenox_minifas"], input_size=(128, 128), positive_index=0,
                color_order="RGB", resize_mode="letterbox_reflect",
            ),
            crop_padding=0.25,
        ))
    evaluator = PadEvaluator(OpenCVYuNetDetector(installed.models["yunet"], score_threshold=0.7), candidates)
    cv2 = _opencv()
    camera: int | str = int(args.camera) if args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera source: {args.camera}")
    label = PadLabel(args.label)
    sample_id = f"{label.value}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    started = monotonic()
    observations = 0
    try:
        while monotonic() - started < args.duration:
            ok, frame = capture.read()
            if not ok:
                break
            append_observation(args.output, evaluator.observe(frame, sample_id=sample_id, label=label))
            observations += 1
            if not args.no_preview:
                preview = frame.copy()
                cv2.putText(preview, f"PAD evaluation: {label.value} (q to stop)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
                cv2.imshow("face-liveness-check PAD evaluation", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    thresholds = {candidate.name: 0.8 if candidate.name == "minifasnet_v2" else 0.5 for candidate in candidates}
    print(json.dumps({"sample_id": sample_id, "label": label.value, "observations": observations, "summary": summarize_observations(args.output, thresholds)}, indent=2))
    return 0


def _resolve_named_pack(args: argparse.Namespace, manager: ModelPackManager, name: str):
    return manager.install(name, accept_model_license=args.accept_model_license) if args.download else manager.resolve(name)


def _resolve_pack(args: argparse.Namespace, manager: ModelPackManager):
    return (
        manager.install(args.pack, accept_model_license=args.accept_model_license)
        if args.download else manager.resolve(args.pack)
    )


def _opencv():
    try:
        import cv2
    except ImportError as error:
        raise ImportError("Install OpenCV support: pip install 'face-liveness-check[opencv]'") from error
    return cv2


def _run_json(run) -> dict[str, object]:
    result = run.result
    return {
        "challenges": list(run.challenges),
        "profile": run.profile,
        "automatic_decision_allowed": run.automatic_decision_allowed,
        "matched": result.matched,
        "similarity": result.similarity,
        "reasons": list(result.reasons),
        "evidence_session_id": run.evidence_session_id,
        "liveness": {
            **asdict(result.liveness),
            "completed_challenges": [challenge.value for challenge in result.liveness.completed_challenges],
            "reasons": list(result.liveness.reasons),
            "warnings": list(result.liveness.warnings),
        },
    }
