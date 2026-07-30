"""Command-line model management, reference extraction, and webcam demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .adapters import OpenCVYuNetDetector
from .model_packs import ModelPackManager, default_registry
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
    _model_source_options(webcam)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = ModelPackManager(default_registry(), cache_dir=getattr(args, "cache_dir", None))
    if args.command == "models":
        return _run_models(args, manager)
    installed = _resolve_pack(args, manager)
    if args.command == "extract":
        destination = extract_reference_face_crop_file(
            args.source, args.destination, OpenCVYuNetDetector(installed.models["yunet"]), pdf_page=args.page,
        )
        print(destination)
        return 0
    if args.command == "webcam":
        verifier = create_opencv_verifier_from_pack(installed)
        reference = load_reference_image_bgr(args.reference, pdf_page=args.reference_page)
        camera: int | str = int(args.camera) if args.camera.isdecimal() else args.camera
        run = verify_webcam(
            verifier, reference, source=camera, duration_s=args.duration, preview=not args.no_preview,
            on_challenges=lambda prompts: print("Complete in order:", " -> ".join(prompts)),
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


def _resolve_pack(args: argparse.Namespace, manager: ModelPackManager):
    return (
        manager.install(args.pack, accept_model_license=args.accept_model_license)
        if args.download else manager.resolve(args.pack)
    )


def _run_json(run) -> dict[str, object]:
    result = run.result
    return {
        "challenges": list(run.challenges),
        "matched": result.matched,
        "similarity": result.similarity,
        "reasons": list(result.reasons),
        "liveness": {
            **asdict(result.liveness),
            "completed_challenges": [challenge.value for challenge in result.liveness.completed_challenges],
            "reasons": list(result.liveness.reasons),
        },
    }
