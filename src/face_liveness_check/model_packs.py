"""Secure, cacheable model-pack installation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

try:
    from platformdirs import user_cache_dir
except ImportError:  # pragma: no cover - package dependency normally provides this
    def user_cache_dir(appname: str) -> str:
        return str(Path.home() / ".cache" / appname)


class ModelPackError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    name: str
    url: str
    sha256: str
    filename: str
    purpose: str
    license_url: str
    preprocessing: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256.lower()):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        if Path(self.filename).name != self.filename:
            raise ValueError("filename must not contain directory components")


@dataclass(frozen=True, slots=True)
class ModelPack:
    name: str
    version: str
    license_notice: str
    artifacts: tuple[ModelArtifact, ...]
    requires_license_acceptance: bool = True


@dataclass(frozen=True, slots=True)
class InstalledModelPack:
    pack: ModelPack
    directory: Path
    models: dict[str, Path]


class ModelPackRegistry:
    def __init__(self) -> None:
        self._packs: dict[str, ModelPack] = {}

    def register(self, pack: ModelPack) -> None:
        if pack.name in self._packs:
            raise ValueError(f"model pack already registered: {pack.name}")
        self._packs[pack.name] = pack

    def get(self, name: str) -> ModelPack:
        try:
            return self._packs[name]
        except KeyError as error:
            raise ModelPackError(f"unknown model pack: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._packs))


class ModelPackManager:
    def __init__(self, registry: ModelPackRegistry, cache_dir: str | Path | None = None) -> None:
        self.registry = registry
        self.cache_dir = Path(cache_dir) if cache_dir else Path(user_cache_dir("face-liveness-check")) / "models"

    def install(self, name: str, *, accept_model_license: bool = False, timeout_s: float = 60.0) -> InstalledModelPack:
        pack = self.registry.get(name)
        if pack.requires_license_acceptance and not accept_model_license:
            raise ModelPackError("pass accept_model_license=True after reviewing this model pack's licence")
        directory = self.cache_dir / pack.name / pack.version
        directory.mkdir(parents=True, exist_ok=True)
        models: dict[str, Path] = {}
        for artifact in pack.artifacts:
            destination = directory / artifact.filename
            if not _matches(destination, artifact.sha256):
                self._download(artifact, destination, timeout_s)
            models[artifact.name] = destination
        _write_json_atomic(
            directory / "manifest.json",
            {
                "pack": asdict(pack),
                "models": {key: str(value) for key, value in models.items()},
                "accepted_model_license": accept_model_license,
            },
        )
        return InstalledModelPack(pack, directory, models)

    def resolve(self, name: str) -> InstalledModelPack:
        """Return a previously verified cached pack without making a network request."""
        pack = self.registry.get(name)
        directory = self.cache_dir / pack.name / pack.version
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ModelPackError(f"model pack is not installed: {name}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelPackError(f"model pack manifest is unreadable: {name}") from error
        if pack.requires_license_acceptance and not manifest.get("accepted_model_license"):
            raise ModelPackError(f"model pack licence has not been accepted: {name}")
        models = {artifact.name: directory / artifact.filename for artifact in pack.artifacts}
        invalid = [artifact.name for artifact in pack.artifacts if not _matches(models[artifact.name], artifact.sha256)]
        if invalid:
            raise ModelPackError(f"model pack is incomplete or has invalid checksums: {', '.join(invalid)}")
        return InstalledModelPack(pack, directory, models)

    @staticmethod
    def _download(artifact: ModelArtifact, destination: Path, timeout_s: float) -> None:
        handle, temporary = tempfile.mkstemp(prefix="model-", suffix=".part", dir=destination.parent)
        temp_path = Path(temporary)
        try:
            with os.fdopen(handle, "wb") as output, urlopen(artifact.url, timeout=timeout_s) as response:
                shutil.copyfileobj(response, output)
            if not _matches(temp_path, artifact.sha256):
                raise ModelPackError(f"checksum mismatch for {artifact.name}")
            temp_path.replace(destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise


def _matches(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    digest = _sha256(path)
    return digest == expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(destination: Path, value: dict[str, Any]) -> None:
    handle, temporary = tempfile.mkstemp(prefix="manifest-", suffix=".json", dir=destination.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def research_default_pack() -> ModelPack:
    """Research-only baseline PAD pack; active challenges remain mandatory.

    This pack intentionally includes no identity or landmark weights. Those models
    have independent licences and output contracts, and must be supplied by the app.
    """
    return ModelPack(
        name="research-default",
        version="2026.1",
        license_notice=(
            "Research baseline only. MiniFASNetV2 is passive PAD evidence, not a "
            "guarantee against presentation attacks or camera injection. Require "
            "active randomized liveness challenges and evaluate on your deployment."
        ),
        artifacts=(
            ModelArtifact(
                name="pad_minifasnet_v2",
                url="https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
                sha256="d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b",
                filename="minifasnet_v2.onnx",
                purpose="Passive face anti-spoofing: live, print attack, replay attack",
                license_url="https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx",
                preprocessing={
                    "input_size": [80, 80], "color_order": "BGR", "range": "zero_to_one",
                    "crop_scale": 2.7, "live_class_index": 0, "output": "three_class_logits",
                },
            ),
        ),
    )


def opencv_default_pack() -> ModelPack:
    """The full default stack: YuNet, SFace, MiniFASNetV2 and Face Landmarker.

    Each URL is versioned or content-addressed by its SHA-256 checksum. A failed
    checksum deliberately prevents use rather than silently accepting changed
    model bytes.
    """
    return ModelPack(
        name="opencv-default",
        version="2026.1",
        license_notice=(
            "This pack combines OpenCV Zoo MIT models, Apache-2.0 MiniFASNetV2, "
            "and the Apache-2.0 MediaPipe Face Landmarker task. It is a baseline, "
            "not a guarantee against presentation attacks or camera injection. "
            "Keep randomized active challenges enabled and calibrate on deployment data."
        ),
        artifacts=(
            ModelArtifact(
                name="yunet",
                url="https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                filename="face_detection_yunet_2023mar.onnx",
                purpose="Face detection and five-point landmarks",
                license_url="https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md",
            ),
            ModelArtifact(
                name="sface",
                url="https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
                filename="face_recognition_sface_2021dec.onnx",
                purpose="Five-point alignment and face identity embedding",
                license_url="https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md",
            ),
            ModelArtifact(
                name="pad_minifasnet_v2",
                url="https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
                sha256="d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b",
                filename="minifasnet_v2.onnx",
                purpose="Passive anti-spoofing: live, print attack, replay attack",
                license_url="https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx",
                preprocessing={
                    "input_size": [80, 80], "color_order": "BGR", "range": "zero_to_one",
                    "crop_scale": 2.7, "live_class_index": 0, "output": "three_class_logits",
                },
            ),
            ModelArtifact(
                name="face_landmarker",
                url="https://storage.googleapis.com/mediapipe-assets/face_landmarker.task?generation=1678323583183024",
                sha256="7cf2bbf1842c429e9defee38e7f1c4238978d8a6faf2da145bb19846f86bd2f4",
                filename="face_landmarker.task",
                purpose="Dense landmarks for blink and nod challenges",
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
            ),
        ),
    )


def facenox_pad_experimental_pack() -> ModelPack:
    """Independent binary PAD candidate for local evaluation only."""
    return ModelPack(
        name="pad-facenox-experimental",
        version="2026.1",
        license_notice=(
            "Experimental PAD candidate. Its reported benchmark metrics must be "
            "independently reproduced on your cameras and presentation attacks "
            "before it is used to make production decisions."
        ),
        artifacts=(
            ModelArtifact(
                name="pad_facenox_minifas",
                url="https://raw.githubusercontent.com/facenox/face-antispoof-onnx/2d4b33a3c0ba6e27772ac3a9b48ec495bf5c1dad/models/best_model_quantized.onnx",
                sha256="fde20585635cae62ed1d41796f76b6f8bc4b92cd91ec1cf0f1bc6485d2d587a9",
                filename="facenox_minifas_quantized.onnx",
                purpose="Experimental binary passive PAD candidate: real or spoof",
                license_url="https://github.com/facenox/face-antispoof-onnx/blob/2d4b33a3c0ba6e27772ac3a9b48ec495bf5c1dad/LICENSE",
                preprocessing={
                    "input_size": [128, 128], "color_order": "RGB", "range": "zero_to_one",
                    "crop_scale": 1.5, "resize_mode": "letterbox_reflect", "live_class_index": 0,
                    "output": "two_class_logits",
                },
            ),
        ),
    )


def default_registry() -> ModelPackRegistry:
    registry = ModelPackRegistry()
    registry.register(research_default_pack())
    registry.register(opencv_default_pack())
    registry.register(facenox_pad_experimental_pack())
    return registry
