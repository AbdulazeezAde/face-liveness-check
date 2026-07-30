"""Opt-in, privacy-conscious storage of suspicious verification evidence.

The liveness engine never enables this module by default. Integrators choose a
sink, obtain consent where needed, and own retention and access controls.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


_CAPTURE_EVENTS = frozenset({"suspicious", "failed", "passed"})
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    """Controls optional storage; disabled by default and consent-gated."""

    enabled: bool = False
    capture_on: frozenset[str] = frozenset({"suspicious", "failed"})
    capture_frames: bool = True
    capture_face_crops: bool = False
    max_frames: int = 3
    require_consent: bool = True
    retention_days: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "capture_on", frozenset(self.capture_on))
        if not self.capture_on or not self.capture_on <= _CAPTURE_EVENTS:
            raise ValueError(f"capture_on must contain only: {', '.join(sorted(_CAPTURE_EVENTS))}")
        if self.max_frames < 1:
            raise ValueError("max_frames must be at least one")
        if self.retention_days is not None and self.retention_days < 1:
            raise ValueError("retention_days must be positive when specified")


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    """Non-biometric metadata associated with a captured verification session."""

    session_id: str
    created_at: str
    categories: tuple[str, ...]
    matched: bool
    liveness_passed: bool
    similarity: float | None
    liveness_reasons: tuple[str, ...]
    liveness_warnings: tuple[str, ...]
    retention_days: int | None

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        categories: Sequence[str],
        matched: bool,
        liveness_passed: bool,
        similarity: float | None,
        liveness_reasons: Sequence[str],
        liveness_warnings: Sequence[str],
        retention_days: int | None,
    ) -> "EvidenceEvent":
        _validate_opaque_id(session_id)
        return cls(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            categories=tuple(categories),
            matched=matched,
            liveness_passed=liveness_passed,
            similarity=similarity,
            liveness_reasons=tuple(liveness_reasons),
            liveness_warnings=tuple(liveness_warnings),
            retention_days=retention_days,
        )

    def payload(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    """An image payload supplied only to an explicitly configured evidence sink."""

    name: str
    content_type: str
    data: bytes

    def __post_init__(self) -> None:
        if not _OPAQUE_ID.fullmatch(self.name):
            raise ValueError("artifact name must be an opaque identifier")
        if not self.content_type or not self.data:
            raise ValueError("evidence artifact requires content type and data")


class EvidenceSink(Protocol):
    """Destination for metadata and optional evidence; implementations own storage."""

    def store(self, event: EvidenceEvent, artifacts: Sequence[EvidenceArtifact]) -> None: ...


class LocalEncryptedEvidenceSink:
    """Store event metadata and NPY image artifacts encrypted with Fernet.

    Install ``face-liveness-check[evidence-local]``. The caller must protect the
    key separately from the evidence directory and enforce deletion at the
    retention deadline recorded in each encrypted event.
    """

    def __init__(self, directory: str | Path, key: bytes | str) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as error:  # pragma: no cover - depends on extras
            raise ImportError("Install local evidence support: pip install 'face-liveness-check[evidence-local]'") from error
        self.directory = Path(directory)
        self._fernet = Fernet(key.encode("ascii") if isinstance(key, str) else key)

    @staticmethod
    def generate_key() -> bytes:
        try:
            from cryptography.fernet import Fernet
        except ImportError as error:  # pragma: no cover - depends on extras
            raise ImportError("Install local evidence support: pip install 'face-liveness-check[evidence-local]'") from error
        return Fernet.generate_key()

    def store(self, event: EvidenceEvent, artifacts: Sequence[EvidenceArtifact]) -> None:
        _validate_opaque_id(event.session_id)
        destination = self.directory / event.session_id
        destination.mkdir(parents=True, exist_ok=True)
        _write_atomic(destination / "event.json.fernet", self._fernet.encrypt(event.payload()))
        for artifact in artifacts:
            _write_atomic(destination / f"{artifact.name}.npy.fernet", self._fernet.encrypt(artifact.data))


class S3EvidenceSink:
    """Store evidence in S3 with mandatory SSE-KMS encryption.

    Install ``face-liveness-check[evidence-s3]``. Bucket policy, IAM permissions,
    lifecycle deletion, and KMS-key access remain the integrator's responsibility.
    """

    def __init__(self, bucket: str, *, kms_key_id: str, prefix: str = "face-liveness-evidence", client: object | None = None) -> None:
        if not bucket or not kms_key_id:
            raise ValueError("bucket and kms_key_id are required for encrypted S3 evidence")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if not self.prefix:
            raise ValueError("prefix must not be empty")
        self.kms_key_id = kms_key_id
        if client is None:
            try:
                import boto3
            except ImportError as error:  # pragma: no cover - depends on extras
                raise ImportError("Install S3 evidence support: pip install 'face-liveness-check[evidence-s3]'") from error
            client = boto3.client("s3")
        self._client = client

    def store(self, event: EvidenceEvent, artifacts: Sequence[EvidenceArtifact]) -> None:
        _validate_opaque_id(event.session_id)
        self._put(event.session_id, "event.json", "application/json", event.payload())
        for artifact in artifacts:
            self._put(event.session_id, f"{artifact.name}.npy", artifact.content_type, artifact.data)

    def _put(self, session_id: str, name: str, content_type: str, data: bytes) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/{session_id}/{name}",
            Body=data,
            ContentType=content_type,
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=self.kms_key_id,
        )


def frame_artifact(name: str, frame_bgr: np.ndarray) -> EvidenceArtifact:
    """Serialize an image as NPY without relying on OpenCV image codecs."""
    _validate_opaque_id(name)
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("evidence frame must be a BGR image with three channels")
    output = io.BytesIO()
    np.save(output, frame_bgr, allow_pickle=False)
    return EvidenceArtifact(name, "application/x-npy", output.getvalue())


def _validate_opaque_id(value: str) -> None:
    if not _OPAQUE_ID.fullmatch(value):
        raise ValueError("evidence identifiers must be opaque alphanumeric, hyphen, or underscore values")


def _write_atomic(destination: Path, data: bytes) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        temporary.write_bytes(data)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
