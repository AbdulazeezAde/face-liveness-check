"""End-to-end reference extraction, live-stream liveness, and identity matching."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .models import LivenessPolicy, VerificationResult
from .pipeline import FrameEvidenceBuilder, ReferenceExtractor
from .session import LivenessSession


@dataclass(frozen=True, slots=True)
class VerificationRun:
    result: VerificationResult
    challenges: tuple[str, ...]


@dataclass(slots=True)
class LiveVerification:
    """An interactive verification session with challenges available before capture."""

    verifier: "LivenessVerifier"
    reference_embedding: np.ndarray
    session: LivenessSession

    @property
    def challenges(self) -> tuple[str, ...]:
        return tuple(challenge.value for challenge in self.session.challenges)

    def observe(self, frame_bgr: np.ndarray, timestamp_s: float) -> None:
        self.session.observe(self.verifier.evidence_builder.build(frame_bgr, timestamp_s))

    def finish(self) -> VerificationRun:
        return VerificationRun(self.session.compare(self.reference_embedding), self.challenges)


class LivenessVerifier:
    """Reusable verifier. A caller owns UI prompts and supplies timestamped BGR frames."""

    def __init__(self, reference_extractor: ReferenceExtractor, evidence_builder: FrameEvidenceBuilder,
                 policy: LivenessPolicy | None = None) -> None:
        self.reference_extractor = reference_extractor
        self.evidence_builder = evidence_builder
        self.policy = policy or LivenessPolicy()

    @classmethod
    def from_model_pack(cls, name: str, *, manager: object, factory: Callable[[object], "LivenessVerifier"],
                        download: bool = True, accept_model_license: bool = False) -> "LivenessVerifier":
        """Resolve a verified pack, then let an application factory build adapters."""
        installed = (
            manager.install(name, accept_model_license=accept_model_license)
            if download else manager.resolve(name)
        )
        return factory(installed)

    def start(self, reference_image_bgr: np.ndarray) -> LiveVerification:
        """Enroll the reference and create a session before showing camera prompts."""
        reference = self.reference_extractor.extract(reference_image_bgr)
        return LiveVerification(self, reference.embedding, LivenessSession(self.policy))

    def verify(self, reference_image_bgr: np.ndarray,
               frames: Iterable[tuple[float, np.ndarray]]) -> VerificationRun:
        live = self.start(reference_image_bgr)
        for timestamp_s, frame_bgr in frames:
            live.observe(frame_bgr, timestamp_s)
        return live.finish()
