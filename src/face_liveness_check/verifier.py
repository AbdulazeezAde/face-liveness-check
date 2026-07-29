"""End-to-end reference extraction, live-stream liveness, and identity matching."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .models import LivenessPolicy, VerificationResult
from .pipeline import FrameEvidenceBuilder, ReferenceExtractor
from .session import LivenessSession


@dataclass(frozen=True, slots=True)
class VerificationRun:
    result: VerificationResult
    challenges: tuple[str, ...]


class LivenessVerifier:
    """Reusable verifier. A caller owns UI prompts and supplies timestamped BGR frames."""

    def __init__(self, reference_extractor: ReferenceExtractor, evidence_builder: FrameEvidenceBuilder,
                 policy: LivenessPolicy | None = None) -> None:
        self.reference_extractor = reference_extractor
        self.evidence_builder = evidence_builder
        self.policy = policy or LivenessPolicy()

    def verify(self, reference_image_bgr: np.ndarray,
               frames: Iterable[tuple[float, np.ndarray]]) -> VerificationRun:
        reference = self.reference_extractor.extract(reference_image_bgr)
        session = LivenessSession(self.policy)
        for timestamp_s, frame_bgr in frames:
            session.observe(self.evidence_builder.build(frame_bgr, timestamp_s))
        result = session.compare(reference.embedding)
        return VerificationRun(result, tuple(challenge.value for challenge in session.challenges))
