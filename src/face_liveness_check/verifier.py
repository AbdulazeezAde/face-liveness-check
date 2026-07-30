"""End-to-end reference extraction, live-stream liveness, and identity matching."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

import numpy as np

from .evidence import EvidenceEvent, EvidencePolicy, EvidenceSink, frame_artifact
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
    evidence_consent: bool = False
    session_id: str = field(default_factory=lambda: uuid4().hex)
    _evidence_frames: deque[np.ndarray] = field(default_factory=deque, repr=False)

    @property
    def challenges(self) -> tuple[str, ...]:
        return tuple(challenge.value for challenge in self.session.challenges)

    def observe(self, frame_bgr: np.ndarray, timestamp_s: float) -> None:
        self.session.observe(self.verifier.evidence_builder.build(frame_bgr, timestamp_s))
        policy = self.verifier.evidence_policy
        if policy.enabled and (policy.capture_frames or policy.capture_face_crops):
            self._evidence_frames.append(frame_bgr.copy())
            while len(self._evidence_frames) > policy.max_frames:
                self._evidence_frames.popleft()

    def finish(self) -> VerificationRun:
        run = VerificationRun(self.session.compare(self.reference_embedding), self.challenges)
        self._store_evidence(run)
        return run

    def _store_evidence(self, run: VerificationRun) -> None:
        policy, sink = self.verifier.evidence_policy, self.verifier.evidence_sink
        if not policy.enabled:
            return
        categories: list[str] = []
        if not run.result.matched or not run.result.liveness.passed:
            categories.append("failed")
        if run.result.liveness.warnings:
            categories.append("suspicious")
        if not categories:
            categories.append("passed")
        categories = [category for category in categories if category in policy.capture_on]
        if not categories:
            return
        artifacts = []
        for index, frame in enumerate(self._evidence_frames):
            if policy.capture_frames:
                artifacts.append(frame_artifact(f"frame_{index:03d}", frame))
            if policy.capture_face_crops:
                crop = self.verifier.evidence_builder.extract_face_crop(frame)
                if crop is not None:
                    artifacts.append(frame_artifact(f"face_{index:03d}", crop))
        event = EvidenceEvent.create(
            session_id=self.session_id,
            categories=categories,
            matched=run.result.matched,
            liveness_passed=run.result.liveness.passed,
            similarity=run.result.similarity,
            liveness_reasons=run.result.liveness.reasons,
            liveness_warnings=run.result.liveness.warnings,
            retention_days=policy.retention_days,
        )
        sink.store(event, artifacts)


class LivenessVerifier:
    """Reusable verifier. A caller owns UI prompts and supplies timestamped BGR frames."""

    def __init__(self, reference_extractor: ReferenceExtractor, evidence_builder: FrameEvidenceBuilder,
                 policy: LivenessPolicy | None = None, *, evidence_policy: EvidencePolicy | None = None,
                 evidence_sink: EvidenceSink | None = None) -> None:
        self.reference_extractor = reference_extractor
        self.evidence_builder = evidence_builder
        self.policy = policy or LivenessPolicy()
        self.evidence_policy = evidence_policy or EvidencePolicy()
        self.evidence_sink = evidence_sink
        if self.evidence_policy.enabled and evidence_sink is None:
            raise ValueError("an evidence_sink is required when evidence capture is enabled")

    @classmethod
    def from_model_pack(cls, name: str, *, manager: object, factory: Callable[[object], "LivenessVerifier"],
                        download: bool = True, accept_model_license: bool = False) -> "LivenessVerifier":
        """Resolve a verified pack, then let an application factory build adapters."""
        installed = (
            manager.install(name, accept_model_license=accept_model_license)
            if download else manager.resolve(name)
        )
        return factory(installed)

    def start(self, reference_image_bgr: np.ndarray, *, evidence_consent: bool = False,
              session_id: str | None = None) -> LiveVerification:
        """Enroll the reference and create a session before showing camera prompts."""
        if self.evidence_policy.enabled and self.evidence_policy.require_consent and not evidence_consent:
            raise PermissionError("evidence capture requires explicit evidence_consent=True")
        reference = self.reference_extractor.extract(reference_image_bgr)
        return LiveVerification(
            self, reference.embedding, LivenessSession(self.policy), evidence_consent,
            session_id or uuid4().hex,
        )

    def verify(self, reference_image_bgr: np.ndarray,
               frames: Iterable[tuple[float, np.ndarray]], *, evidence_consent: bool = False,
               session_id: str | None = None) -> VerificationRun:
        live = self.start(reference_image_bgr, evidence_consent=evidence_consent, session_id=session_id)
        for timestamp_s, frame_bgr in frames:
            live.observe(frame_bgr, timestamp_s)
        return live.finish()
