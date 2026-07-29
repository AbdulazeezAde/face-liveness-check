"""Stateful active-liveness session and embedding comparison."""

from __future__ import annotations

import secrets

import numpy as np

from .models import Challenge, FrameEvidence, LivenessPolicy, LivenessResult, VerificationResult


class LivenessSession:
    """Consumes model evidence in timestamp order for one short-lived session."""

    def __init__(self, policy: LivenessPolicy | None = None) -> None:
        self.policy = policy or LivenessPolicy()
        if not 1 <= self.policy.challenge_count <= len(Challenge):
            raise ValueError("challenge_count must be between 1 and 4")
        self.challenges = tuple(secrets.SystemRandom().sample(list(Challenge), self.policy.challenge_count))
        self._completed: list[Challenge] = []
        self._reasons: list[str] = []
        self._frames: list[FrameEvidence] = []
        self._tracking_id: str | None = None

    def observe(self, evidence: FrameEvidence) -> None:
        """Record one frame's evidence; observations must be chronological."""
        if self._frames and evidence.timestamp_s <= self._frames[-1].timestamp_s:
            raise ValueError("frame timestamps must be strictly increasing")
        self._frames.append(evidence)

        if evidence.face_count != 1:
            self._add_reason("expected exactly one face throughout the session")
            return
        if evidence.tracking_id:
            if self._tracking_id is None:
                self._tracking_id = evidence.tracking_id
            elif evidence.tracking_id != self._tracking_id:
                self._add_reason("face tracking identity changed during the session")
                return

        expected = self.challenges[len(self._completed)] if len(self._completed) < len(self.challenges) else None
        if expected is not None and evidence.motion == expected:
            self._completed.append(expected)

    def result(self) -> LivenessResult:
        reasons = list(self._reasons)
        valid_faces = [frame for frame in self._frames if frame.face_count == 1]
        if len(valid_faces) < self.policy.min_face_frames:
            reasons.append("not enough single-face frames")
        if len(self._completed) != len(self.challenges):
            reasons.append("active challenge sequence was not completed")
        if valid_faces:
            if min(f.quality_score for f in valid_faces) < self.policy.min_quality_score:
                reasons.append("face quality is too low")
            if min(f.lighting_score for f in valid_faces) < self.policy.min_lighting_score:
                reasons.append("lighting is insufficient for reliable analysis")
            passive = [f.passive_antispoof_score for f in valid_faces if f.passive_antispoof_score is not None]
            if not passive:
                reasons.append("passive anti-spoof evidence is missing")
            elif float(np.median(passive)) < self.policy.min_passive_score:
                reasons.append("passive anti-spoof check failed")
            fingerprints = [f.frame_fingerprint for f in valid_faces if f.frame_fingerprint]
            if fingerprints and 1 - len(set(fingerprints)) / len(fingerprints) > self.policy.max_duplicate_ratio:
                reasons.append("too many duplicate video frames")
        else:
            reasons.append("no usable single-face frame")

        confidence = self._confidence(valid_faces, reasons)
        return LivenessResult(not reasons, confidence, tuple(self._completed), tuple(reasons), len(self._frames))

    def compare(self, reference_embedding: np.ndarray) -> VerificationResult:
        """Compare reference embedding only after liveness evaluates successfully."""
        liveness = self.result()
        if not liveness.passed:
            return VerificationResult(False, None, liveness, ("liveness did not pass",))
        embeddings = [f.embedding for f in self._frames if f.embedding is not None and f.face_count == 1]
        if len(embeddings) < self.policy.minimum_live_embeddings:
            return VerificationResult(False, None, liveness, ("not enough live embeddings",))
        reference = _normalize(reference_embedding)
        live = _normalize(np.mean(np.stack(embeddings), axis=0))
        similarity = float(np.dot(reference, live))
        return VerificationResult(similarity >= self.policy.min_match_similarity, similarity, liveness)

    def _confidence(self, valid_faces: list[FrameEvidence], reasons: list[str]) -> float:
        if not valid_faces:
            return 0.0
        values = [f.quality_score for f in valid_faces] + [f.lighting_score for f in valid_faces]
        values += [f.passive_antispoof_score for f in valid_faces if f.passive_antispoof_score is not None]
        base = float(np.mean(values)) if values else 0.0
        sequence = len(self._completed) / len(self.challenges)
        return round(max(0.0, min(1.0, base * sequence * (0.5 if reasons else 1.0))), 4)

    def _add_reason(self, reason: str) -> None:
        if reason not in self._reasons:
            self._reasons.append(reason)


def _normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("embedding must not be a zero vector")
    return array / norm
