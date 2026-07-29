"""Stateful active-liveness orchestration and face comparison."""
from __future__ import annotations
import secrets
import numpy as np
from .models import Challenge, FrameEvidence, LivenessPolicy, LivenessResult, VerificationResult

class LivenessSession:
    def __init__(self, policy: LivenessPolicy | None = None) -> None:
        self.policy = policy or LivenessPolicy()
        if not 1 <= self.policy.challenge_count <= len(Challenge):
            raise ValueError("challenge_count must be between 1 and 4")
        self.challenges = tuple(secrets.SystemRandom().sample(list(Challenge), self.policy.challenge_count))
        self._frames: list[FrameEvidence] = []
        self._completed: list[Challenge] = []
        self._reasons: list[str] = []
        self._tracking_id: str | None = None

    def observe(self, evidence: FrameEvidence) -> None:
        if self._frames and evidence.timestamp_s <= self._frames[-1].timestamp_s:
            raise ValueError("frame timestamps must be strictly increasing")
        self._frames.append(evidence)
        if evidence.face_count != 1:
            self._reason("expected exactly one face throughout the session")
            return
        if evidence.tracking_id:
            if self._tracking_id is None:
                self._tracking_id = evidence.tracking_id
            elif evidence.tracking_id != self._tracking_id:
                self._reason("face tracking identity changed during the session")
                return
        if len(self._completed) < len(self.challenges) and evidence.motion == self.challenges[len(self._completed)]:
            self._completed.append(evidence.motion)

    def result(self) -> LivenessResult:
        reasons = list(self._reasons)
        frames = [f for f in self._frames if f.face_count == 1]
        if len(frames) < self.policy.min_face_frames: reasons.append("not enough single-face frames")
        if len(self._completed) != len(self.challenges): reasons.append("active challenge sequence was not completed")
        if frames:
            if min(f.quality_score for f in frames) < self.policy.min_quality_score: reasons.append("face quality is too low")
            if min(f.lighting_score for f in frames) < self.policy.min_lighting_score: reasons.append("lighting is insufficient for reliable analysis")
            passive = [f.passive_antispoof_score for f in frames if f.passive_antispoof_score is not None]
            if not passive: reasons.append("passive anti-spoof evidence is missing")
            elif float(np.median(passive)) < self.policy.min_passive_score: reasons.append("passive anti-spoof check failed")
            hashes = [f.frame_fingerprint for f in frames if f.frame_fingerprint]
            if hashes and 1 - len(set(hashes)) / len(hashes) > self.policy.max_duplicate_ratio: reasons.append("too many duplicate video frames")
        else: reasons.append("no usable single-face frame")
        reasons = list(dict.fromkeys(reasons))
        return LivenessResult(not reasons, self._confidence(frames, reasons), tuple(self._completed), tuple(reasons), len(self._frames))

    def compare(self, reference_embedding: np.ndarray) -> VerificationResult:
        live_result = self.result()
        if not live_result.passed: return VerificationResult(False, None, live_result, ("liveness did not pass",))
        vectors = [f.embedding for f in self._frames if f.embedding is not None and f.face_count == 1]
        if len(vectors) < self.policy.minimum_live_embeddings:
            return VerificationResult(False, None, live_result, ("not enough live embeddings",))
        score = float(np.dot(_normalize(reference_embedding), _normalize(np.mean(np.stack(vectors), axis=0))))
        return VerificationResult(score >= self.policy.min_match_similarity, score, live_result)

    def _confidence(self, frames, reasons) -> float:
        if not frames: return 0.0
        values = [f.quality_score for f in frames] + [f.lighting_score for f in frames]
        values += [f.passive_antispoof_score for f in frames if f.passive_antispoof_score is not None]
        factor = len(self._completed) / len(self.challenges)
        return round(max(0.0, min(1.0, float(np.mean(values)) * factor * (.5 if reasons else 1))), 4)

    def _reason(self, text: str) -> None:
        if text not in self._reasons: self._reasons.append(text)

def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm == 0: raise ValueError("embedding must not be a zero vector")
    return value / norm
