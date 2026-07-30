"""Public types shared between model adapters and the liveness engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Challenge(str, Enum):
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NOD = "nod"


class PassiveAntiSpoofMode(str, Enum):
    """How passive PAD contributes to the verification decision."""

    REQUIRED = "required"
    ADVISORY = "advisory"


class VerificationProfileName(str, Enum):
    """Named decision modes with intentionally different operational meaning."""

    STRICT = "strict"
    ACTIVE_FIRST = "active_first"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True)
class LivenessPolicy:
    """Thresholds must be calibrated using genuine and attack samples."""

    challenge_count: int = 3
    min_passive_score: float = 0.80
    min_quality_score: float = 0.55
    min_lighting_score: float = 0.35
    min_face_frames: int = 12
    max_duplicate_ratio: float = 0.20
    min_match_similarity: float = 0.42
    minimum_live_embeddings: int = 3
    passive_antispoof_mode: PassiveAntiSpoofMode = PassiveAntiSpoofMode.REQUIRED


def active_first_policy(**overrides: Any) -> LivenessPolicy:
    """Create a policy where PAD produces a warning rather than an auto-reject.

    Active randomized challenges, face quality, duplicate-frame detection, and
    identity matching remain required. Pair this policy with an opt-in
    ``EvidencePolicy`` when suspicious sessions must be retained.
    """
    overrides.setdefault("passive_antispoof_mode", PassiveAntiSpoofMode.ADVISORY)
    return LivenessPolicy(**overrides)


@dataclass(frozen=True, slots=True)
class VerificationProfile:
    """A policy plus whether its output may be used for an automatic decision."""

    name: VerificationProfileName
    policy: LivenessPolicy
    allows_automatic_decision: bool


def strict_profile(**overrides: Any) -> VerificationProfile:
    """Require all configured signals, including passive PAD, for a decision."""
    return VerificationProfile(VerificationProfileName.STRICT, LivenessPolicy(**overrides), True)


def active_first_profile(**overrides: Any) -> VerificationProfile:
    """Use active challenges as the primary gate and flag PAD concerns for review."""
    return VerificationProfile(VerificationProfileName.ACTIVE_FIRST, active_first_policy(**overrides), True)


def evaluation_profile(**overrides: Any) -> VerificationProfile:
    """Run the active-first signal path but mark results as evaluation-only.

    This is suitable for calibration and integration exercises, never for an
    automated acceptance or rejection decision.
    """
    return VerificationProfile(VerificationProfileName.EVALUATION, active_first_policy(**overrides), False)


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    """Evidence produced by application-owned vision model adapters for one frame.

    ``motion`` is cumulative or per-frame detected activity; a blink is counted once
    after an eye-open → eye-closed → eye-open transition by the landmark adapter.
    ``frame_fingerprint`` should be a perceptual hash to flag replayed frames.
    """

    timestamp_s: float
    face_count: int
    tracking_id: str | None
    quality_score: float
    lighting_score: float
    passive_antispoof_score: float | None
    motion: Challenge | None = None
    embedding: np.ndarray | None = None
    frame_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class LivenessResult:
    passed: bool
    confidence: float
    completed_challenges: tuple[Challenge, ...]
    reasons: tuple[str, ...]
    frames_seen: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    matched: bool
    similarity: float | None
    liveness: LivenessResult
    reasons: tuple[str, ...] = field(default_factory=tuple)
