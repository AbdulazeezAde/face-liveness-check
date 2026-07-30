"""Public types shared between model adapters and the liveness engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
