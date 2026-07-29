"""Public evidence and result types."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class Challenge(str, Enum):
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NOD = "nod"

@dataclass(frozen=True, slots=True)
class LivenessPolicy:
    challenge_count: int = 3
    min_passive_score: float = 0.80
    min_quality_score: float = 0.55
    min_lighting_score: float = 0.35
    min_face_frames: int = 12
    max_duplicate_ratio: float = 0.20
    min_match_similarity: float = 0.42
    minimum_live_embeddings: int = 3

@dataclass(frozen=True, slots=True)
class FrameEvidence:
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

@dataclass(frozen=True, slots=True)
class VerificationResult:
    matched: bool
    similarity: float | None
    liveness: LivenessResult
    reasons: tuple[str, ...] = field(default_factory=tuple)
