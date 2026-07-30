"""Privacy-preserving PAD evaluation utilities.

The evaluator records labels, model scores, and face-count metadata only. It
never writes frames, face crops, embeddings, document paths, or identity data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .adapters import FaceDetector, PassiveAntiSpoof


class PadLabel(str, Enum):
    GENUINE = "genuine"
    PRINT = "print"
    REPLAY = "replay"
    MASK = "mask"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PadCandidate:
    """A named scorer and the crop margin required by its training pipeline."""

    name: str
    scorer: PassiveAntiSpoof
    crop_padding: float

    def __post_init__(self) -> None:
        if not self.name or self.crop_padding < 0:
            raise ValueError("candidate name must be non-empty and crop_padding non-negative")


@dataclass(frozen=True, slots=True)
class PadObservation:
    sample_id: str
    label: PadLabel
    observed_at: str
    face_count: int
    scores: Mapping[str, float]

    def to_record(self) -> dict[str, object]:
        return asdict(self)


class PadEvaluator:
    """Apply multiple PAD candidates to the same frame for fair comparison."""

    def __init__(self, detector: FaceDetector, candidates: Sequence[PadCandidate]) -> None:
        if not candidates:
            raise ValueError("at least one PAD candidate is required")
        names = [candidate.name for candidate in candidates]
        if len(names) != len(set(names)):
            raise ValueError("PAD candidate names must be unique")
        self.detector, self.candidates = detector, tuple(candidates)

    def observe(self, frame_bgr: np.ndarray, *, sample_id: str, label: PadLabel) -> PadObservation:
        _validate_sample_id(sample_id)
        detections = list(self.detector.detect(frame_bgr))
        scores: dict[str, float] = {}
        if len(detections) == 1:
            detection = detections[0]
            for candidate in self.candidates:
                score = candidate.scorer.score(detection.crop(frame_bgr, padding=candidate.crop_padding))
                if not 0 <= score <= 1:
                    raise ValueError(f"{candidate.name} returned a PAD score outside [0, 1]")
                scores[candidate.name] = float(score)
        return PadObservation(
            sample_id=sample_id,
            label=label,
            observed_at=datetime.now(timezone.utc).isoformat(),
            face_count=len(detections),
            scores=scores,
        )


def append_observation(path: str | Path, observation: PadObservation) -> Path:
    """Append score-only JSONL; the caller is responsible for label consent."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(observation.to_record(), sort_keys=True) + "\n")
    return destination


def summarize_observations(path: str | Path, thresholds: Mapping[str, float]) -> dict[str, dict[str, float | int]]:
    """Return genuine acceptance and attack rejection rates from score-only JSONL."""
    observations = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary: dict[str, dict[str, float | int]] = {}
    for name, threshold in thresholds.items():
        if not 0 <= threshold <= 1:
            raise ValueError("PAD thresholds must be in [0, 1]")
        genuine = [row for row in observations if row["label"] == PadLabel.GENUINE.value and name in row["scores"]]
        attacks = [row for row in observations if row["label"] in {PadLabel.PRINT.value, PadLabel.REPLAY.value, PadLabel.MASK.value} and name in row["scores"]]
        genuine_accepted = sum(row["scores"][name] >= threshold for row in genuine)
        attacks_rejected = sum(row["scores"][name] < threshold for row in attacks)
        summary[name] = {
            "genuine_samples": len(genuine),
            "attack_samples": len(attacks),
            "genuine_accept_rate": genuine_accepted / len(genuine) if genuine else 0.0,
            "attack_reject_rate": attacks_rejected / len(attacks) if attacks else 0.0,
        }
    return summary


def _validate_sample_id(sample_id: str) -> None:
    if not sample_id or any(marker in sample_id for marker in ("/", "\\", ":")):
        raise ValueError("sample_id must be a non-empty opaque identifier, not a file path")
