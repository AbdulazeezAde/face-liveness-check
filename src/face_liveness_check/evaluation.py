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
from typing import Mapping, Sequence, TypedDict

import numpy as np

from .adapters import FaceDetector, PassiveAntiSpoof


class _ObservationRecord(TypedDict):
    sample_id: str
    label: str
    scores: dict[str, float]


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


@dataclass(frozen=True, slots=True)
class PadThresholdMetrics:
    """Sample-level PAD metrics for one candidate at one decision threshold."""

    candidate: str
    threshold: float
    genuine_samples: int
    attack_samples: int
    print_samples: int
    replay_samples: int
    mask_samples: int
    genuine_accept_rate: float
    false_reject_rate: float
    attack_reject_rate: float
    attack_accept_rate: float
    print_reject_rate: float
    replay_reject_rate: float
    mask_reject_rate: float

    def to_record(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PadCalibrationResult:
    """A proposed threshold, or an explicit reason why none was proposed."""

    candidate: str
    threshold: float | None
    eligible: bool
    reason: str | None
    metrics: PadThresholdMetrics | None

    def to_record(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "threshold": self.threshold,
            "eligible": self.eligible,
            "reason": self.reason,
            "metrics": self.metrics.to_record() if self.metrics else None,
        }


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


def summarize_observations(path: str | Path, thresholds: Mapping[str, float]) -> dict[str, dict[str, float | int | str]]:
    """Return sample-level genuine acceptance and attack rejection metrics.

    Repeated frames from one ``sample_id`` are averaged before metrics are
    calculated. This prevents a long recording from outweighing another
    consented sample in threshold selection.
    """
    samples = _aggregate_samples(_read_observations(path))
    summary: dict[str, dict[str, float | int | str]] = {}
    for name, threshold in thresholds.items():
        summary[name] = _threshold_metrics(name, samples, threshold).to_record()
    return summary


def calibrate_thresholds(
    path: str | Path,
    *,
    candidates: Sequence[str] | None = None,
    target_genuine_accept_rate: float = 0.95,
    target_attack_reject_rate: float = 0.95,
    minimum_samples_per_label: int = 20,
) -> dict[str, PadCalibrationResult]:
    """Propose thresholds from a labelled, score-only PAD evaluation file.

    A proposal is deliberately withheld until each genuine, print, replay, and
    mask group has the configured minimum number of distinct opaque sample IDs.
    The result is a calibration aid, not a claim of PAD certification.
    """
    if not 0 < target_genuine_accept_rate <= 1 or not 0 < target_attack_reject_rate <= 1:
        raise ValueError("target acceptance and rejection rates must be in (0, 1]")
    if minimum_samples_per_label < 1:
        raise ValueError("minimum_samples_per_label must be positive")
    samples = _aggregate_samples(_read_observations(path))
    available = sorted({name for sample in samples for name in sample["scores"]})
    selected = tuple(candidates) if candidates is not None else tuple(available)
    results: dict[str, PadCalibrationResult] = {}
    for candidate in selected:
        counts = _label_counts(candidate, samples)
        missing = [label.value for label in (PadLabel.GENUINE, PadLabel.PRINT, PadLabel.REPLAY, PadLabel.MASK) if counts[label.value] < minimum_samples_per_label]
        if missing:
            results[candidate] = PadCalibrationResult(
                candidate, None, False,
                f"insufficient distinct samples for: {', '.join(missing)} (minimum {minimum_samples_per_label} each)",
                None,
            )
            continue
        scores = sorted({float(sample["scores"][candidate]) for sample in samples if candidate in sample["scores"]})
        eligible = [
            _threshold_metrics(candidate, samples, threshold)
            for threshold in scores
            if _threshold_metrics(candidate, samples, threshold).genuine_accept_rate >= target_genuine_accept_rate
            and _threshold_metrics(candidate, samples, threshold).attack_reject_rate >= target_attack_reject_rate
        ]
        if not eligible:
            results[candidate] = PadCalibrationResult(
                candidate, None, False,
                "no observed threshold met both target rates; collect representative data or revise the targets",
                None,
            )
            continue
        # Prefer stopping more observed attacks; break ties in favour of fewer
        # genuine rejections and then the lower threshold.
        selected_metrics = max(
            eligible,
            key=lambda metric: (metric.attack_reject_rate, metric.genuine_accept_rate, -metric.threshold),
        )
        results[candidate] = PadCalibrationResult(candidate, selected_metrics.threshold, True, None, selected_metrics)
    return results


def _read_observations(path: str | Path) -> list[_ObservationRecord]:
    records: list[_ObservationRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            sample_id, label, scores = record["sample_id"], record["label"], record["scores"]
            if not isinstance(sample_id, str) or not isinstance(label, str):
                raise ValueError("sample_id and label must be strings")
            _validate_sample_id(sample_id)
            PadLabel(label)
            if not isinstance(scores, dict):
                raise ValueError("scores must be an object")
            normalized_scores = {str(name): float(score) for name, score in scores.items()}
            if any(not 0 <= score <= 1 for score in normalized_scores.values()):
                raise ValueError("scores must be in [0, 1]")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid PAD observation at line {line_number}: {error}") from error
        records.append({"sample_id": sample_id, "label": label, "scores": normalized_scores})
    if not records:
        raise ValueError("PAD observation file contains no records")
    return records


def _aggregate_samples(records: Sequence[_ObservationRecord]) -> list[_ObservationRecord]:
    grouped: dict[str, tuple[str, dict[str, list[float]]]] = {}
    for record in records:
        sample_id, label = record["sample_id"], record["label"]
        existing = grouped.get(sample_id)
        if existing is None:
            existing = (label, {})
            grouped[sample_id] = existing
        existing_label, buckets = existing
        if existing_label != label:
            raise ValueError(f"sample_id {sample_id!r} has inconsistent labels")
        for name, score in record["scores"].items():
            buckets.setdefault(name, []).append(float(score))
    return [
        {"sample_id": sample_id, "label": label, "scores": {name: float(np.mean(scores)) for name, scores in buckets.items()}}
        for sample_id, (label, buckets) in grouped.items()
    ]


def _label_counts(candidate: str, samples: Sequence[_ObservationRecord]) -> dict[str, int]:
    return {
        label.value: sum(sample["label"] == label.value and candidate in sample["scores"] for sample in samples)
        for label in (PadLabel.GENUINE, PadLabel.PRINT, PadLabel.REPLAY, PadLabel.MASK)
    }


def _threshold_metrics(candidate: str, samples: Sequence[_ObservationRecord], threshold: float) -> PadThresholdMetrics:
    if not 0 <= threshold <= 1:
        raise ValueError("PAD thresholds must be in [0, 1]")
    grouped = {
        label.value: [float(sample["scores"][candidate]) for sample in samples if sample["label"] == label.value and candidate in sample["scores"]]
        for label in (PadLabel.GENUINE, PadLabel.PRINT, PadLabel.REPLAY, PadLabel.MASK)
    }
    genuine, print_scores, replay_scores, mask_scores = (grouped[label.value] for label in (PadLabel.GENUINE, PadLabel.PRINT, PadLabel.REPLAY, PadLabel.MASK))
    attacks = print_scores + replay_scores + mask_scores
    def reject_rate(scores: Sequence[float]) -> float:
        return sum(score < threshold for score in scores) / len(scores) if scores else 0.0
    genuine_accept_rate = sum(score >= threshold for score in genuine) / len(genuine) if genuine else 0.0
    attack_reject_rate = reject_rate(attacks)
    return PadThresholdMetrics(
        candidate=candidate, threshold=threshold, genuine_samples=len(genuine), attack_samples=len(attacks),
        print_samples=len(print_scores), replay_samples=len(replay_scores), mask_samples=len(mask_scores),
        genuine_accept_rate=genuine_accept_rate, false_reject_rate=1 - genuine_accept_rate if genuine else 0.0,
        attack_reject_rate=attack_reject_rate, attack_accept_rate=1 - attack_reject_rate if attacks else 0.0,
        print_reject_rate=reject_rate(print_scores), replay_reject_rate=reject_rate(replay_scores), mask_reject_rate=reject_rate(mask_scores),
    )


def _validate_sample_id(sample_id: str) -> None:
    if not sample_id or any(marker in sample_id for marker in ("/", "\\", ":")):
        raise ValueError("sample_id must be a non-empty opaque identifier, not a file path")
