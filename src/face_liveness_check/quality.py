"""OpenCV frame-quality and replay indicators used by liveness pipelines."""

from __future__ import annotations

import numpy as np


def lighting_score(face_bgr: np.ndarray) -> float:
    """Score usable exposure from 0 to 1; this is not a daylight requirement."""
    gray = _opencv().cvtColor(face_bgr, _opencv().COLOR_BGR2GRAY)
    mean = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 64.0
    exposure = max(0.0, 1.0 - abs(mean - 0.5) / 0.5)
    return round(min(1.0, 0.7 * exposure + 0.3 * min(1.0, contrast)), 4)


def blur_score(face_bgr: np.ndarray, reference_variance: float = 150.0) -> float:
    """Convert Laplacian variance into a bounded sharpness score."""
    cv2 = _opencv()
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(min(1.0, variance / reference_variance), 4)


def perceptual_fingerprint(frame_bgr: np.ndarray, size: int = 8) -> str:
    """Return a compact dHash for duplicate/replay-frame detection."""
    cv2 = _opencv()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).astype(np.uint8).reshape(-1)
    return ''.join('1' if bit else '0' for bit in bits)


def _opencv() -> object:
    try:
        import cv2
    except ImportError as error:
        raise ImportError("Install OpenCV support: pip install 'face-liveness-check[opencv]'") from error
    return cv2
