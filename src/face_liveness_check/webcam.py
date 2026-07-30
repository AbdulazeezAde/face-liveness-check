"""Interactive webcam verification for the package CLI and simple demos."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .verifier import LivenessVerifier, VerificationRun


def verify_webcam(
    verifier: LivenessVerifier,
    reference_image_bgr: np.ndarray,
    *,
    source: int | str = 0,
    duration_s: float = 15.0,
    preview: bool = True,
    on_challenges: Callable[[tuple[str, ...]], None] | None = None,
    evidence_consent: bool = False,
    session_id: str | None = None,
) -> VerificationRun:
    """Run a short webcam session; press ``q`` in the preview to finish early."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    cv2 = _opencv()
    live = verifier.start(reference_image_bgr, evidence_consent=evidence_consent, session_id=session_id)
    if on_challenges:
        on_challenges(live.challenges)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"could not open video source: {source!r}")
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration_s:
            ok, frame = capture.read()
            if not ok:
                break
            elapsed = time.monotonic() - started
            live.observe(frame, elapsed)
            if preview:
                label = " -> ".join(challenge.replace("_", " ") for challenge in live.challenges)
                cv2.putText(frame, f"Do in order: {label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(frame, "Press q to finish", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("Face liveness check", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if preview:
            cv2.destroyAllWindows()
    return live.finish()


def _opencv() -> object:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Install OpenCV support: pip install 'face-liveness-check[opencv]'") from error
    return cv2
