"""Optional OpenCV video capture helpers."""
from collections.abc import Iterator
from typing import Any

def iter_frames(source: int | str, max_frames: int | None = None) -> Iterator[tuple[float, Any]]:
    try:
        import cv2
    except ImportError as error:
        raise ImportError("Install the OpenCV extra: pip install 'face-liveness-check[opencv]'") from error
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"could not open video source: {source!r}")
    try:
        count = 0
        while max_frames is None or count < max_frames:
            ok, frame = capture.read()
            if not ok: break
            yield capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, frame
            count += 1
    finally:
        capture.release()
