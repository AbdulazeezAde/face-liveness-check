"""Read portraits from image files or the first page of an identity document."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .adapters import FaceDetection, FaceDetector


def load_reference_image_bgr(path: str | Path, *, pdf_page: int = 0, dpi: int = 200) -> np.ndarray:
    """Decode an image, or rasterize one page of a PDF, as an OpenCV BGR image."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".pdf":
        return _render_pdf_page_bgr(source, pdf_page, dpi)
    cv2 = _opencv()
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode reference image: {source}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise ValueError(f"reference image must have 1, 3, or 4 channels: {source}")
    return image


def extract_reference_face_crop(image_bgr: np.ndarray, detector: FaceDetector, *, padding: float = 0.10) -> np.ndarray:
    """Return the only detected portrait face; reject ambiguous source documents."""
    detections = list(detector.detect(image_bgr))
    if len(detections) != 1:
        raise ValueError(f"reference document must contain exactly one face; found {len(detections)}")
    return detections[0].crop(image_bgr, padding=padding)


def extract_reference_face_crop_file(
    source: str | Path, destination: str | Path, detector: FaceDetector, *,
    pdf_page: int = 0, dpi: int = 200, padding: float = 0.10,
) -> Path:
    """Extract a portrait from an image/PDF ID and save it as a lossless PNG."""
    crop = extract_reference_face_crop(
        load_reference_image_bgr(source, pdf_page=pdf_page, dpi=dpi), detector, padding=padding,
    )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() != ".png":
        raise ValueError("reference face crop destination must end in .png")
    if not _opencv().imwrite(str(target), crop):
        raise OSError(f"could not write face crop: {target}")
    return target


def _render_pdf_page_bgr(source: Path, pdf_page: int, dpi: int) -> np.ndarray:
    if pdf_page < 0 or dpi <= 0:
        raise ValueError("pdf_page must be non-negative and dpi must be positive")
    try:
        import fitz
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Install document support: pip install 'face-liveness-check[documents]'") from error
    document = fitz.open(source)
    try:
        if pdf_page >= document.page_count:
            raise ValueError(f"PDF page {pdf_page} does not exist; document has {document.page_count} pages")
        pixmap = document.load_page(pdf_page).get_pixmap(dpi=dpi, alpha=False)
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        return _opencv().cvtColor(rgb, _opencv().COLOR_RGB2BGR)
    finally:
        document.close()


def _opencv() -> object:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Install OpenCV support: pip install 'face-liveness-check[opencv]'") from error
    return cv2
