"""Optional OCR adapters for identity-document extraction."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .id_document import OcrTextBlock


class PaddleOcrEngine:
    """PaddleOCR 3.x adapter using local inference and in-memory BGR images.

    PaddleOCR owns its own model download/cache. Pass an already configured
    runner to control model locations, model versions, and inference hardware.
    The adapter never writes source documents or OCR responses.
    """

    def __init__(self, runner: object | None = None) -> None:
        if runner is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as error:  # pragma: no cover - optional dependency
                raise ImportError("Install ID OCR support: pip install 'face-liveness-check[id-ocr]'") from error
            runner = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="paddle",
            )
        self._runner = runner

    def read(self, image_bgr: np.ndarray) -> tuple[OcrTextBlock, ...]:
        predictions = self._runner.predict(image_bgr)
        blocks: list[OcrTextBlock] = []
        for prediction in predictions:
            payload = _prediction_payload(prediction)
            result = payload.get("res", payload)
            texts, scores, polygons = result.get("rec_texts", ()), result.get("rec_scores", ()), result.get("rec_polys", ())
            for index, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    continue
                confidence = float(scores[index]) if index < len(scores) else 0.0
                polygon = _polygon(polygons[index]) if index < len(polygons) else ()
                blocks.append(OcrTextBlock(text.strip(), min(1.0, max(0.0, confidence)), polygon))
        return tuple(blocks)


def _prediction_payload(prediction: object) -> Mapping[str, Any]:
    if isinstance(prediction, Mapping):
        return prediction
    value = getattr(prediction, "json", None)
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        return value
    raise ValueError("PaddleOCR prediction did not expose a JSON-compatible result")


def _polygon(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Iterable):
        return ()
    points: list[tuple[float, float]] = []
    for point in value:
        if isinstance(point, Iterable):
            coordinates = tuple(point)
            if len(coordinates) >= 2:
                points.append((float(coordinates[0]), float(coordinates[1])))
    return tuple(points)
