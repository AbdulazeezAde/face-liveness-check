"""Optional local barcode adapters for document extraction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np

from .id_document import BarcodePayload, ExtractedField, FieldSource


class ZxingBarcodeReader:
    """Decode QR, PDF417, and other supported barcodes locally with zxing-cpp."""

    def __init__(self, decoder: object | None = None) -> None:
        if decoder is None:
            try:
                import zxingcpp
            except ImportError as error:  # pragma: no cover - optional dependency
                raise ImportError("Install barcode support: pip install 'face-liveness-check[id-ocr]'") from error
            decoder = zxingcpp.read_barcodes
        self._decoder = decoder

    def read(self, image_bgr: np.ndarray) -> tuple[BarcodePayload, ...]:
        payloads: list[BarcodePayload] = []
        for result in self._decoder(image_bgr):
            text = str(getattr(result, "text", "")).strip()
            barcode_format = str(getattr(result, "format", "unknown")).strip()
            if text:
                payloads.append(BarcodePayload(barcode_format, text))
        return tuple(payloads)


class JsonBarcodeFieldParser:
    """Opt-in parser for barcodes whose payload is a flat JSON field mapping.

    It intentionally ignores unknown payload formats. Integrators should provide
    a document-specific parser for PDF417, CBOR, encrypted, or signed barcodes.
    """

    def parse(self, barcodes: Sequence[BarcodePayload]) -> Mapping[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        for barcode in barcodes:
            try:
                payload = json.loads(barcode.text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            for name, value in payload.items():
                if isinstance(name, str) and isinstance(value, (str, int, float)):
                    fields[name] = ExtractedField(str(value), 1.0, FieldSource.BARCODE, True)
        return fields
