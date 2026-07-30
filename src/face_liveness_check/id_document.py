"""Privacy-preserving identity-document extraction primitives.

The module performs all work locally and returns extracted content in memory.
It never writes document images, text, portrait crops, barcode data, or OCR
blocks unless an integrating application explicitly chooses to store them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .adapters import FaceDetector
from .reference_io import load_reference_image_bgr


class DocumentType(str, Enum):
    UNKNOWN = "unknown"
    PASSPORT_TD3 = "passport_td3"
    CARD = "card"
    NIGERIA_NIN_SLIP = "nigeria_nin_slip"


class FieldSource(str, Enum):
    OCR = "ocr"
    MRZ = "mrz"
    BARCODE = "barcode"


@dataclass(frozen=True, slots=True)
class OcrTextBlock:
    """One recognised text region, with its source-space quadrilateral."""

    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("OCR text must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("OCR confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class BarcodePayload:
    """A locally decoded barcode; its text is sensitive and remains in memory."""

    format: str
    text: str

    def __post_init__(self) -> None:
        if not self.format or not self.text:
            raise ValueError("barcode format and text must not be empty")


@dataclass(frozen=True, slots=True)
class ExtractedField:
    """A normalised value plus its origin and validation state."""

    value: str | None
    confidence: float
    source: FieldSource
    validated: bool

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("field confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DocumentQuality:
    """Non-biometric image-quality observations for review routing."""

    boundary_found: bool
    blur_score: float | None
    glare_ratio: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """A flattened document image and the quality signals used to create it."""

    image_bgr: np.ndarray
    quality: DocumentQuality


@dataclass(frozen=True, slots=True)
class IdExtractionResult:
    """In-memory ID extraction output; it does not imply document authenticity."""

    document_type: DocumentType
    fields: Mapping[str, ExtractedField]
    quality: DocumentQuality
    portrait_crop_bgr: np.ndarray | None
    ocr_blocks: tuple[OcrTextBlock, ...]
    barcodes: tuple[BarcodePayload, ...]
    warnings: tuple[str, ...]
    requires_manual_review: bool
    normalized_document_bgr: np.ndarray | None = None


class OcrEngine(Protocol):
    def read(self, image_bgr: np.ndarray) -> Sequence[OcrTextBlock]: ...


class BarcodeReader(Protocol):
    def read(self, image_bgr: np.ndarray) -> Sequence[BarcodePayload]: ...


class BarcodeFieldParser(Protocol):
    def parse(self, barcodes: Sequence[BarcodePayload]) -> Mapping[str, ExtractedField]: ...


class DocumentTemplate(Protocol):
    document_type: DocumentType

    def matches(self, blocks: Sequence[OcrTextBlock]) -> bool: ...

    def extract(self, blocks: Sequence[OcrTextBlock]) -> tuple[Mapping[str, ExtractedField], tuple[str, ...]]: ...


class DocumentNormalizer:
    """Detect and flatten a four-corner document using OpenCV.

    Failure to find a trustworthy boundary is not treated as a successful
    correction: the source image is retained in memory with a review warning.
    """

    def __init__(self, *, minimum_edge_px: int = 480, maximum_glare_ratio: float = .20) -> None:
        if minimum_edge_px < 1 or not 0 <= maximum_glare_ratio <= 1:
            raise ValueError("invalid document-normalization thresholds")
        self.minimum_edge_px = minimum_edge_px
        self.maximum_glare_ratio = maximum_glare_ratio

    def normalize(self, image_bgr: np.ndarray) -> NormalizedDocument:
        _validate_bgr(image_bgr)
        cv2 = _opencv()
        grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        contour = self._largest_quadrilateral(grayscale, cv2)
        boundary_found = contour is not None
        normalized = _warp_document(image_bgr, contour, cv2) if contour is not None else image_bgr.copy()
        blur_variance = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        blur_score = min(1.0, blur_variance / 150.0)
        glare_ratio = float(np.mean(np.all(image_bgr >= 245, axis=2)))
        warnings: list[str] = []
        if not boundary_found:
            warnings.append("document boundary was not found; perspective correction was skipped")
        if min(normalized.shape[:2]) < self.minimum_edge_px:
            warnings.append("document resolution is below the configured minimum edge length")
        if blur_score < .25:
            warnings.append("document image appears blurred")
        if glare_ratio > self.maximum_glare_ratio:
            warnings.append("document image contains substantial glare")
        return NormalizedDocument(
            normalized,
            DocumentQuality(boundary_found, round(blur_score, 4), round(glare_ratio, 4), tuple(warnings)),
        )

    @staticmethod
    def _largest_quadrilateral(grayscale: np.ndarray, cv2: object) -> np.ndarray | None:
        edges = cv2.Canny(cv2.GaussianBlur(grayscale, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, .02 * perimeter, True)
            if len(polygon) == 4 and cv2.contourArea(polygon) > grayscale.shape[0] * grayscale.shape[1] * .10:
                return polygon.reshape(4, 2).astype(np.float32)
        return None


class PassportTd3Template:
    """Extract and checksum-validate the two 44-character ICAO TD3 MRZ lines."""

    document_type = DocumentType.PASSPORT_TD3

    def matches(self, blocks: Sequence[OcrTextBlock]) -> bool:
        lines = _mrz_lines(blocks)
        return len(lines) >= 2 and lines[-2].startswith("P<")

    def extract(self, blocks: Sequence[OcrTextBlock]) -> tuple[Mapping[str, ExtractedField], tuple[str, ...]]:
        lines = _mrz_lines(blocks)
        if len(lines) < 2:
            return {}, ("passport TD3 MRZ was not found",)
        first, second = lines[-2:]
        if len(first) != 44 or len(second) != 44 or not first.startswith("P<"):
            return {}, ("passport MRZ has an unexpected TD3 layout",)
        confidence = min(block.confidence for block in blocks if _normalise_mrz(block.text) in {first, second})
        names = first[5:44].split("<<", 1)
        surname = names[0].replace("<", " ").strip()
        given_names = names[1].replace("<", " ").strip() if len(names) == 2 else ""
        validations = {
            "document_number": _mrz_check(second[0:9], second[9]),
            "date_of_birth": _mrz_check(second[13:19], second[19]),
            "expiry_date": _mrz_check(second[21:27], second[27]),
            "personal_number": _mrz_check(second[28:42], second[42]),
            "composite": _mrz_check(second[0:10] + second[13:20] + second[21:43], second[43]),
        }
        fields = {
            "surname": ExtractedField(surname or None, confidence, FieldSource.MRZ, bool(surname)),
            "given_names": ExtractedField(given_names or None, confidence, FieldSource.MRZ, bool(given_names)),
            "document_number": ExtractedField(second[0:9].replace("<", "") or None, confidence, FieldSource.MRZ, validations["document_number"]),
            "nationality": ExtractedField(second[10:13].replace("<", "") or None, confidence, FieldSource.MRZ, True),
            "date_of_birth_yymmdd": ExtractedField(second[13:19], confidence, FieldSource.MRZ, validations["date_of_birth"]),
            "sex": ExtractedField(second[20].replace("<", "") or None, confidence, FieldSource.MRZ, second[20] in {"M", "F", "<"}),
            "expiry_date_yymmdd": ExtractedField(second[21:27], confidence, FieldSource.MRZ, validations["expiry_date"]),
        }
        warnings = () if all(validations.values()) else ("passport MRZ checksum validation failed",)
        return fields, warnings


class LabelledCardTemplate:
    """Configurable card extractor for known, versioned document layouts.

    It prioritises ``LABEL: value`` text blocks, then uses the next OCR block as
    a fallback. Production templates should combine this generic extraction with
    document-specific markers and validators.
    """

    document_type = DocumentType.CARD

    def __init__(
        self,
        labels: Mapping[str, Sequence[str]],
        *,
        markers: Sequence[str] = (),
        required_fields: Sequence[str] = (),
        validators: Mapping[str, Callable[[str], bool]] | None = None,
    ) -> None:
        if not labels:
            raise ValueError("a card template requires at least one field label")
        self.labels = {field: tuple(label.upper() for label in values) for field, values in labels.items()}
        self.markers = tuple(marker.upper() for marker in markers)
        self.required_fields = frozenset(required_fields)
        self.validators = dict(validators or {})
        if not self.required_fields <= self.labels.keys():
            raise ValueError("required card fields must be declared in labels")

    def matches(self, blocks: Sequence[OcrTextBlock]) -> bool:
        content = "\n".join(block.text.upper() for block in blocks)
        if self.markers:
            return all(marker in content for marker in self.markers)
        hits = sum(any(label in content for label in aliases) for aliases in self.labels.values())
        return hits >= min(2, len(self.labels))

    def extract(self, blocks: Sequence[OcrTextBlock]) -> tuple[Mapping[str, ExtractedField], tuple[str, ...]]:
        fields: dict[str, ExtractedField] = {}
        warnings: list[str] = []
        for name, aliases in self.labels.items():
            match = _label_value(blocks, aliases)
            if match is None:
                if name in self.required_fields:
                    warnings.append(f"required card field is missing: {name}")
                continue
            value, confidence = match
            validator = self.validators.get(name)
            fields[name] = ExtractedField(value, confidence, FieldSource.OCR, validator(value) if validator else True)
            if validator and not fields[name].validated:
                warnings.append(f"card field failed validation: {name}")
        return fields, tuple(warnings)


class NigeriaNinSlipTemplate(LabelledCardTemplate):
    """Conservative OCR template for labelled Nigerian NIN-slip fields.

    This only checks an extracted NIN's 11-digit shape. It cannot authenticate a
    NIN, QR code, holder, or document. Use an authorised NIMC verification
    service for that separate decision.
    """

    document_type = DocumentType.NIGERIA_NIN_SLIP

    def __init__(self) -> None:
        super().__init__(
            {
                "nin": ("NATIONAL IDENTIFICATION NUMBER", "NIN"),
                "surname": ("SURNAME", "LAST NAME"),
                "given_names": ("GIVEN NAMES", "FIRST NAME", "OTHER NAMES"),
                "date_of_birth": ("DATE OF BIRTH", "DOB"),
                "gender": ("GENDER", "SEX"),
                "address": ("ADDRESS",),
            },
            markers=("NIN",),
            required_fields=("nin",),
            validators={"nin": lambda value: _valid_nigeria_nin(value), "date_of_birth": lambda value: _valid_date(value)},
        )

    def matches(self, blocks: Sequence[OcrTextBlock]) -> bool:
        content = "\n".join(block.text.upper() for block in blocks)
        return "NIN" in content and ("NATIONAL" in content or bool(_nin_candidates(blocks)))

    def extract(self, blocks: Sequence[OcrTextBlock]) -> tuple[Mapping[str, ExtractedField], tuple[str, ...]]:
        fields, inherited_warnings = super().extract(blocks)
        values = dict(fields)
        warnings = list(inherited_warnings)
        candidates = _nin_candidates(blocks)
        existing = values.get("nin")
        if existing is None or not _valid_nigeria_nin(existing.value or ""):
            if len(candidates) == 1:
                value, confidence = candidates[0]
                values["nin"] = ExtractedField(value, confidence, FieldSource.OCR, True)
                warnings = [warning for warning in warnings if warning != "card field failed validation: nin"]
            elif len(candidates) > 1:
                warnings.append("multiple 11-digit NIN candidates were found")
        if "nin" not in values or not values["nin"].validated:
            warnings.append("NIN must contain exactly 11 digits; document requires manual review")
        return values, tuple(dict.fromkeys(warnings))


class IdDocumentExtractor:
    """Combine local image normalisation, OCR, document templates, and portrait crops."""

    def __init__(
        self,
        ocr: OcrEngine,
        *,
        normalizer: DocumentNormalizer | None = None,
        detector: FaceDetector | None = None,
        templates: Sequence[DocumentTemplate] = (PassportTd3Template(), NigeriaNinSlipTemplate()),
        barcode_reader: BarcodeReader | None = None,
        barcode_field_parser: BarcodeFieldParser | None = None,
    ) -> None:
        self.ocr = ocr
        self.normalizer = normalizer or DocumentNormalizer()
        self.detector = detector
        self.templates = {template.document_type: template for template in templates}
        self.barcode_reader = barcode_reader
        self.barcode_field_parser = barcode_field_parser

    def extract(
        self,
        source: np.ndarray | str | Path,
        *,
        document_type: DocumentType = DocumentType.UNKNOWN,
        return_normalized_document: bool = False,
        pdf_page: int = 0,
    ) -> IdExtractionResult:
        image = load_reference_image_bgr(source, pdf_page=pdf_page) if isinstance(source, (str, Path)) else source
        normalized = self.normalizer.normalize(image)
        blocks = tuple(self.ocr.read(normalized.image_bgr))
        barcodes = tuple(self.barcode_reader.read(normalized.image_bgr)) if self.barcode_reader else ()
        template, detected_type = self._select_template(blocks, document_type)
        fields, template_warnings = template.extract(blocks) if template else ({}, ("document type is not supported",))
        fields, barcode_warnings = self._merge_barcode_fields(fields, barcodes)
        portrait, portrait_warnings = self._portrait(normalized.image_bgr)
        warnings = tuple(dict.fromkeys((*normalized.quality.warnings, *template_warnings, *barcode_warnings, *portrait_warnings)))
        return IdExtractionResult(
            detected_type,
            fields,
            normalized.quality,
            portrait,
            blocks,
            barcodes,
            warnings,
            bool(warnings) or any(not field.validated for field in fields.values()),
            normalized.image_bgr.copy() if return_normalized_document else None,
        )

    def _select_template(self, blocks: Sequence[OcrTextBlock], requested: DocumentType) -> tuple[DocumentTemplate | None, DocumentType]:
        if requested is not DocumentType.UNKNOWN:
            return self.templates.get(requested), requested
        for document_type, template in self.templates.items():
            if template.matches(blocks):
                return template, document_type
        return None, DocumentType.UNKNOWN

    def _merge_barcode_fields(
        self, fields: Mapping[str, ExtractedField], barcodes: Sequence[BarcodePayload],
    ) -> tuple[Mapping[str, ExtractedField], tuple[str, ...]]:
        if not barcodes or self.barcode_field_parser is None:
            return fields, ()
        merged = dict(fields)
        warnings: list[str] = []
        for name, field in self.barcode_field_parser.parse(barcodes).items():
            existing = merged.get(name)
            if existing and existing.value and field.value and existing.value != field.value:
                warnings.append(f"barcode value conflicts with extracted field: {name}")
                continue
            if existing is None:
                merged[name] = field
        return merged, tuple(warnings)

    def _portrait(self, image_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[str, ...]]:
        if self.detector is None:
            return None, ()
        detections = list(self.detector.detect(image_bgr))
        if len(detections) != 1:
            return None, (f"document portrait extraction requires exactly one face; found {len(detections)}",)
        return detections[0].crop(image_bgr, padding=.05), ()


def _mrz_lines(blocks: Sequence[OcrTextBlock]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        line = _normalise_mrz(block.text)
        if len(line) >= 40:
            lines.append(line)
    return lines


def _label_value(blocks: Sequence[OcrTextBlock], aliases: Sequence[str]) -> tuple[str, float] | None:
    upper_aliases = tuple(alias.upper() for alias in aliases)
    for index, block in enumerate(blocks):
        upper = block.text.upper().strip()
        for alias in upper_aliases:
            if upper.startswith(alias):
                value = block.text[len(alias):].lstrip(" :#-\t")
                if value:
                    return value, block.confidence
                if index + 1 < len(blocks):
                    next_block = blocks[index + 1]
                    return next_block.text, min(block.confidence, next_block.confidence)
    return None


def _nin_candidates(blocks: Sequence[OcrTextBlock]) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    for block in blocks:
        for value in re.findall(r"(?<!\d)(\d{11})(?!\d)", block.text):
            candidates.append((value, block.confidence))
    return candidates


def _valid_nigeria_nin(value: str) -> bool:
    return bool(re.fullmatch(r"\d{11}", value.replace(" ", "")))


def _valid_date(value: str) -> bool:
    for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            datetime.strptime(value.strip(), pattern)
            return True
        except ValueError:
            pass
    return False


def _normalise_mrz(value: str) -> str:
    return "".join(character for character in value.upper().replace(" ", "") if character.isalnum() or character == "<")


def _mrz_check(value: str, expected: str) -> bool:
    weights = (7, 3, 1)
    total = sum(_mrz_value(character) * weights[index % 3] for index, character in enumerate(value))
    return expected.isdigit() and total % 10 == int(expected)


def _mrz_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    return 0


def _warp_document(image_bgr: np.ndarray, points: np.ndarray, cv2: object) -> np.ndarray:
    ordered = _order_points(points)
    top = np.linalg.norm(ordered[1] - ordered[0])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])
    width, height = max(1, round(max(top, bottom))), max(1, round(max(left, right)))
    destination = np.array(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)), dtype=np.float32)
    return cv2.warpPerspective(image_bgr, cv2.getPerspectiveTransform(ordered, destination), (width, height))


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.empty((4, 2), dtype=np.float32)
    sums, differences = points.sum(axis=1), np.diff(points, axis=1).reshape(-1)
    ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
    return ordered


def _validate_bgr(image_bgr: np.ndarray) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("document image must be a BGR image with three channels")


def _opencv() -> object:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install document OCR support: pip install 'face-liveness-check[id-ocr]'") from error
    return cv2
