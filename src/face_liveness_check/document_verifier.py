"""Safe orchestration between local ID extraction and active liveness checks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .id_document import DocumentType, IdDocumentExtractor, IdExtractionResult
from .verifier import LiveVerification, LivenessVerifier, VerificationRun


@dataclass(frozen=True, slots=True)
class DocumentVerificationRun:
    """One document-to-live verification outcome with an explicit review path."""

    extraction: IdExtractionResult
    verification: VerificationRun | None
    reasons: tuple[str, ...]
    requires_manual_review: bool

    @property
    def matched(self) -> bool:
        return bool(self.verification and self.verification.result.matched and not self.requires_manual_review)

    @property
    def automatic_decision_allowed(self) -> bool:
        return bool(self.verification and self.verification.automatic_decision_allowed and not self.requires_manual_review)


@dataclass(slots=True)
class DocumentLiveVerification:
    """A document-gated live session; prompts appear only after document checks."""

    extraction: IdExtractionResult
    live: LiveVerification | None
    reasons: tuple[str, ...]

    @property
    def challenges(self) -> tuple[str, ...]:
        return self.live.challenges if self.live is not None else ()

    @property
    def requires_manual_review(self) -> bool:
        return self.live is None

    def observe(self, frame_bgr: np.ndarray, timestamp_s: float) -> None:
        if self.live is None:
            raise RuntimeError("cannot capture live frames until document review is resolved")
        self.live.observe(frame_bgr, timestamp_s)

    def finish(self) -> DocumentVerificationRun:
        if self.live is None:
            return DocumentVerificationRun(self.extraction, None, self.reasons, True)
        verification = self.live.finish()
        reasons = tuple(dict.fromkeys((*self.reasons, *verification.result.reasons, *verification.result.liveness.reasons)))
        return DocumentVerificationRun(self.extraction, verification, reasons, False)


class DocumentLivenessVerifier:
    """Require an extracted, reviewable ID portrait before liveness begins.

    This class does not authenticate the document or any government identifier.
    It only creates the reference portrait for the liveness verifier after the
    configured local extraction checks have passed.
    """

    def __init__(self, extractor: IdDocumentExtractor, verifier: LivenessVerifier) -> None:
        self.extractor = extractor
        self.verifier = verifier

    def start(
        self,
        document: np.ndarray | str | Path,
        *,
        document_type: DocumentType = DocumentType.UNKNOWN,
        pdf_page: int = 0,
        evidence_consent: bool = False,
        session_id: str | None = None,
    ) -> DocumentLiveVerification:
        extraction = self.extractor.extract(document, document_type=document_type, pdf_page=pdf_page)
        reasons = list(extraction.warnings)
        if extraction.requires_manual_review:
            reasons.insert(0, "document extraction requires manual review before live verification")
            return DocumentLiveVerification(extraction, None, tuple(dict.fromkeys(reasons)))
        if extraction.portrait_crop_bgr is None:
            reasons.insert(0, "document contains no usable portrait for live verification")
            return DocumentLiveVerification(extraction, None, tuple(dict.fromkeys(reasons)))
        live = self.verifier.start(
            extraction.portrait_crop_bgr,
            evidence_consent=evidence_consent,
            session_id=session_id,
        )
        return DocumentLiveVerification(extraction, live, tuple(reasons))

    def verify(
        self,
        document: np.ndarray | str | Path,
        frames: Iterable[tuple[float, np.ndarray]],
        *,
        document_type: DocumentType = DocumentType.UNKNOWN,
        pdf_page: int = 0,
        evidence_consent: bool = False,
        session_id: str | None = None,
    ) -> DocumentVerificationRun:
        live = self.start(
            document,
            document_type=document_type,
            pdf_page=pdf_page,
            evidence_consent=evidence_consent,
            session_id=session_id,
        )
        if live.live is not None:
            for timestamp_s, frame_bgr in frames:
                live.observe(frame_bgr, timestamp_s)
        return live.finish()
