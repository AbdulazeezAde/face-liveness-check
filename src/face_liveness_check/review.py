"""Opt-in, metadata-only events for manual verification review workflows."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_DISPOSITIONS = frozenset({"passed", "failed", "suspicious"})
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass(frozen=True, slots=True)
class ReviewPolicy:
    """Controls opt-in delivery of non-biometric manual-review summaries."""

    enabled: bool = False
    dispatch_on: frozenset[str] = frozenset({"failed", "suspicious"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "dispatch_on", frozenset(self.dispatch_on))
        if not self.dispatch_on or not self.dispatch_on <= _DISPOSITIONS:
            raise ValueError(f"dispatch_on must contain only: {', '.join(sorted(_DISPOSITIONS))}")


@dataclass(frozen=True, slots=True)
class ReviewEvent:
    """A summary safe to route to a review queue; it contains no biometric payloads."""

    event_id: str
    session_id: str
    created_at: str
    disposition: str
    profile: str
    automatic_decision_allowed: bool
    matched: bool
    liveness_passed: bool
    similarity: float | None
    liveness_reasons: tuple[str, ...]
    liveness_warnings: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        session_id: str,
        profile: str,
        automatic_decision_allowed: bool,
        matched: bool,
        liveness_passed: bool,
        similarity: float | None,
        liveness_reasons: Sequence[str],
        liveness_warnings: Sequence[str],
    ) -> "ReviewEvent":
        _validate_opaque_id(event_id)
        _validate_opaque_id(session_id)
        disposition = "failed" if not matched or not liveness_passed else "suspicious" if liveness_warnings else "passed"
        return cls(
            event_id=event_id,
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            disposition=disposition,
            profile=profile,
            automatic_decision_allowed=automatic_decision_allowed,
            matched=matched,
            liveness_passed=liveness_passed,
            similarity=similarity,
            liveness_reasons=tuple(liveness_reasons),
            liveness_warnings=tuple(liveness_warnings),
        )

    def payload(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")


class ReviewSink(Protocol):
    """An application-owned review destination; invoked only when explicitly enabled."""

    def publish(self, event: ReviewEvent) -> None: ...


class WebhookReviewSink:
    """POST signed review summaries to an HTTPS review endpoint.

    The request body is only ``ReviewEvent.payload()``. It never includes a
    reference image, video frame, face crop, embedding, or document path.
    Delivery failures raise an error so applications can surface a missed review
    notification rather than silently losing it.
    """

    def __init__(
        self,
        url: str,
        *,
        signing_key: bytes | str,
        timeout_s: float = 5.0,
        allow_insecure_http: bool = False,
        opener: Callable[..., object] | None = None,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ({"https", "http"} if allow_insecure_http else {"https"}) or not parsed.netloc:
            raise ValueError("review webhook URL must be HTTPS with a host")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        if not key:
            raise ValueError("signing_key must not be empty")
        self.url = url
        self.timeout_s = timeout_s
        self._signing_key = key
        self._opener = opener or urlopen

    def publish(self, event: ReviewEvent) -> None:
        body = event.payload()
        signature = hmac.new(self._signing_key, body, hashlib.sha256).hexdigest()
        request = Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "face-liveness-check-review-webhook/1",
                "X-Face-Liveness-Event-Id": event.event_id,
                "X-Face-Liveness-Signature": f"sha256={signature}",
            },
        )
        with self._opener(request, timeout=self.timeout_s) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if status < 200 or status >= 300:
                raise RuntimeError(f"review webhook returned HTTP {status}")


def _validate_opaque_id(value: str) -> None:
    if not _OPAQUE_ID.fullmatch(value):
        raise ValueError("review identifiers must be opaque alphanumeric, hyphen, or underscore values")
