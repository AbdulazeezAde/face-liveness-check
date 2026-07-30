"""Local WebSocket browser integration example for face-liveness-check.

This is a development integration UI, not a production authentication service.
It keeps reference images and webcam frames in memory only for a short active
session, and does not enable evidence retention.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from face_liveness_check import (
    LivenessVerifier,
    ModelPackManager,
    active_first_policy,
    create_opencv_verifier_from_pack,
    default_registry,
)
from face_liveness_check.verifier import LiveVerification, VerificationRun


_MAX_REFERENCE_BYTES = 5 * 1024 * 1024
_DEFAULT_MAX_FRAME_BYTES = 1 * 1024 * 1024
_SESSION_TTL_S = 120.0
_DEFAULT_MAX_FRAME_RATE_HZ = 2.0
_STATIC_DIRECTORY = Path(__file__).parent / "static"


class _MessageModel(BaseModel):
    """Strict, serializable contract shared by HTTP and WebSocket messages."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(_MessageModel):
    status: Literal["ok"]
    model_pack: str
    models_loaded: bool
    frame_retention: Literal["memory-only during an active session"]
    stream_transport: Literal["websocket"]
    session_ttl_seconds: float = Field(gt=0)
    max_frame_bytes: int = Field(gt=0)
    max_frame_rate_hz: float = Field(gt=0)
    allowed_origins: list[str]


class SessionStartResponse(_MessageModel):
    session_token: str = Field(min_length=1)
    challenges: list[str]
    stream_path: Literal["/api/stream"]
    expires_in_seconds: float = Field(gt=0)


class StreamAuthenticateRequest(_MessageModel):
    type: Literal["authenticate"]
    session_token: str = Field(min_length=1)


class StreamCommandRequest(_MessageModel):
    type: Literal["finish", "cancel"]


class StreamAuthenticatePrompt(_MessageModel):
    type: Literal["authenticate"] = "authenticate"


class StreamReadyMessage(_MessageModel):
    type: Literal["ready"] = "ready"
    max_frame_bytes: int = Field(gt=0)


class StreamProgressMessage(_MessageModel):
    type: Literal["progress"] = "progress"
    frames_seen: int = Field(ge=0)
    completed_challenges: list[str]
    warnings: list[str]


class StreamWarningMessage(_MessageModel):
    type: Literal["warning"] = "warning"
    code: Literal["frame_rate_limited"]
    detail: str


class StreamErrorMessage(_MessageModel):
    type: Literal["error"] = "error"
    detail: str


class LivenessResponse(_MessageModel):
    passed: bool
    confidence: float
    completed_challenges: list[str]
    warnings: list[str]
    reasons: list[str]
    frames_seen: int = Field(ge=0)


class VerificationResultResponse(_MessageModel):
    challenges: list[str]
    matched: bool
    similarity: float | None
    reasons: list[str]
    liveness: LivenessResponse


class StreamResultMessage(_MessageModel):
    type: Literal["result"] = "result"
    result: VerificationResultResponse


@dataclass(frozen=True, slots=True)
class DemoConfig:
    cache_dir: Path | None = None
    model_pack: str = "opencv-default"
    download_models: bool = False
    accept_model_license: bool = False
    session_secret: str | None = None
    session_ttl_s: float = _SESSION_TTL_S
    max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES
    max_frame_rate_hz: float = _DEFAULT_MAX_FRAME_RATE_HZ
    port: int = 8000
    additional_allowed_origins: tuple[str, ...] = ()


@dataclass(slots=True)
class _DemoSession:
    live: LiveVerification
    created_at: float
    expires_at: float
    last_frame_at: float | None = None
    lock: Lock = field(default_factory=Lock)


class DemoSessionError(ValueError):
    """A session token is malformed, expired, or no longer active."""


class DemoRateLimitError(DemoSessionError):
    """The browser is sending frames faster than the configured safe rate."""


class DemoOriginError(ValueError):
    """A browser request came from an origin this local server does not trust."""


class DemoService:
    """Owns signed short-lived sessions and lazily loads the verified model pack."""

    def __init__(self, config: DemoConfig) -> None:
        if config.session_ttl_s <= 0:
            raise ValueError("session_ttl_s must be positive")
        if config.max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        if config.max_frame_rate_hz <= 0:
            raise ValueError("max_frame_rate_hz must be positive")
        self.config = config
        self._verifier: LivenessVerifier | None = None
        self._sessions: dict[str, _DemoSession] = {}
        self._lock = Lock()
        self._model_lock = Lock()
        # An unset secret becomes a fresh, process-local key. A restart invalidates every session.
        self._signing_key = (config.session_secret or secrets.token_urlsafe(32)).encode("utf-8")

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        defaults = (f"http://127.0.0.1:{self.config.port}", f"http://localhost:{self.config.port}")
        return tuple(dict.fromkeys((*defaults, *self.config.additional_allowed_origins)))

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_pack=self.config.model_pack,
            models_loaded=self._verifier is not None,
            frame_retention="memory-only during an active session",
            stream_transport="websocket",
            session_ttl_seconds=self.config.session_ttl_s,
            max_frame_bytes=self.config.max_frame_bytes,
            max_frame_rate_hz=self.config.max_frame_rate_hz,
            allowed_origins=list(self.allowed_origins),
        )

    def start(self, reference_bgr: np.ndarray) -> tuple[str, tuple[str, ...]]:
        with self._model_lock:
            live = self._load_verifier().start(reference_bgr)
        session_id = secrets.token_urlsafe(24)
        now = time.monotonic()
        expires_at = math.ceil(time.time() + self.config.session_ttl_s)
        with self._lock:
            self._remove_expired_locked()
            self._sessions[session_id] = _DemoSession(live, now, now + self.config.session_ttl_s)
        return self._sign(session_id, expires_at), live.challenges

    def observe(self, session_token: str, frame_bgr: np.ndarray) -> StreamProgressMessage:
        session = self._session(session_token)
        with session.lock, self._model_lock:
            now = time.monotonic()
            minimum_interval = 1 / self.config.max_frame_rate_hz
            if session.last_frame_at is not None and now - session.last_frame_at < minimum_interval:
                raise DemoRateLimitError("frame rate limit exceeded; wait before sending another frame")
            session.last_frame_at = now
            session.live.observe(frame_bgr, now - session.created_at)
            liveness = session.live.session.result()
        return StreamProgressMessage(
            frames_seen=liveness.frames_seen,
            completed_challenges=[challenge.value for challenge in liveness.completed_challenges],
            warnings=list(liveness.warnings),
        )

    def finish(self, session_token: str) -> VerificationResultResponse:
        session_id = self._verified_session_id(session_token)
        with self._lock:
            self._remove_expired_locked()
            try:
                session = self._sessions.pop(session_id)
            except KeyError as error:
                raise DemoSessionError("session was not found or has expired") from error
        with session.lock, self._model_lock:
            return _run_json(session.live.finish())

    def cancel(self, session_token: str) -> None:
        session_id = self._verified_session_id(session_token)
        with self._lock:
            self._sessions.pop(session_id, None)

    def authorize(self, session_token: str) -> None:
        self._session(session_token)

    def _session(self, session_token: str) -> _DemoSession:
        session_id = self._verified_session_id(session_token)
        with self._lock:
            self._remove_expired_locked()
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise DemoSessionError("session was not found or has expired") from error

    def _verified_session_id(self, session_token: str) -> str:
        try:
            session_id, expires_at_text, signature = session_token.split(".", maxsplit=2)
            expires_at = int(expires_at_text)
        except (AttributeError, TypeError, ValueError) as error:
            raise DemoSessionError("invalid session token") from error
        if not hmac.compare_digest(signature, self._signature(session_id, expires_at)):
            raise DemoSessionError("invalid session token")
        if time.time() >= expires_at:
            raise DemoSessionError("session token has expired")
        return session_id

    def _sign(self, session_id: str, expires_at: int) -> str:
        return f"{session_id}.{expires_at}.{self._signature(session_id, expires_at)}"

    def _signature(self, session_id: str, expires_at: int) -> str:
        payload = f"{session_id}.{expires_at}".encode("ascii")
        return hmac.new(self._signing_key, payload, hashlib.sha256).hexdigest()

    def _remove_expired_locked(self) -> None:
        now = time.monotonic()
        for session_id, session in list(self._sessions.items()):
            if now >= session.expires_at:
                del self._sessions[session_id]

    def _load_verifier(self) -> LivenessVerifier:
        if self._verifier is None:
            manager = ModelPackManager(default_registry(), cache_dir=self.config.cache_dir)
            installed = (
                manager.install(self.config.model_pack, accept_model_license=self.config.accept_model_license)
                if self.config.download_models else manager.resolve(self.config.model_pack)
            )
            self._verifier = create_opencv_verifier_from_pack(installed, policy=active_first_policy())
        return self._verifier


def create_app(config: DemoConfig | None = None, *, service: DemoService | None = None) -> Any:
    """Create the local FastAPI app without importing demo dependencies at install time."""
    try:
        from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise ImportError(
            "Install this checkout's demo dependencies: "
            "python -m pip install -r examples/web_demo/requirements.txt"
        ) from error

    # FastAPI resolves postponed endpoint annotations from module globals. Keep
    # these imports lazy so installing face-liveness-check never requires FastAPI.
    globals().update({"Request": Request, "UploadFile": UploadFile, "WebSocket": WebSocket})

    service = service or DemoService(config or DemoConfig())
    app = FastAPI(title="Face Liveness Check Demo", docs_url=None, redoc_url=None)
    app.state.demo_service = service

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIRECTORY / "index.html")

    @app.get("/static/{asset_name}")
    def static_asset(asset_name: str) -> FileResponse:
        destination = (_STATIC_DIRECTORY / asset_name).resolve()
        if destination.parent != _STATIC_DIRECTORY.resolve() or not destination.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(destination)

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return service.health()

    @app.post("/api/sessions", response_model=SessionStartResponse)
    async def create_session(request: Request, reference: UploadFile = File(...)) -> SessionStartResponse:
        try:
            _require_allowed_origin(request.headers.get("origin"), service.allowed_origins)
            image = await _read_bgr(reference)
            session_token, challenges = service.start(image)
        except DemoOriginError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"unable to start verification: {error}") from error
        return SessionStartResponse(
            session_token=session_token,
            challenges=list(challenges),
            stream_path="/api/stream",
            expires_in_seconds=service.config.session_ttl_s,
        )

    @app.websocket("/api/stream")
    async def stream_session(websocket: WebSocket) -> None:
        if not _is_allowed_origin(websocket.headers.get("origin"), service.allowed_origins):
            await websocket.close(code=1008, reason="origin is not allowed")
            return
        await websocket.accept()
        await _send_message(websocket, StreamAuthenticatePrompt())
        try:
            initial_message = await websocket.receive()
            if initial_message["type"] == "websocket.disconnect":
                return
            session_token = _read_session_token(initial_message.get("text"))
            if session_token is None:
                await _send_message(websocket, StreamErrorMessage(detail="authenticate with a signed session token first"))
                await websocket.close(code=1008)
                return
            try:
                service.authorize(session_token)
            except DemoSessionError:
                await websocket.close(code=1008, reason="invalid or expired session")
                return
            await _send_message(websocket, StreamReadyMessage(max_frame_bytes=service.config.max_frame_bytes))
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                payload = message.get("bytes")
                if payload is not None:
                    try:
                        progress = service.observe(session_token, _decode_bgr(payload, service.config.max_frame_bytes, kind="frame"))
                    except DemoRateLimitError as error:
                        await _send_message(websocket, StreamWarningMessage(code="frame_rate_limited", detail=str(error)))
                    except (DemoSessionError, ValueError) as error:
                        await _send_message(websocket, StreamErrorMessage(detail=str(error)))
                        await websocket.close(code=1008)
                        return
                    else:
                        await _send_message(websocket, progress)
                    continue
                command = _read_command(message.get("text"))
                if command == "finish":
                    try:
                        await _send_message(websocket, StreamResultMessage(result=service.finish(session_token)))
                    except DemoSessionError as error:
                        await _send_message(websocket, StreamErrorMessage(detail=str(error)))
                    await websocket.close(code=1000)
                    return
                if command == "cancel":
                    service.cancel(session_token)
                    await websocket.close(code=1000)
                    return
                await _send_message(websocket, StreamErrorMessage(detail="expected a binary JPEG frame, finish, or cancel"))
        except WebSocketDisconnect:
            return

    return app


async def _read_bgr(upload: Any) -> np.ndarray:
    payload = await upload.read(_MAX_REFERENCE_BYTES + 1)
    return _decode_bgr(payload, _MAX_REFERENCE_BYTES, kind="reference image")


def _decode_bgr(payload: bytes, maximum_bytes: int, *, kind: str = "image") -> np.ndarray:
    if not payload or len(payload) > maximum_bytes:
        maximum_mb = maximum_bytes // (1024 * 1024)
        raise ValueError(f"{kind} must be between 1 byte and {maximum_mb} MB")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise ValueError("OpenCV is required; from the checkout run: python -m pip install -e '.[full]'") from error
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{kind} must be a valid image")
    return image


def _read_command(payload: str | None) -> str | None:
    data = _read_json_object(payload)
    try:
        return StreamCommandRequest.model_validate(data).type
    except ValidationError:
        return None


def _read_session_token(payload: str | None) -> str | None:
    data = _read_json_object(payload)
    try:
        return StreamAuthenticateRequest.model_validate(data).session_token
    except ValidationError:
        return None


def _read_json_object(payload: str | None) -> dict[str, object] | None:
    if payload is None:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


async def _send_message(websocket: Any, message: _MessageModel) -> None:
    await websocket.send_json(message.model_dump(mode="json"))


def _is_allowed_origin(origin: str | None, allowed_origins: tuple[str, ...]) -> bool:
    """Allow non-browser test clients, but reject every supplied foreign origin."""
    return origin is None or origin in allowed_origins


def _require_allowed_origin(origin: str | None, allowed_origins: tuple[str, ...]) -> None:
    if not _is_allowed_origin(origin, allowed_origins):
        raise DemoOriginError("origin is not allowed")


def _run_json(run: VerificationRun) -> VerificationResultResponse:
    result = run.result
    return VerificationResultResponse(
        challenges=list(run.challenges),
        matched=result.matched,
        similarity=result.similarity,
        reasons=list(result.reasons),
        liveness=LivenessResponse(
            passed=result.liveness.passed,
            confidence=result.liveness.confidence,
            completed_challenges=[challenge.value for challenge in result.liveness.completed_challenges],
            warnings=list(result.liveness.warnings),
            reasons=list(result.liveness.reasons),
            frames_seen=result.liveness.frames_seen,
        ),
    )


def main(*, host: str = "127.0.0.1") -> None:
    parser = argparse.ArgumentParser(description="Run the local face-liveness-check browser integration example")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--accept-model-license", action="store_true")
    parser.add_argument("--session-ttl", type=float, default=_SESSION_TTL_S, help="signed session lifetime in seconds")
    parser.add_argument("--max-frame-rate", type=float, default=_DEFAULT_MAX_FRAME_RATE_HZ, help="maximum accepted webcam frames per second")
    parser.add_argument("--max-frame-bytes", type=int, default=_DEFAULT_MAX_FRAME_BYTES, help="maximum accepted binary frame size")
    parser.add_argument("--session-secret-env", default="FACE_LIVENESS_DEMO_SESSION_SECRET", help="optional environment variable holding a session-signing secret")
    arguments = parser.parse_args()
    if arguments.download_models and not arguments.accept_model_license:
        parser.error("--download-models requires --accept-model-license")
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise SystemExit(
            "Install this checkout's demo dependencies: "
            "python -m pip install -r examples/web_demo/requirements.txt"
        ) from error
    config = DemoConfig(
        cache_dir=arguments.cache_dir,
        download_models=arguments.download_models,
        accept_model_license=arguments.accept_model_license,
        session_secret=os.environ.get(arguments.session_secret_env),
        session_ttl_s=arguments.session_ttl,
        max_frame_rate_hz=arguments.max_frame_rate,
        max_frame_bytes=arguments.max_frame_bytes,
        port=arguments.port,
    )
    uvicorn.run(create_app(config), host=host, port=arguments.port)


if __name__ == "__main__":
    main()
