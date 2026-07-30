"""Local browser demo for face-liveness-check.

This is an integration test UI, not a production authentication service. It
keeps reference images and webcam frames in memory only for the short active
session, and does not enable evidence retention.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np

from face_liveness_check import (
    LivenessVerifier,
    ModelPackManager,
    active_first_policy,
    create_opencv_verifier_from_pack,
    default_registry,
)
from face_liveness_check.verifier import LiveVerification, VerificationRun


_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_SESSION_TTL_S = 120.0
_STATIC_DIRECTORY = Path(__file__).parent / "static"


@dataclass(frozen=True, slots=True)
class DemoConfig:
    cache_dir: Path | None = None
    model_pack: str = "opencv-default"
    download_models: bool = False
    accept_model_license: bool = False


@dataclass(slots=True)
class _DemoSession:
    live: LiveVerification
    created_at: float
    lock: Lock = field(default_factory=Lock)


class DemoService:
    """Owns short-lived sessions and lazily loads the verified model pack."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self._verifier: LivenessVerifier | None = None
        self._sessions: dict[str, _DemoSession] = {}
        self._lock = Lock()
        self._model_lock = Lock()

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model_pack": self.config.model_pack,
            "models_loaded": self._verifier is not None,
            "frame_retention": "memory-only during an active session",
        }

    def start(self, reference_bgr: np.ndarray) -> tuple[str, tuple[str, ...]]:
        with self._model_lock:
            live = self._load_verifier().start(reference_bgr)
        session_id = uuid4().hex
        with self._lock:
            self._remove_expired_locked()
            self._sessions[session_id] = _DemoSession(live, time.monotonic())
        return session_id, live.challenges

    def observe(self, session_id: str, frame_bgr: np.ndarray) -> dict[str, object]:
        session = self._session(session_id)
        with session.lock, self._model_lock:
            session.live.observe(frame_bgr, time.monotonic() - session.created_at)
            liveness = session.live.session.result()
        return {
            "frames_seen": liveness.frames_seen,
            "completed_challenges": [challenge.value for challenge in liveness.completed_challenges],
            "warnings": list(liveness.warnings),
        }

    def finish(self, session_id: str) -> dict[str, object]:
        with self._lock:
            try:
                session = self._sessions.pop(session_id)
            except KeyError as error:
                raise KeyError("session was not found or has expired") from error
        with session.lock, self._model_lock:
            run = session.live.finish()
        return _run_json(run)

    def _session(self, session_id: str) -> _DemoSession:
        with self._lock:
            self._remove_expired_locked()
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise KeyError("session was not found or has expired") from error

    def _remove_expired_locked(self) -> None:
        now = time.monotonic()
        for session_id, session in list(self._sessions.items()):
            if now - session.created_at > _SESSION_TTL_S:
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


def create_app(config: DemoConfig | None = None) -> Any:
    """Create the local FastAPI app without importing demo dependencies at install time."""
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.responses import FileResponse
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise ImportError("Install the demo dependencies: pip install 'face-liveness-check[full,demo]'") from error

    service = DemoService(config or DemoConfig())
    app = FastAPI(title="Face Liveness Check Demo", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIRECTORY / "index.html")

    @app.get("/static/{asset_name}")
    def static_asset(asset_name: str) -> FileResponse:
        destination = (_STATIC_DIRECTORY / asset_name).resolve()
        if destination.parent != _STATIC_DIRECTORY.resolve() or not destination.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(destination)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return service.health()

    @app.post("/api/sessions")
    async def create_session(reference: UploadFile = File(...)) -> dict[str, object]:
        try:
            image = await _read_bgr(reference)
            session_id, challenges = service.start(image)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"unable to start verification: {error}") from error
        return {"session_id": session_id, "challenges": list(challenges)}

    @app.post("/api/sessions/{session_id}/frames")
    async def add_frame(session_id: str, frame: UploadFile = File(...)) -> dict[str, object]:
        try:
            return service.observe(session_id, await _read_bgr(frame))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/sessions/{session_id}/finish")
    def finish_session(session_id: str) -> dict[str, object]:
        try:
            return service.finish(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app


async def _read_bgr(upload: Any) -> np.ndarray:
    payload = await upload.read(_MAX_IMAGE_BYTES + 1)
    if not payload or len(payload) > _MAX_IMAGE_BYTES:
        raise ValueError("image must be between 1 byte and 5 MB")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise ValueError("OpenCV is required; install face-liveness-check[full,demo]") from error
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("upload must be a valid image")
    return image


def _run_json(run: VerificationRun) -> dict[str, object]:
    result = run.result
    return {
        "challenges": list(run.challenges),
        "matched": result.matched,
        "similarity": result.similarity,
        "reasons": list(result.reasons),
        "liveness": {
            "passed": result.liveness.passed,
            "confidence": result.liveness.confidence,
            "completed_challenges": [challenge.value for challenge in result.liveness.completed_challenges],
            "warnings": list(result.liveness.warnings),
            "reasons": list(result.liveness.reasons),
            "frames_seen": result.liveness.frames_seen,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local face-liveness-check browser demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--accept-model-license", action="store_true")
    arguments = parser.parse_args()
    if arguments.download_models and not arguments.accept_model_license:
        parser.error("--download-models requires --accept-model-license")
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise SystemExit("Install the demo dependencies: pip install 'face-liveness-check[full,demo]'") from error
    uvicorn.run(
        create_app(DemoConfig(arguments.cache_dir, download_models=arguments.download_models,
                              accept_model_license=arguments.accept_model_license)),
        host=arguments.host,
        port=arguments.port,
    )


if __name__ == "__main__":
    main()
