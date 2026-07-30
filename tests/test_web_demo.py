"""Tests for the repository-only local browser integration example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _demo_server():
    path = Path(__file__).parents[1] / "examples" / "web_demo" / "server.py"
    spec = importlib.util.spec_from_file_location("web_demo_server", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeLive:
    challenges = ("blink", "turn_left")

    def __init__(self) -> None:
        self.session = SimpleNamespace(
            result=lambda: SimpleNamespace(frames_seen=1, completed_challenges=(), warnings=()),
        )

    def observe(self, _frame, _timestamp) -> None:
        return None


class _FakeVerifier:
    def start(self, _reference):
        return _FakeLive()


def _service(server, **config_values):
    service = server.DemoService(server.DemoConfig(session_secret="test-signing-secret", **config_values))
    service._load_verifier = lambda: _FakeVerifier()
    return service


def test_browser_demo_serves_ui_and_memory_only_health_status():
    server = _demo_server()
    app = server.create_app(server.DemoConfig())
    client = TestClient(app)

    health = client.get("/api/health")
    page = client.get("/")
    script = client.get("/static/app.js")

    assert health.json()["frame_retention"] == "memory-only during an active session"
    assert health.json()["stream_transport"] == "websocket"
    assert "Face Liveness Check" in page.text
    assert "new WebSocket" in script.text
    assert "captureFrame" in script.text


def test_signed_session_token_rejects_tampering_and_enforces_frame_rate():
    server = _demo_server()
    service = _service(server, max_frame_rate_hz=1_000_000_000)
    token, challenges = service.start(np.zeros((2, 2, 3), dtype=np.uint8))

    service.authorize(token)
    assert challenges == ("blink", "turn_left")
    assert service.observe(token, np.zeros((2, 2, 3), dtype=np.uint8))["frames_seen"] == 1
    with pytest.raises(server.DemoRateLimitError):
        service.observe(token, np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(server.DemoSessionError):
        service.authorize(f"{token}tampered")


def test_websocket_requires_local_origin_and_accepts_a_signed_active_session():
    server = _demo_server()
    config = server.DemoConfig(session_secret="test-signing-secret", port=8123)
    service = _service(server, port=8123)
    token, _ = service.start(np.zeros((2, 2, 3), dtype=np.uint8))
    app = server.create_app(config, service=service)
    client = TestClient(app)
    path = "/api/stream"

    with client.websocket_connect(path, headers={"origin": "http://127.0.0.1:8123"}) as websocket:
        assert websocket.receive_json() == {"type": "authenticate"}
        websocket.send_json({"type": "authenticate", "session_token": token})
        assert websocket.receive_json() == {"type": "ready", "max_frame_bytes": config.max_frame_bytes}
        websocket.send_json({"type": "cancel"})

    with pytest.raises(WebSocketDisconnect) as disconnect:
        with client.websocket_connect(path, headers={"origin": "https://example.test"}):
            pass
    assert disconnect.value.code == 1008
