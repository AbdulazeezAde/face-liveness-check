"""Tests for the repository-only local browser integration example."""

from __future__ import annotations

import base64
import importlib.util
import json
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


def _prerecorded_frame() -> tuple[dict[str, object], bytes]:
    fixture_path = Path(__file__).parent / "fixtures" / "web_demo" / "synthetic_prerecorded_frame.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    return fixture, base64.b64decode(fixture["payload_base64"])


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


def test_typed_stream_models_reject_unexpected_request_fields():
    server = _demo_server()
    health = server.DemoService(server.DemoConfig()).health()

    assert isinstance(health, server.HealthResponse)
    assert server._read_session_token('{"type":"authenticate","session_token":"token"}') == "token"
    assert server._read_session_token('{"type":"authenticate","session_token":"token","extra":true}') is None
    assert server._read_command('{"type":"finish"}') == "finish"
    assert server._read_command('{"type":"finish","extra":true}') is None


def test_signed_session_token_rejects_tampering_and_enforces_frame_rate(monkeypatch):
    server = _demo_server()
    monotonic_times = iter((10.0, 10.0, 10.0, 10.0, 10.0, 10.1, 10.1))
    monkeypatch.setattr(server.time, "monotonic", lambda: next(monotonic_times))
    service = _service(server, max_frame_rate_hz=2)
    token, challenges = service.start(np.zeros((2, 2, 3), dtype=np.uint8))

    service.authorize(token)
    assert challenges == ("blink", "turn_left")
    assert service.observe(token, np.zeros((2, 2, 3), dtype=np.uint8)).frames_seen == 1
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


def test_websocket_replays_a_prerecorded_non_biometric_frame(monkeypatch):
    server = _demo_server()
    fixture, payload = _prerecorded_frame()
    assert fixture["subject"] == "synthetic non-biometric test pattern"
    service = _service(server, port=8124)
    token, _ = service.start(np.zeros((2, 2, 3), dtype=np.uint8))
    app = server.create_app(server.DemoConfig(session_secret="test-signing-secret", port=8124), service=service)
    client = TestClient(app)

    def decode_frame(frame_payload, *_args, **_kwargs):
        assert frame_payload == payload
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(server, "_decode_bgr", decode_frame)
    with client.websocket_connect("/api/stream", headers={"origin": "http://127.0.0.1:8124"}) as websocket:
        assert websocket.receive_json() == {"type": "authenticate"}
        websocket.send_json({"type": "authenticate", "session_token": token})
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(payload)
        assert websocket.receive_json() == {
            "type": "progress",
            "frames_seen": 1,
            "completed_challenges": [],
            "warnings": [],
        }
