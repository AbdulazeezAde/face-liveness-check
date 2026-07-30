"""Smoke tests for the local browser-integration demo without model downloads."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from fastapi.testclient import TestClient


def _demo_server():
    path = Path(__file__).parents[1] / "examples" / "web_demo" / "server.py"
    spec = importlib.util.spec_from_file_location("web_demo_server", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_browser_demo_serves_ui_and_memory_only_health_status():
    server = _demo_server()
    app = server.create_app(server.DemoConfig())
    client = TestClient(app)

    health = client.get("/api/health")
    page = client.get("/")
    script = client.get("/static/app.js")

    assert health.json()["frame_retention"] == "memory-only during an active session"
    assert "Face Liveness Check" in page.text
    assert "captureFrame" in script.text
