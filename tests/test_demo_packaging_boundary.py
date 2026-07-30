"""Keep the repository-only browser example out of the public package API."""

from __future__ import annotations

import tomllib
from pathlib import Path

from face_liveness_check.cli import build_parser


ROOT = Path(__file__).parents[1]


def test_browser_demo_is_not_a_public_extra_or_cli_command():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]

    assert "demo" not in extras
    assert "web" not in build_parser()._subparsers._group_actions[0].choices


def test_browser_demo_has_its_own_development_requirements():
    requirements = (ROOT / "examples" / "web_demo" / "requirements.txt").read_text(encoding="utf-8")

    assert "fastapi" in requirements
    assert "uvicorn" in requirements
