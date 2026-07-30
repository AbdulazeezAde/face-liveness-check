"""Keep the repository-only browser example out of the public package API."""

from __future__ import annotations

from pathlib import Path

from face_liveness_check.cli import build_parser


ROOT = Path(__file__).parents[1]


def test_browser_demo_is_not_a_public_extra_or_cli_command():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extras = pyproject.split("[project.optional-dependencies]", maxsplit=1)[1].split("\n[", maxsplit=1)[0]

    assert not any(line.startswith("demo =") for line in extras.splitlines())
    assert "web" not in build_parser()._subparsers._group_actions[0].choices


def test_browser_demo_has_its_own_development_requirements():
    requirements = (ROOT / "examples" / "web_demo" / "requirements.txt").read_text(encoding="utf-8")

    assert "fastapi" in requirements
    assert "uvicorn" in requirements
