# Release checklist

Use this checklist before creating a PyPI release.

1. Confirm the release branch is reviewed, CI is green, and no secrets or model
   weights are included in the source distribution or wheel.
2. Run `ruff check src tests`, `mypy`, `pytest --cov`, `python -m build`, and
   `twine check dist/*` locally when practical. CI repeats these checks and also
   builds the Docker demo image.
3. Update the version, release notes, supported Python versions, model-pack
   checksums/licenses, and migration/security notes.
4. Verify the package boundary: the browser demo remains under `examples/` and
   is not exposed as a package CLI command or bundled into the wheel.
5. Test install in a clean environment, including the intended optional extras.
   Use TestPyPI first for pre-releases.
6. Review PAD threshold changes with the protocol in `PAD_EVALUATION_PROTOCOL.md`;
   do not release a claimed accuracy result without representative, consented
   evaluation data and a documented holdout result.
7. Confirm evidence-operation documentation, lifecycle configuration, trusted
   publisher configuration, and incident contacts are current.
8. Create the signed/tagged GitHub release only after all required checks pass;
   then verify the PyPI artifact metadata and clean-install it from PyPI.
