# Face Liveness Check

`face-liveness-check` is a model-agnostic Python package for orchestrating face verification with active liveness challenges, passive anti-spoof scores, video-integrity checks, and embedding comparison.

> This project provides decision orchestration, not a guarantee that a person is live. Use a validated presentation-attack-detection (PAD) model, calibrate thresholds on representative data, and provide a safe fallback path for people who cannot complete a motion challenge.

## What it does

- Generates a cryptographically shuffled challenge sequence: blink, turn left, turn right, or nod.
- Verifies that activity occurs in the requested order.
- Requires stable single-face tracking, acceptable lighting/quality, passive anti-spoof scores, and non-duplicated video frames.
- Compares normalized face embeddings with cosine similarity only after liveness succeeds.
- Leaves model selection to the application: use licensed ONNX detector, landmark, embedding, and PAD models appropriate for your deployment.

## Install

```bash
pip install face-liveness-check
# Full default stack, webcam support, and PDF ID input
pip install 'face-liveness-check[full]'
```

## Quick start

Supply licensed detector, landmark, embedding, and PAD adapters, or use the
checksum-pinned default stack described below. `OnnxArcFaceEmbedder` and
`OnnxPassiveAntiSpoof` are ready for compatible ONNX models; detector and
landmark adapters remain application-owned where output contracts vary by model.

```python
from face_liveness_check import LivenessPolicy, LivenessVerifier
from face_liveness_check.pipeline import FrameEvidenceBuilder, ReferenceExtractor

reference_extractor = ReferenceExtractor(detector, embedder, aligner)
evidence_builder = FrameEvidenceBuilder(
    detector, embedder, pad_model,
    aligner=aligner,
    landmarks=landmark_model,
    activity=activity_detector,
)
verifier = LivenessVerifier(reference_extractor, evidence_builder, LivenessPolicy())

run = verifier.verify(id_portrait_bgr, live_frames())
print(run.challenges)  # display these randomized prompts to the person
print(run.result.matched, run.result.liveness.passed, run.result.similarity)
```

`live_frames()` yields `(timestamp_seconds, BGR_frame)` tuples, such as those
from `face_liveness_check.video.iter_frames(0)`. Use a short-lived session and
reject missing PAD evidence, multiple faces, subject switches, low quality, and
duplicate-frame replays.

## Model contracts

- **Detector:** returns `FaceDetection` objects. It must detect every face, not
  only the largest one.
- **Aligner:** aligns a detected face before embedding. ArcFace needs aligned
  faces; use an adapter matching your detector's landmark convention.
- **Landmark estimator:** returns normalized Face Mesh-compatible landmarks, or
  configure `LandmarkIndices` for another landmark model.
- **PAD model:** returns a bona-fide probability or logits. Set its live-class
  index explicitly and calibrate the threshold before deployment.

## Default model pack

Weights are not bundled into the PyPI wheel. The supported `opencv-default` pack
downloads four separately licensed models once to the operating-system cache
(normally `~/.cache/face-liveness-check/models/`), verifies every SHA-256 digest,
records the accepted licence notice, and reuses the files on later runs.

```python
from face_liveness_check import (
    LivenessVerifier, ModelPackManager, create_opencv_verifier_from_pack,
    default_registry,
)

manager = ModelPackManager(default_registry())
verifier = LivenessVerifier.from_model_pack(
    "opencv-default",
    manager=manager,
    factory=create_opencv_verifier_from_pack,
    download=True,
    accept_model_license=True,
)
```

Use `download=False` on a later startup to require a previously verified cache
without making a network request. The pack contains YuNet for detection, SFace
for alignment and identity embeddings, MiniFASNetV2 for passive live/print/replay
PAD evidence, and MediaPipe Face Landmarker for dense blink/nod landmarks.

`research-default` remains a smaller MiniFASNetV2-only pack for applications
providing their own detection, identity, and landmark models.

Dense landmarks are intentionally required for the active default policy; blink
and nod are not silently replaced with weak image-motion heuristics.

## Command line and webcam demo

Install the model pack after reviewing its licence notice:

```bash
face-liveness-check models list
face-liveness-check models install opencv-default --accept-model-license
```

Extract the single portrait from an image or PDF identity document, then run a
short interactive camera session. The CLI prints the randomized challenge order
before capture and emits a JSON result. Press `q` to end capture early.

```bash
face-liveness-check extract id.pdf id-portrait.png
face-liveness-check webcam id-portrait.png --duration 15
```

For a first-time one-command flow, pass `--download --accept-model-license` to
`extract` or `webcam`. PDF reading is local: the tool rasterizes only the selected
page and does not upload or retain ID images.

```bash
face-liveness-check webcam id.pdf --download --accept-model-license
```

## Security notes

Do not use a fixed challenge order. Do not accept a single frame. Treat daylight as a quality signal—not a requirement—because indoor, low-light, and accessibility scenarios are valid. Reject or manually review multiple faces, identity switching, replay-like duplicate frames, poor quality, and missing PAD evidence.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```
