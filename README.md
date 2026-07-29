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
# Video integration
pip install 'face-liveness-check[opencv]'
```

## Quick start

Supply licensed detector, landmark, embedding, and PAD adapters. The package does not download model weights or impose a model licence. `OnnxArcFaceEmbedder` and `OnnxPassiveAntiSpoof` are ready for compatible ONNX models; detector and landmark adapters remain application-owned because output contracts vary by model.

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

## Security notes

Do not use a fixed challenge order. Do not accept a single frame. Treat daylight as a quality signal—not a requirement—because indoor, low-light, and accessibility scenarios are valid. Reject or manually review multiple faces, identity switching, replay-like duplicate frames, poor quality, and missing PAD evidence.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```
