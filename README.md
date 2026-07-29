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

## Security notes

Do not use a fixed challenge order. Do not accept a single frame. Treat daylight as a quality signal—not a requirement—because indoor, low-light, and accessibility scenarios are valid. Reject or manually review multiple faces, identity switching, replay-like duplicate frames, poor quality, and missing PAD evidence.
