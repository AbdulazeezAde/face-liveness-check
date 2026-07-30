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

## Development-only browser integration example

The repository includes a small local browser frontend in
[`examples/web_demo`](examples/web_demo) for testing an integration. It is not
part of the installed library, does not add a CLI command, and is not included
in the wheel. It uploads a reference portrait, samples webcam frames to a
loopback-only FastAPI service, streams frames over WebSocket, shows the
randomized prompts, and renders the verification result. Frames and the
reference image remain in memory only for the active session. Each session has
an HMAC-signed, expiring token; the example enforces a local-origin allowlist,
frame-size limit, and frame-rate limit.

Run it only from a repository checkout:

```powershell
python -m pip install -e ".[full]"
python -m pip install -r examples/web_demo/requirements.txt
python examples/web_demo/server.py --download-models --accept-model-license
```

Open <http://127.0.0.1:8000> and use the `--pad-advisory` behavior built into
the example's active-first policy. See the demo README for the stream contract,
configuration, and security boundaries.

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

## PAD evaluation

Do not choose a passive PAD model from a single webcam session. The package
includes a score-only `PadEvaluator` for comparing candidates on the same
labelled samples. Its JSONL output contains only an opaque sample ID, consented
attack label, timestamp, face count, and scores—never frames, face crops,
embeddings, document paths, or identity data.

`pad-facenox-experimental` is an independently trained, Apache-2.0 binary PAD
candidate. It uses 128×128 RGB letterboxed inputs and is deliberately separate
from `opencv-default`: its reported benchmark must be reproduced on the target
cameras and attacks before it can become a default.

Evaluate genuine, print, screen-replay, and mask samples with a consistent
consented protocol. Select thresholds from the recorded genuine-accept and
attack-reject rates, not from one model's claimed accuracy.

Use the score-only webcam collector to create labelled records. The label must
describe what is physically presented to the camera: use `replay` only while a
portrait is displayed on a separate screen, or `print` for a printed portrait.

```bash
face-liveness-check evaluate-webcam --label genuine --output pad-scores.jsonl --download --accept-model-license
face-liveness-check evaluate-webcam --label replay --output pad-scores.jsonl --download --accept-model-license
```

## Command-line tools

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

## Optional evidence storage

Evidence retention is disabled by default. To record only failed or suspicious
sessions, the integrator must explicitly enable a policy, provide a sink, and
pass explicit consent for each session. Reference images and embeddings are
never captured by this feature.

```python
from face_liveness_check import (
    EvidencePolicy, LocalEncryptedEvidenceSink, active_first_policy,
)

sink = LocalEncryptedEvidenceSink("./encrypted-evidence", key=key_from_a_secret_manager)
verifier = LivenessVerifier(
    reference_extractor,
    evidence_builder,
    active_first_policy(),
    evidence_policy=EvidencePolicy(
        enabled=True,
        capture_on={"suspicious", "failed"},
        capture_frames=True,
        capture_face_crops=True,
        max_frames=3,
        retention_days=30,
    ),
    evidence_sink=sink,
)
run = verifier.verify(id_portrait_bgr, live_frames(), evidence_consent=True)
```

Install `face-liveness-check[evidence-local]` for encrypted local files. The
local sink writes encrypted event metadata and NPY image artifacts; keep its
Fernet key in a separate secret manager and delete evidence at the recorded
retention deadline. For AWS, install `face-liveness-check[evidence-s3]` and use
`S3EvidenceSink(bucket, kms_key_id="...")`; it writes objects with SSE-KMS.
Bucket lifecycle rules, IAM access, KMS permissions, consent, and regional
biometric-data obligations remain the integrator's responsibility.

The webcam CLI exposes the same local flow without ever placing the key in shell
history. Generate and store a Fernet key in your secret manager, then expose it
to the process as an environment variable. `--pad-advisory` makes low PAD scores
record a suspicious session rather than automatically reject it.

```powershell
$env:FACE_LIVENESS_EVIDENCE_KEY = "<Fernet key from your secret manager>"
face-liveness-check webcam id-portrait.png --pad-advisory `
  --evidence-local-dir .\encrypted-evidence --evidence-consent `
  --evidence-capture-face-crops --evidence-retention-days 30
```

## Release and TestPyPI

The verified TestPyPI candidate is `0.1.0rc1`; the next production release is
`0.1.0`. CI tests Python 3.10 through 3.12, builds both distributions, and
checks their metadata. A tag such as `v0.1.0rc1` triggers TestPyPI publishing;
a published GitHub Release for `v0.1.0` triggers the production PyPI workflow.

Before publishing, create Trusted Publishers in TestPyPI and PyPI with these
exact values:

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| Owner | `AbdulazeezAde` | `AbdulazeezAde` |
| Repository | `face-liveness-check` | `face-liveness-check` |
| Workflow | `release-testpypi.yml` | `release.yml` |
| Environment | `testpypi` | `pypi` |

Protect both GitHub environments with required reviewers. After the TestPyPI
workflow succeeds, install the candidate in a clean environment with:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ face-liveness-check==0.1.0rc1
```

## Security notes

Do not use a fixed challenge order. Do not accept a single frame. Treat daylight as a quality signal—not a requirement—because indoor, low-light, and accessibility scenarios are valid. Reject or manually review multiple faces, identity switching, replay-like duplicate frames, poor quality, and missing PAD evidence.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```
