# Browser integration demo

This local FastAPI example proves the browser-to-package integration: the browser
uploads one consented identity-document image, extracts a portrait and reviewed
fields locally, then sends sampled webcam frames to the local Python process.
If the document needs review or lacks a usable portrait, the demo never opens a
camera session or issues a WebSocket token. It uses the active-first policy, so
PAD warnings are shown but do not by themselves reject a session. Evidence
retention is disabled. It is development-only: it is not an installed CLI
command, and it is not included in the package wheel.

Version 2 streams binary JPEG frames through a localhost WebSocket only after
the document review passes. A successful review starts a short-lived
HMAC-signed session. The server accepts only the current local origin, limits
every frame to 1 MB, and accepts at most two frames per second by default. The
service is intentionally bound to `127.0.0.1`; it cannot be exposed with a
host flag.

## Typed FastAPI and WebSocket contracts

[`server.py`](server.py) is the reference FastAPI/WebSocket integration. Its
Pydantic request and response models are strict (`extra="forbid"`) and cover
health, session creation, authentication, commands, progress, warnings, errors,
and final verification results. Use these names when adapting the example:

- `HealthResponse`, `SessionStartResponse`, `DocumentReviewResponse`, and
  `DocumentVerificationResultResponse` for HTTP/final-result contracts.
- `StreamAuthenticateRequest` and `StreamCommandRequest` for browser messages.
- `StreamReadyMessage`, `StreamProgressMessage`, `StreamWarningMessage`,
  `StreamErrorMessage`, and `StreamResultMessage` for server messages.

The HTTP endpoints declare their `response_model`, while WebSocket messages are
validated before parsing or sending. This makes the example a concrete contract
for an integrator without adding FastAPI or Pydantic to the published package.

Install from the repository checkout:

```powershell
python -m pip install -e ".[onnx,mediapipe,documents,id-ocr]"
python -m pip install -r examples/web_demo/requirements.txt
python examples/web_demo/server.py --download-models --accept-model-license
```

Open <http://127.0.0.1:8000>. Models are downloaded once to the normal package
cache; on later runs omit `--download-models` to require the verified cache.

## Integration contract

`POST /api/sessions` accepts one multipart `document` image plus an optional
`document_type` (`unknown`, `passport_td3`, or `nigeria_nin_slip`). The image
is handled in memory only; PDF uploads are intentionally not part of this demo.
It returns a safe document review summary, never the source image, OCR regions,
barcode payloads, normalized document, or portrait crop.

```json
{
  "session_token": "signed-expiring-bearer-token-or-null",
  "challenges": ["blink", "turn_left"],
  "stream_path": "/api/stream-or-null",
  "expires_in_seconds": 120,
  "document": {
    "document_type": "passport_td3",
    "fields": {},
    "quality_warnings": [],
    "warnings": [],
    "portrait_available": true,
    "requires_manual_review": false
  }
}
```

When `requires_manual_review` is true, `session_token`, `stream_path`, and
`expires_in_seconds` are null and `challenges` is empty. Resolve the document
review first: the browser does not ask for webcam permission in that state.
Document extraction is local OCR and quality assessment, not proof that an ID
or government identifier is authentic.

Connect to the returned path with `ws://127.0.0.1:8000`. First send
`{"type":"authenticate","session_token":"..."}`; the token is kept out of
the URL so ordinary server access logs do not record it. Then send binary JPEG
frame bytes and read `progress`, `warning`, `error`, and final `result` JSON
messages. Send `{"type":"finish"}` to receive the result and close the session, or
`{"type":"cancel"}` to discard it. The token is a short-lived bearer token:
do not log, store, or send it to a third party.

## Local configuration and diagnostics

Use `--help` to discover every local setting:

```powershell
python examples/web_demo/server.py --help
python examples/web_demo/server.py --port 8010 --session-ttl 90 --max-frame-rate 1.5
```

The signing key is generated freshly for each process by default, which makes
all sessions invalid after a restart. For a controlled local restart, put a
secret in your environment—not on the command line—and select its variable:

```powershell
$env:FACE_LIVENESS_DEMO_SESSION_SECRET = "development-only-secret"
python examples/web_demo/server.py --session-secret-env FACE_LIVENESS_DEMO_SESSION_SECRET
```

`GET /api/health` reports the active limits, local allowed origins, model-pack
state, and the memory-only retention guarantee. It deliberately never reports
the signing secret or session tokens.

## Docker demo

The included Docker configuration still publishes only to the local machine.
From the repository root, build and run it with the model licence acceptance
that you reviewed:

```powershell
docker build -f examples/web_demo/Dockerfile -t face-liveness-check-demo .
docker run --rm -p 127.0.0.1:8000:8000 `
  -v face-liveness-model-cache:/root/.cache/face-liveness-check/models `
  face-liveness-check-demo --download-models --accept-model-license
```

Then open <http://127.0.0.1:8000>. The named volume reuses checksum-verified
models on later runs, so omit `--download-models` after the first one. A compose
configuration is also available:

```powershell
docker compose -f examples/web_demo/docker-compose.yml run --rm --service-ports `
  web-demo --download-models --accept-model-license
```

The container listens on `0.0.0.0` only inside Docker because that is required
for port forwarding; `docker-compose.yml` and the documented command bind it to
`127.0.0.1` on the host. Do not change that mapping to a public interface.

## Replay-test fixtures

The WebSocket integration test replays a generated prerecorded frame from
[`tests/fixtures/web_demo`](../../tests/fixtures/web_demo). It is deliberately
non-biometric. The fixture README defines the provenance and consent gate for a
separate, restricted real-person fixture pack; do not add real face recordings
to this public repository without explicit redistribution consent.

Do not treat this example as an internet-facing authentication service. A real
deployment still needs application authentication and authorization, TLS,
audited rate limiting, consent handling, a reviewed evidence-retention policy,
and PAD calibration on representative attacks.
