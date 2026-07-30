# Browser integration demo

This local FastAPI example proves the browser-to-package integration: the browser
uploads one reference portrait, sends sampled webcam frames to the local Python
process, and displays the active liveness and identity result. It uses the
active-first policy, so PAD warnings are shown but do not by themselves reject a
session. Evidence retention is disabled. It is development-only: it is not an
installed CLI command, and it is not included in the package wheel.

Version 2 streams binary JPEG frames through a localhost WebSocket. A reference
upload starts a short-lived HMAC-signed session. The server accepts only the
current local origin, limits every frame to 1 MB, and accepts at most two frames
per second by default. The service is intentionally bound to `127.0.0.1`; it
cannot be exposed with a host flag.

Install from the repository checkout:

```powershell
python -m pip install -e ".[full]"
python -m pip install -r examples/web_demo/requirements.txt
python examples/web_demo/server.py --download-models --accept-model-license
```

Open <http://127.0.0.1:8000>. Models are downloaded once to the normal package
cache; on later runs omit `--download-models` to require the verified cache.

## Integration contract

`POST /api/sessions` accepts one multipart `reference` image and returns:

```json
{
  "session_token": "signed-expiring-bearer-token",
  "challenges": ["blink", "turn_left"],
  "stream_path": "/api/stream",
  "expires_in_seconds": 120
}
```

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

Do not treat this example as an internet-facing authentication service. A real
deployment still needs application authentication and authorization, TLS,
audited rate limiting, consent handling, a reviewed evidence-retention policy,
and PAD calibration on representative attacks.
