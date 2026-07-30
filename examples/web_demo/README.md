# Browser integration demo

This local FastAPI example proves the browser-to-package integration: the browser
uploads one reference portrait, sends sampled webcam frames to the local Python
process, and displays the active liveness and identity result. It uses the
active-first policy, so PAD warnings are shown but do not by themselves reject a
session. Evidence retention is disabled. It is development-only: it is not an
installed CLI command, and it is not included in the package wheel.

Install from the repository checkout:

```powershell
python -m pip install -e ".[full]"
python -m pip install -r examples/web_demo/requirements.txt
python examples/web_demo/server.py --download-models --accept-model-license
```

Open <http://127.0.0.1:8000>. Models are downloaded once to the normal package
cache; on later runs omit `--download-models` to require the verified cache.

The demo binds to localhost by default. Do not expose it to the internet or use
it as an authentication service without adding application-specific session
security, rate limiting, CSRF protection, consent handling, and a reviewed
evidence-retention policy.
