# Intent Agent

[![Build Python project for master](https://github.com/indykite/intent-agent/actions/workflows/master-build.yaml/badge.svg)](https://github.com/indykite/intent-agent/actions/workflows/master-build.yaml)
[![Build & publish release artifacts](https://github.com/indykite/intent-agent/actions/workflows/publish-release.yaml/badge.svg)](https://github.com/indykite/intent-agent/actions/workflows/publish-release.yaml)
[![codecov](https://codecov.io/gh/indykite/intent-agent/branch/master/graph/badge.svg?token=gpcjoHjGyj)](https://codecov.io/gh/indykite/intent-agent) <!-- skipcq: SCT-1000 -->

## Release strategy

- `dev`: automatically on each push to `master`
- `rc`: automatically on each release
- `prod`: on each release via PR (needs to be merged manually)

## Local installation

These instructions are written for macOS 15, but mostly are OS-agnostic.

Pre-requisites:

- Python 3.13 (`brew install python@3.13`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv`)
- GNU make `> 3.81` (`brew install make` on macOS)
    - **Warning: version that is included with macOS by default is too old (`GNU Make 3.81`)**
    - Check your version by running `/usr/bin/make --version`
    - If `make` is installed via Homebrew, it will be available as `gmake`. Make sure to use `gmake` instead of `make` in all the commands below.
    - Alternatively, set an alias or add the proper path to the `PATH`
        - for instance, add the following line to `~/.zshrc`: `export PATH="$HOMEBREW_PREFIX/opt/make/libexec/gnubin:$PATH"`

```sh
make install_dev
```

## Build/execution make targets

Although python execution is done through `uv`, all commands are defined in the `Makefile`.

| Command                          | Purpose                                               |
|----------------------------------|-------------------------------------------------------|
| `make -i`                        | Run all checks (doesn't change files)                 |
| `make fix`                       | Re-format files: imports, code style (Changes files)  |
| `make test`                      | Run unit tests                                        |
| `make test_coverage`             | Run code coverage and integration tests               |
| `make test_coverage_report_text` | Report code coverage as text in cli                   |
| `make test_coverage_report_html` | Report code coverage as an index.html file in htmlcov |

## Configuration

Example `.env` file can be copied from `.env.example`:

```sh
cp .env.example .env
```

## Running the agent

### Locally, with uv

```sh
cp .env.example .env   # then set GEMINI_API_KEY
uv run python -m indykite.intent_agent
```

This starts the A2A JSON-RPC server on `INTENT_PORT` (default `6003`) and a health endpoint on
`HEALTH_PORT` (default `8001`).

Verify it's up:

```sh
curl http://localhost:8001/health
curl http://localhost:6003/.well-known/agent-card.json
```

Send it a test prompt (note the required `A2A-Version` header, the `SendMessage` method name, and
`ROLE_USER` role — this is the A2A JSON-RPC transport, not a REST API):

```sh
curl -X POST http://localhost:6003/ \
  -H "Content-Type: application/json" \
  -H "A2A-Version: 1.0" \
  -H "Authorization: Bearer <any-non-empty-token>" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "SendMessage",
    "params": {
      "message": {
        "role": "ROLE_USER",
        "parts": [{"text": "Can I buy 50 shares of AAPL?"}],
        "messageId": "test-msg-1"
      }
    }
  }'
```

### Testing end-to-end against real RC + Redis + Gemini

The quickstart above works with a schema-shaped local setup, but doesn't exercise the
real platform. To test against a real RC environment, a real Redis-backed schema
cache, and real Gemini calls:

1. Set `INDYKITE_BASE_URL=https://api.rc.indykite.xyz` and a valid `IK_APP_AGENT_KEY`
   for that environment in `.env` (ask a teammate/platform contact for a key - never
   commit one).
2. Bring up Redis only, then point at it via `localhost` - the `redis` hostname in
   `.env.example` only resolves inside the `docker-compose` network, not when running
   the agent directly via `uv run`:

   ```sh
   docker compose up redis -d
   # then in .env: REDIS_URL=redis://localhost:6379
   ```

3. `INTROSPECT_URL` needs a real IdP or a throwaway local stub - there's no
   "accept any token" mode. A minimal mock that always returns an active token:

   ```python
   # mock_introspect.py
   import json
   from http.server import BaseHTTPRequestHandler, HTTPServer

   class Handler(BaseHTTPRequestHandler):
       def do_POST(self):
           body = json.dumps({
               "active": True,
               "sub": "<customer_external_id>",
               "act": {"sub": "<customer_external_id>"},
               "aud": "<customer_external_id>",
           }).encode()
           self.send_response(200)
           self.send_header("Content-Type", "application/json")
           self.send_header("Content-Length", str(len(body)))
           self.end_headers()
           self.wfile.write(body)

   HTTPServer(("127.0.0.1", 9001), Handler).serve_forever()
   ```

   Run it, then set `INTROSPECT_URL=http://127.0.0.1:9001/introspect` with any
   `INTROSPECT_CLIENT_ID`/`INTROSPECT_CLIENT_SECRET`.
4. Send a prompt that actually matches your graph's schema. `intent-agent`'s subject
   is always auto-bound to the caller (from the introspection `sub` claim, per
   `SUBJECT_TYPE`), never from prompt text - a third-person prompt like "what can
   user X buy" doesn't test what it looks like it tests. Prefer first-person prompts,
   and check the platform schema for resource types that actually exist before
   assuming an entity is graph-modeled (e.g. an individual stock ticker usually
   isn't a node - that data tends to come from an external source instead).

### With Docker

```sh
make build_docker_image
make start_docker_app # runs the image reachable locally
```
