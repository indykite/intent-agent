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

### Audit log

Every request produces one audit record (TDD 4.9) with the fields `timestamp`, `request_id`,
`subject_id`, `subject_type`, `requesting_agent_id`, `context_id`, `prompt`, `evaluations`,
`schema_version`, `llm_provider`, `llm_model`, `latency_ms`, `validation_retries`, `outcome`
(`success` / `validation_failed` / `unmatched` / `error`) and `message`. Delivery mirrors the
[Agent Gateway's audit configuration](https://docs.indykite.com/docs/agent-gateway/configuration#audit),
flattened to env vars:

| Env var | IAG key | Meaning |
| --- | --- | --- |
| `AUDIT_DELIVERY` | `audit.delivery` | **Required.** `file` or `webhook` |
| `AUDIT_ASYNC` | `audit.async` | `true` buffers records behind a background worker (default `false`) |
| `AUDIT_ASYNC_BUFFER_SIZE` | `audit.async.buffer_size` | Queue size; records are dropped with a warning when full (default `512`) |
| `AUDIT_FILE_STORAGE_PATH` | `audit.file.storage_path` | Directory for `audit-<name>-<unixNano>.<format>` files, created if missing |
| `AUDIT_FILE_FORMAT` | `audit.file.format` | `csv`, `json` (one object per line) or `txt` |
| `AUDIT_FILE_ROTATION_STRATEGY` | `audit.file.rotation_strategy` | `time`, `size` or `size-and-time` |
| `AUDIT_FILE_ROTATION_INTERVAL_SECONDS` | `audit.file.rotation_interval` | Default `86400` (24h) |
| `AUDIT_FILE_ROTATION_MAX_BYTES` | `audit.file.rotation_max_bytes` | Default `104857600` (100 MiB) |
| `AUDIT_HTTP_URL` | `audit.http.url` | Webhook receiving one request per record, the record as JSON body |
| `AUDIT_HTTP_METHOD` | `audit.http.method` | `POST` or `PUT` |
| `AUDIT_HTTP_AUTH_TYPE` | `audit.http.auth.type` | `mTLS`, `api-key` or `no-auth` |
| `AUDIT_HTTP_MTLS_CERTIFICATE_FILE_PATH` | `audit.http.auth.mtls_certificate_file_path` | Client certificate (mTLS, TLS 1.3 minimum) |
| `AUDIT_HTTP_MTLS_PRIVATE_KEY_PATH` | `audit.http.auth.mtls_private_key_path` | Client key (mTLS) |
| `AUDIT_HTTP_API_KEY` | `audit.http.auth.api_key` | Key sent in the header below (api-key) |
| `AUDIT_HTTP_API_KEY_HEADER` | `audit.http.auth.api_key_header` | Default `X-API-Key` |
| `AUDIT_HTTP_TIMEOUT_SECONDS` | - | Per-delivery timeout (default `5.0`) |

Unlike the Agent Gateway, auditing cannot be switched off: `AUDIT_DELIVERY` has no default and
the process fails at startup without it. A failed delivery (unreachable webhook, disk error) is
logged and never fails the request being audited.

#### Trying the audit log by hand

Prerequisites: the agent running locally with the mock IdP (see "Running the agent" below -
`manual_testing/mock_introspect.py` on port 9001, `INTROSPECT_URL=http://127.0.0.1:9001/introspect`).
`Settings` reads `.env` once at startup, so **restart the agent after every `.env` change** in
the steps below.

1. **File delivery, success.** With `.env.example`'s defaults every request appends one JSON line
   under `./.private/audit/`:

   ```sh
   ./manual_testing/call_agent.sh
   tail -n1 .private/audit/audit-intent_agent-*.json | jq .
   ```

   Expect `outcome: success`, the bound `evaluations`, `subject_id` from the mock IdP, a UUID
   `request_id` and a small `latency_ms`.
2. **Failures are audited with the caller's message.** Send the request without the
   `Authorization` header (401, `outcome: error`, `message: Authorization required`); stop the
   mock IdP and call again (`Token introspection request failed` - restart it afterwards); send
   an unrelated prompt (`outcome: unmatched`); set `SUBJECT_TYPE` to a type the schema doesn't
   have (`outcome: error` with the subject-type reason).
3. **Request id propagation.** Add `-H "X-Request-ID: my-trace-1"` to the curl: `request_id`
   becomes `my-trace-1`. A value over 40 characters falls back to a generated UUID.
4. **File formats and rotation.** `AUDIT_FILE_FORMAT=csv` (or `txt`) writes new records to a
   new `.csv` / `.txt` file - the old `.json` file stays as it was, so look at the new one:
   `cat .private/audit/*.csv`. `AUDIT_FILE_ROTATION_STRATEGY=size` with
   `AUDIT_FILE_ROTATION_MAX_BYTES=500` starts a new file on every request (a record with a
   prompt and evaluations is 600-900 bytes; rotation happens before a write once the current
   file is already at the limit).
5. **Webhook delivery.** Run the throwaway receiver, point the agent at it and restart:

   ```sh
   uv run python manual_testing/mock_audit_webhook.py   # prints each record it receives; --port N for another port
   # in .env: AUDIT_DELIVERY=webhook / AUDIT_HTTP_URL=http://127.0.0.1:9002/audit
   #          AUDIT_HTTP_METHOD=POST / AUDIT_HTTP_AUTH_TYPE=api-key / AUDIT_HTTP_API_KEY=anything
   ```

   The agent log shows `started auditing through HTTP with API key`; each
   `./manual_testing/call_agent.sh` makes the receiver print `POST /audit api-key='anything'`
   and the record. The rejected-request checks from step 2 produce `outcome: error` records here
   too.
6. **A failed delivery never fails the request.** Stop the receiver and call the agent: the
   response is the normal one and the agent log shows `audit: failed to send HTTP request`.
   Restart the receiver with `--fail` (answers 503): still the normal response, and the log shows
   `audit: failed to send HTTP request, unexpected status code` with `status_code: 503`.
7. **Webhook variants.** `AUDIT_HTTP_METHOD=PUT` and `AUDIT_HTTP_AUTH_TYPE=no-auth` make the
   receiver print `PUT /audit api-key=None`.
8. **Buffered delivery.** `AUDIT_ASYNC=true` delivers records from the background worker instead
   of inline; `AUDIT_ASYNC_BUFFER_SIZE=1` plus a burst of parallel requests logs
   `audit: dropping payload, async queue is full`. Ctrl+C the agent: queued records are flushed
   before `Application shutdown complete`.
9. **Startup validation.** Each of these refuses to start with an `audit: ...` message: no
   `AUDIT_DELIVERY`; `webhook` with an empty `AUDIT_HTTP_URL`; `api-key` auth with an empty
   `AUDIT_HTTP_API_KEY`; `AUDIT_HTTP_METHOD=GET`; `mTLS` without both file paths.

## Running the agent

### Locally, with uv

`.env.example` leaves several settings blank that `Settings` requires at startup - set at least
`GEMINI_API_KEY`, `INDYKITE_BASE_URL`, `IK_APP_AGENT_KEY`, `INTROSPECT_URL`,
`INTROSPECT_CLIENT_ID`, `INTROSPECT_CLIENT_SECRET`, and `SUBJECT_TYPE` before starting (see
"Testing end-to-end against real RC + Redis + Gemini" below for where to get real values, or a
local introspection stub, for the introspection/schema settings):

```sh
cp .env.example .env   # then fill in the required settings listed above
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
2. Bring up Redis only, then point at it via `localhost` - the `redis` hostname used by the
   `docker-compose` stack (see `.env.docker-compose.example`) only resolves inside that
   network, not when running the agent directly via `uv run`:

   ```sh
   docker compose up redis -d
   # then in .env: CACHE_BACKEND=redis / REDIS_URL=redis://localhost:6379
   ```

3. `INTROSPECT_URL` needs a real IdP or a throwaway local stub - there's no
   "accept any token" mode. A minimal mock that always returns an active token:

   ```python
   # mock_introspect.py
   import json
   from http.server import BaseHTTPRequestHandler, HTTPServer


   class Handler(BaseHTTPRequestHandler):
       def do_POST(self):
           body = json.dumps(
               {
                   "active": True,
                   "sub": "<customer_external_id>",
                   "act": {"sub": "<customer_external_id>"},
                   "aud": "<customer_external_id>",
               }
           ).encode()
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

### Testing end-to-end against real RC + Redis + Gemini with scripts

`make start_app` and `make start_mock_introspect_service` both run in the foreground, so start
each in its own terminal and wait for both to report ready before running the script:

```sh
# terminal 1
make start_mock_introspect_service

# terminal 2
export INTROSPECT_URL=http://127.0.0.1:9001/introspect
make start_app

# terminal 3, once both are up
manual_testing/call_agent.sh
```

### With Docker

```sh
make build_docker_image
make start_docker_app # runs the image reachable locally
```

### Multi-instance, with docker-compose

`docker-compose.yml` runs two agent instances behind an nginx load balancer, backed by a
shared Redis - the real deployment topology this service is designed for (shared schema
cache, no per-instance refresh collisions). A single `uv run` process never exercises this.

```sh
cp .env.docker-compose.example .env   # different from the plain .env.example above -
                                       # REDIS_URL/INTROSPECT_URL need container-network values
docker compose up --build -d
```

Then verify it's actually working as multi-instance, not just that it started:

```sh
# both instances started clean (real schema fetch, no traceback):
docker logs intent-agent-intent-agent-1-1
docker logs intent-agent-intent-agent-2-1

# they share one cache entry, not two - two keys means the cache-key-collision bug is back:
docker exec intent-agent-redis-1 redis-cli KEYS "ikg_schema*"

# the load balancer actually distributes traffic, not just accepts it:
curl -X POST http://localhost:6003/ ...   # (see the curl example above), a few times
docker logs intent-agent-intent-agent-1-1 | grep -c "POST /"
docker logs intent-agent-intent-agent-2-1 | grep -c "POST /"   # both should be nonzero
```
