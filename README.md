# Intent Agent

[![Build for master](https://github.com/indykite/intent-agent/actions/workflows/master-build.yaml/badge.svg)](https://github.com/indykite/intent-agent/actions/workflows/master-build.yaml)
[![codecov](https://codecov.io/gh/indykite/intent-agent/branch/master/graph/badge.svg?token=gpcjoHjGyj)](https://codecov.io/gh/indykite/intent-agent) <!-- skipcq: SCT-1000 -->

The Intent Agent turns a natural-language prompt into an
[AuthZEN](https://openid.net/specs/authorization-api-1_0.html)-style intent: a list of
`{subject, action, resource, context}` evaluations that an authorization engine can decide on. It
is an [A2A](https://a2a-protocol.org/)-compliant service built for the
[IndyKite Agent Gateway](https://developer.indykite.com/guides/guide-agent-gateway): the gateway authenticates
an AI agent, forwards its prompt here with the caller's identity, and sends the extracted intent
on to the IndyKite authorization engine.

What it does:

- Extracts `{prompt, evaluations[]}` from the prompt with Gemini, constrained by the vocabulary of
  your IndyKite Knowledge Graph (node types, relations and policy-granted actions fetched from the
  platform's data-schema and policy APIs).
- Binds the `subject` of every evaluation about the caller to the identity the gateway supplies
  with the request, so prompt text can never impersonate someone else. The only evaluations that
  keep a model-chosen subject are a Subject Search ("who can read X?", a type with no id) and the
  marked second-subject pattern gated by the caller's own first evaluation.
- Validates every evaluation against the schema, including any CEL `filters` expression, and asks
  the model again when it strays.
- Writes one audit record per request, to a file or a webhook.

What it does not do: make authorization decisions. A decision is the downstream engine's job.

## Quick start

Each release publishes the image to Docker Hub as `docker.io/indykite/intent-agent`, with a Helm chart
under `chart/`. `latest` tag is not published. To build locally instead:

```sh
cp .env.example .env     # fill in the required settings listed under Configuration
docker buildx build --platform linux/amd64 --load -t intent-agent:local .
docker run --platform linux/amd64 --rm -p 6003:6003 -p 8001:8001 --env-file .env \
  -e AUDIT_FILE_STORAGE_PATH=/tmp/audit intent-agent:local
```

The image runs as a non-root user and `/app` is not writable, so `.env.example`'s relative audit
directory must be overridden to a writable path such as `/tmp/audit` (as above); otherwise the
container exits at startup. The base images are pinned to multi-arch digests; the command above
forces `linux/amd64`, so on Apple Silicon it runs under emulation (drop `--platform` to build a
native arm64 image). The `make build_docker_image` target builds for `linux/amd64` and
`linux/arm64` at once with `--load`, which needs Docker's containerd image store; on the default
image store it fails with "docker exporter does not currently support exporting manifest lists".

Check it is up:

```sh
curl http://localhost:8001/health
curl http://localhost:6003/.well-known/agent-card.json
```

Send a prompt. This is the A2A JSON-RPC transport, not a REST API: note the `A2A-Version` header,
the `SendMessage` method name, the `ROLE_USER` role and the caller's `subject_type`/`subject_id`
in `message.metadata`. The Bearer token authenticates the calling client (via RFC 7662 token
introspection), not the subject. Every JSON-RPC method needs it, but only `SendMessage` is served
(`GetTask`, `ListTasks` and `CancelTask` answer "unsupported operation"); the agent card stays public:

```sh
curl -X POST http://localhost:6003/ \
  -H "Content-Type: application/json" \
  -H "A2A-Version: 1.0" \
  -H "Authorization: Bearer <token-accepted-by-your-introspection-endpoint>" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "SendMessage",
    "params": {
      "message": {
        "role": "ROLE_USER",
        "parts": [{"text": "Can I buy 50 shares of AAPL?"}],
        "messageId": "test-msg-1",
        "metadata": {
          "subject_type": "Customer",
          "subject_id": "test-subject"
        }
      }
    }
  }'
```

The HTTP response is an A2A JSON-RPC task envelope: `result.task.status.state` is
`TASK_STATE_COMPLETED` on success, and `result.task.artifacts[0].parts[0].text` holds the
extraction as a JSON string, either `{prompt, evaluations}` or, when nothing coherent can be
extracted, the error shape `{prompt, error, reason, obligation}`. To run from source, with a mock
introspection endpoint and no Docker, see
[`CONTRIBUTING.md`](./CONTRIBUTING.md#running-the-agent-locally).

## Configuration

All settings are environment variables (or a `.env` file), read once at startup by a validated
`Settings` object. Copy `.env.example` and fill in the blank values; the service refuses to start
if any of these is missing or blank:

| Setting | Meaning |
| --- | --- |
| `GEMINI_API_KEY` | Google Gemini API key used for extraction |
| `INDYKITE_BASE_URL` | IndyKite platform base URL; the graph schema and policies are fetched from it |
| `IK_APP_AGENT_KEY` | IndyKite AppAgent credential sent as `X-IK-ClientKey` |
| `INTROSPECT_URL` | RFC 7662 token introspection endpoint of the identity provider |
| `INTROSPECT_CLIENT_ID`, `INTROSPECT_CLIENT_SECRET` | Credentials for that endpoint |
| `AUDIT_DELIVERY` | `file` or `webhook`; auditing cannot be switched off |

Commonly tuned optional settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `INTENT_PORT`, `HEALTH_PORT` | `6003`, `8001` | A2A and health listeners |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model used for extraction |
| `LLM_TIMEOUT_SECONDS`, `REQUEST_TIMEOUT_SECONDS` | `3.0`, `10.0` | Per LLM call, and for the whole request |
| `VALIDATION_MAX_RETRIES` | `1` | Extra attempts after an off-schema answer (`0` disables) |
| `TOOL_CALL_MAX_ITERATIONS` | `5` | Tool-call rounds before giving up |
| `SCHEMA_REFRESH_TTL_MINUTES`, `POLICY_REFRESH_TTL_MINUTES` | `5.0` | How long a cached schema / policy list is served |
| `CACHE_BACKEND`, `REDIS_URL` | `memory`, - | `redis` shares the schema cache between instances; `REDIS_URL` is then required |
| `EXPECTED_TOKEN_AUDIENCE` | `intent_agent` | Required `aud` of the caller's token |
| `LOG_LEVEL` | `INFO` | Python log level (`.env.example` sets `DEBUG` for local runs) |

### Audit log

Every request produces one audit record with the fields `timestamp`, `request_id`,
`subject_id`, `subject_type`, `requesting_agent_id`, `context_id`, `prompt`, `evaluations`,
`schema_version`, `policy_version`, `llm_provider`, `llm_model`, `latency_ms`,
`validation_retries`, `policy_fetch_failed`, `outcome` (`success` / `validation_failed` /
`unmatched` / `error`) and `message`. The delivery options are
the same as the
[Agent Gateway's](https://developer.indykite.com/guides/guide-agent-gateway#configuration):

| Env var | Meaning |
| --- | --- |
| `AUDIT_DELIVERY` | **Required.** `file` or `webhook` |
| `AUDIT_ASYNC` | `true` delivers through a bounded in-memory queue drained by a background worker, flushed on shutdown (default `false`, inline delivery) |
| `AUDIT_ASYNC_BUFFER_SIZE` | Queue length; a full queue drops records with the warning `dropping payload, async queue is full` (default `512`) |
| `AUDIT_FILE_STORAGE_PATH` | Directory for `audit-<name>-<unixNano>.<format>` files, created if missing |
| `AUDIT_FILE_FORMAT` | `csv`, `json` (one object per line) or `txt` |
| `AUDIT_FILE_ROTATION_STRATEGY` | `time`, `size` or `size-and-time` |
| `AUDIT_FILE_ROTATION_INTERVAL_SECONDS` | Default `86400` (24h) |
| `AUDIT_FILE_ROTATION_MAX_BYTES` | Default `104857600` (100 MiB) |
| `AUDIT_HTTP_URL` | Webhook receiving one request per record, the record as JSON body |
| `AUDIT_HTTP_METHOD` | `POST` or `PUT` |
| `AUDIT_HTTP_AUTH_TYPE` | `mTLS`, `api-key` or `no-auth` |
| `AUDIT_HTTP_MTLS_CERTIFICATE_FILE_PATH` | Client certificate (mTLS, TLS 1.3 minimum) |
| `AUDIT_HTTP_MTLS_PRIVATE_KEY_PATH` | Client key (mTLS) |
| `AUDIT_HTTP_API_KEY` | Key sent in the header below (api-key) |
| `AUDIT_HTTP_API_KEY_HEADER` | Default `X-API-Key` |
| `AUDIT_HTTP_TIMEOUT_SECONDS` | Per-delivery timeout (default `5.0`) |

Auditing cannot be switched off: `AUDIT_DELIVERY` has no default and the process fails at startup
without it. The chosen delivery has required settings of its own, also checked at startup:
`file` needs `AUDIT_FILE_STORAGE_PATH`, `AUDIT_FILE_FORMAT` and `AUDIT_FILE_ROTATION_STRATEGY`;
`webhook` needs `AUDIT_HTTP_URL`, `AUDIT_HTTP_METHOD` and `AUDIT_HTTP_AUTH_TYPE`, plus both mTLS
file paths for `mTLS` or `AUDIT_HTTP_API_KEY` for `api-key`. A failed delivery (unreachable
webhook, disk error) is logged and never fails the request being audited.

## Releases

Releases are cut with release-please from Conventional Commits on `master`; the container image
and the Helm chart are published to Docker Hub on each release. IndyKite's own `dev` and `rc`
environments are updated automatically from `master` pushes and releases respectively; neither
is needed to build, run or contribute to the agent.

## Development and testing

- [`CONTRIBUTING.md`](./CONTRIBUTING.md): prerequisites, make targets, running from source with a
  mock introspection endpoint, Docker and the multi-instance docker-compose stack, the PR workflow.
- [`docs/testing.md`](./docs/testing.md): the golden eval suite that measures the real model
  against fixture prompts, and hands-on walkthroughs for the audit log, an end-to-end run against a
  real IndyKite environment, and Subject Search.
- [`docs/maintainers.md`](./docs/maintainers.md): for maintainers - triage, reviewing and
  evaluating pull requests from forks, merging, and what to check before a release.

## Contributing

Contributions are welcome. [`CONTRIBUTING.md`](./CONTRIBUTING.md) describes the workflow (fork,
branch off `master`, Conventional Commits, `make fix` then `make build_integration`), the ground rules for
changes to the pipeline, and which shared configuration files are kept identical across IndyKite
repos and should not be changed in a feature PR. Everyone participating is expected to follow the
[Code of Conduct](./CODE_OF_CONDUCT.md).

Before starting on a non-trivial change, open an issue and wait until a maintainer labels it
`accepted`; ideas and questions go to [Discussions](https://github.com/indykite/intent-agent/discussions).

## Security

Please do not report security issues in public issues or PRs. See [`SECURITY.md`](./SECURITY.md)
and the full [responsible disclosure policy](./responsible_disclosure.md) for the private
reporting channel, scope and response times.

## License

Licensed under the [Apache License, Version 2.0](./LICENSE); the `LICENSE` file at the repo root
applies to every file in the tree.
