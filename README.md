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

### With Docker

```sh
make build_docker_image
```

`make run_docker_image` runs the image as-is, with no ports published and no env file — useful
as a smoke test that the image starts, but not for actually reaching the agent. To run it
reachable locally (the base images are pinned to `linux/amd64`, so this runs under emulation on
Apple Silicon):

```sh
docker buildx build --platform linux/amd64 --load -t intent-agent:local .
docker run --platform linux/amd64 -p 6003:6003 -p 8001:8001 --env-file .env intent-agent:local
```
