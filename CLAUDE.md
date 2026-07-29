# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Production Intent Agent for IndyKite Agent Gateway: an A2A-compliant service that extracts
AuthZEN-style intent (`{prompt, evaluations[]}`) from a natural-language prompt using Gemini.
It does not make authorization decisions — that's downstream (IBAC Engine). This repo is a
structural port of the `iag-demo/intent_agent` prototype into this org's production repo
template (package layout, CI/CD, strict linting, distroless Docker), not a from-scratch rewrite
of the extraction pipeline.

## Working here: keep changes cohesive across the team

Multiple people use Claude Code in this repo. The goal is code that reads like one team wrote
it, and diffs that don't step on each other. Concretely:

- **One-time setup per clone:** run `pre-commit install && pre-commit install -t commit-msg`.
  Without this, nothing stops a commit that violates the rules below — `ruff-format`,
  `ruff-check`, `uv-lock`, `gitleaks`, and commit-message linting only run automatically once
  hooks are installed; otherwise they only surface later, in CI, against someone else's PR.
- **Definition of done for any change:** `make fix` then `make build`, both green, before calling
  a task finished. This is the shared bar — don't leave it to CI or a teammate to discover that
  lint or tests fail.
- **No direct commits to `dev`, `master`, or `main`** — a pre-commit hook blocks it. Work happens
  on a branch, merged via PR.
- **Commit messages must follow Conventional Commits** (`commitlint --strict`, shared org config
  via the `ops-common` repo). A non-conforming message fails the hook once commit-msg linting is
  installed (see setup above).
- **Stay inside the scope of your task.** `ruff`'s `select = ["ALL"]` (see below) will surface
  pre-existing issues in files you touch incidentally, not just the lines you meant to change.
  Fix what your task requires; don't drive-by-refactor a teammate's file because the linter found
  something else in it — raise it separately (issue, or its own PR) instead. Large, mixed-purpose
  diffs are exactly what causes unwanted clashes between people's work.
- **Don't route around the strict ruleset with a local `# noqa`.** If a rule genuinely doesn't fit
  a case, that's a `pyproject.toml` change the team agrees on together (it's not in
  `CODEOWNERS`, but it's shared config — flag it), not a silent per-line suppression. Scattered
  individual overrides are how a shared style config quietly stops being shared.
- **Extend the existing module boundaries** (`config.py` / `agent.py` / `__main__.py` / `skills/`,
  see Architecture below) rather than introducing a parallel pattern for something that already
  has a home — e.g. new runtime config is a typed field on `Settings`, not a fresh `os.getenv`
  call; a new capability for the LLM is a new `skills/<name>/SKILL.md`, not inline prompt text
  bolted onto `agent.py`.
- **`CODEOWNERS` gates review**, not just permissions, on `.github/`, `chart/`, `Dockerfile`, and
  the various lint/security config files (`@indykite/ai-ml-team`, `@indykite/ops`) — expect and
  route review there rather than treating them as free to change unilaterally.
- **Don't reformat whitespace beyond what a fix actually requires.** Re-aligning a Markdown
  table's column padding, reflowing unrelated lines, or letting an autoformatter touch a whole
  file is exactly the kind of change that turns a one-line fix into a noisy, hard-to-review diff.
  If a formatter/linter genuinely requires wider realignment to pass (e.g. `markdownlint`'s table
  rules), say so explicitly and let the person confirm before applying it — don't do it silently
  as a side effect of an unrelated task.

## Commands

Dependency management is `uv` only (no pip/poetry). GNU Make must be `>3.81`; on macOS the
Homebrew version installs as `gmake` — use that if `/usr/bin/make` is the stock 3.81.

```sh
make install_dev          # uv lock --check (verifies lockfile is current)
make fix                  # ruff format + ruff check --fix
make codestyle_check       # ruff check only (no autofix)
make test                  # uv run --locked pytest test
make test_coverage         # coverage across test/ + integration/
make build                 # the full gate: project_check + codestyle_check + test
make build_docker_image    # docker buildx build --platform linux/amd64 ...
make run_docker_image
```

Run a single test with plain pytest node-id syntax (the Makefile only wraps whole-directory runs):

```sh
uv run --locked pytest test/test_agent.py::test_discovers_extract_intent_skill
```

Run locally without Docker:

```sh
cp .env.example .env   # set GEMINI_API_KEY at minimum
uv run python -m indykite.intent_agent
```

`make build` is the actual CI gate: `project_check` validates `pyproject.toml` and the lockfile,
`codestyle_check` is `ruff check`, then `pytest test` runs.

## Linting mechanics (`select = ["ALL"]`)

`pyproject.toml` sets `ruff` to `select = ["ALL"]`, only ignoring docstring rules (`D`),
trailing-comma (`COM812`), and import-outside-top-level (`PLC0415`). In practice this means:

- **Docstrings are ignored by lint (`D`), but are still expected on every function, method, and
  class — including test functions in `test/` and `integration/`.** This is a team convention,
  not something `ruff` enforces — don't skip them just because nothing will flag it, and don't
  read the relaxed `test/**/*.py`/`integration/**/*.py` rules (`S101`, `ARG`, `FBT`, `SLF`, `ANN`,
  see above) as covering docstrings too, since it doesn't. For a test, one line stating what
  behavior it verifies is enough (e.g. "Settings must fail fast when GEMINI_API_KEY is missing").
  For everything else, one concise line describing what it does/returns; match the existing style
  rather than writing multi-paragraph docstrings.
- Magic numbers in comparisons (`PLR2004`) must be named module-level constants.
- Functions have enforced complexity/branch/return-count ceilings (`C901`, `PLR0911`,
  `PLR0912`) — if you're adding branches to `agent.py`'s tool-call loop or the skill parser,
  extract a helper method rather than growing the existing one.
- `test/**/*.py` and `integration/**/*.py` get relaxed rules (`S101`, `ARG`, `FBT`, `SLF`, `ANN`)
  via `[tool.ruff.lint.extend-per-file-ignores]` — but `PLR2004` and `INP001` still apply there,
  so test files need an `__init__.py` and named constants for any non-trivial literal too.
- Run `uv run --locked ruff check` before assuming a change is clean; `make fix` autofixes what
  it can, but complexity/magic-value findings need manual refactoring.

## Architecture

Code lives under `indykite/intent_agent/` (a namespace package under `indykite/`).

Every `__init__.py` in this repo (`indykite/__init__.py`, `indykite/intent_agent/__init__.py`,
`test/__init__.py`) uses the same `pkgutil.extend_path` namespace-package pattern (auto-generated
boilerplate, do not hand-edit or replace with real code) — for `indykite/*` this is what lets it
be re-opened by other, separately-installed `indykite.*` packages instead of being a closed,
single-repo package. Actual code always goes in a sibling module (`config.py`, `agent.py`, a
`test_*.py` file, ...), never in an `__init__.py`.

Three files carry the whole service:

- **`config.py`** — `Settings(BaseSettings)` is the single source of runtime config, loaded from
  the environment or `.env`. `gemini_api_key` has no default, so constructing `Settings()` fails
  fast (`pydantic.ValidationError`) if it's missing — this is the startup gate, not a runtime
  check. Add new config as typed fields here; don't reach for `os.getenv` elsewhere.
- **`agent.py`** — the actual pipeline: `SKILL.md` discovery/parsing, the Gemini LLM binding,
  the `IntentAgent` class (owns the LLM, discovered skills, schema cache) and the A2A
  `IntentExecutor`. `build_app(settings) -> (Starlette app, IntentAgent)` is the entrypoint into
  this module.
- **`__main__.py`** — process entrypoint (`python -m indykite.intent_agent`). Wires `Settings()`,
  calls `build_app`, starts a background health server on `HEALTH_PORT` (default 8001) using
  stdlib `http.server` — deliberately **not** part of the main Starlette app, it's a second
  listener — then `uvicorn.run`s the main app on `INTENT_PORT` (default 6003).

**Skills / progressive disclosure.** `skills/<name>/SKILL.md` (YAML frontmatter + markdown body)
are discovered relative to `agent.py`'s own directory at `IntentAgent` construction time. The
system prompt only lists skill names/descriptions; the LLM calls `activate_skill` to fetch a
skill's full body on demand, and for `extract-intent` specifically, the cached IndyKite schema
and the token's subject id get injected into that response. Currently one skill ships:
`extract-intent` — there's deliberately no separate `get-data-schema` skill; schema fetching is
internal infrastructure (`IntentAgent.load_data_schema`), not an LLM-facing capability.

**Schema loading.** `IntentAgent.load_data_schema()` fetches the IndyKite graph schema (JGF) from
`{INDYKITE_BASE_URL}/data-schema/v1` using `IK_APP_AGENT_KEY`, cached once at startup via
`preload_schema()`. If either env var is unset or the fetch fails, the agent still starts and
runs — the schema tool just reports it's unavailable and the LLM falls back to generic types.

**A2A transport quirks** (worth knowing before curling this service or writing an integration
test): the JSON-RPC method name is `SendMessage`, not `message/send`; requests need an
`A2A-Version: 1.0` header; message `role` must be the literal `ROLE_USER`; and any non-empty
`Authorization: Bearer <token>` is required — the `sub` claim is extracted from the JWT payload
directly, with no signature verification (that hardening is tracked separately, not implemented
in this port).

## Docker — two things that look wrong but are load-bearing

The multi-stage `Dockerfile` builds a `python-deps` stage then copies into a
`gcr.io/distroless/python3-debian13` runtime stage. Two non-obvious constraints:

1. **Copy order in the builder stage matters.** `pyproject.toml`/`uv.lock`/`README.md` and the
   `indykite/` source must all be copied in *before* `uv sync` runs, because `uv sync` builds and
   installs the local project itself (`[tool.setuptools] packages = ["indykite"]` in
   `pyproject.toml`), which needs the source tree and README present to succeed. Copying
   `indykite/` after `uv sync` breaks the build with a "package directory 'indykite' does not
   exist" error from setuptools.
2. **The runtime image has no shell and no `curl`.** `HEALTHCHECK` must be an exec-form
   (JSON-array) `python3 -c "..."` command using stdlib `urllib.request` — a shell pipeline or a
   `curl` invocation will fail on every single healthcheck, marking the container permanently
   "unhealthy" regardless of whether the app is actually fine.

To build and smoke-test locally on Apple Silicon (the base images are pinned to `linux/amd64`
digests, so this runs under emulation):

```sh
docker buildx build --platform linux/amd64 --load -t intent-agent:local .
docker run --platform linux/amd64 -p 6003:6003 -p 8001:8001 --env-file .env intent-agent:local
```
