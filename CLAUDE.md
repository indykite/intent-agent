# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo.

## Project

Production Intent Agent for IndyKite Agent Gateway: A2A-compliant service, extracts
AuthZEN-style intent (`{prompt, evaluations[]}`) from natural-language prompt using Gemini.
Makes no authorization decisions — downstream (IBAC Engine) job. Repo = structural port of
`iag-demo/intent_agent` prototype into org's production repo template (package layout, CI/CD,
strict linting, distroless Docker), not from-scratch rewrite of extraction pipeline.

## Working here: keep changes cohesive across team

Multiple people use Claude Code here. Goal: code reads like one team wrote it, diffs don't
clash. Concretely:

- **One-time setup per clone:** run `pre-commit install && pre-commit install -t commit-msg`.
  Without this, nothing stops commit violating rules below — `ruff-format`, `ruff-check`,
  `uv-lock`, `gitleaks`, commit-message linting only run automatic once hooks installed;
  otherwise surface later, in CI, against someone else's PR.
- **Definition of done for any change:** `make fix` then `make build`, both green, before calling
  task finished. Shared bar — don't leave lint/test failures for CI or teammate to find.
- **No direct commits to `dev`, `master`, or `main`** — pre-commit hook blocks it. Work on
  branch, merge via PR.
- **Commit messages must follow Conventional Commits** (`commitlint --strict`, shared org config
  via `ops-common` repo). Non-conforming message fails hook once commit-msg linting installed
  (see setup above).
- **Stay inside scope of your task.** `ruff`'s `select = ["ALL"]` (see below) surfaces
  pre-existing issues in files you touch incidentally, not just lines you meant to change.
  Fix what task requires; don't drive-by-refactor teammate's file cause linter found something
  else — raise separately (issue, own PR) instead. Large mixed-purpose diffs cause unwanted
  clashes between people's work.
- **Don't route around strict ruleset with local `# noqa`.** Rule genuinely doesn't fit case?
  That's `pyproject.toml` change team agrees together (not in `CODEOWNERS`, but shared config —
  flag it), not silent per-line suppression. Scattered overrides = shared style config quietly
  stops being shared.
- **Extend existing module boundaries** (`config.py` / `agent.py` / `auth.py` / `schema.py` /
  `skills.py` / `__main__.py` / `skills/`, see Architecture below) rather than parallel pattern
  for something already has home — e.g. new runtime config = typed field on `Settings`, not fresh
  `os.getenv` call; new LLM capability = new `skills/<name>/SKILL.md`, not inline prompt text
  bolted onto `agent.py`.
- **`CODEOWNERS` gates review**, not just permissions, on `.github/`, `chart/`, `Dockerfile`, and
  various lint/security config files (`@indykite/ai-ml-team`, `@indykite/ops`) — expect and route
  review there, not free to change unilaterally.
- **Don't reformat whitespace beyond what fix actually requires.** Re-aligning Markdown table
  column padding, reflowing unrelated lines, letting autoformatter touch whole file — exactly
  kind of change turns one-line fix into noisy hard-to-review diff. Formatter/linter genuinely
  needs wider realignment to pass (e.g. `markdownlint` table rules)? Say so explicit, let person
  confirm before applying — don't do silent as side effect of unrelated task.

## Commands

Dependency management: `uv` only (no pip/poetry). GNU Make must be `>3.81`; macOS Homebrew
version installs as `gmake` — use that if `/usr/bin/make` stock 3.81.

```sh
make install_dev          # uv lock --check (verifies lockfile is current)
make fix                  # ruff format + ruff check --fix
make codestyle_check       # ruff check only (no autofix)
make test                  # uv run --locked pytest test
make test_coverage         # coverage across test/ + integration/
make build                 # the full gate: project_check + codestyle_check + test
make build_docker_image    # docker buildx build --platform linux/amd64 ...
make start_docker_app
```

Run single test, plain pytest node-id syntax (Makefile only wraps whole-directory runs):

```sh
uv run --locked pytest test/test_agent.py::test_discovers_extract_intent_skill
```

Run locally without Docker:

```sh
cp .env.example .env   # set GEMINI_API_KEY at minimum
uv run python -m indykite.intent_agent
```

`make build` = actual CI gate: `project_check` runs `uv lock --check` + `uv run --locked
validate-pyproject pyproject.toml`; then `codestyle_check`/`test` both use `uv run --locked` too,
so `.venv` drifted from `uv.lock` fails loud right there instead of surfacing later as confusing
import error; then `codestyle_check` = `ruff check`, then `pytest test` runs.

## Linting mechanics (`select = ["ALL"]`)

`pyproject.toml` file contains `[tool.ruff.lint]` section with up-to-date rules.

Run `uv run --locked ruff check` before assuming change clean; `make fix` autofixes what it
can, but complexity/magic-value findings need manual refactor.

## Architecture

Code lives under `indykite/intent_agent/` (namespace package under `indykite/`).

Every `__init__.py` in repo (`indykite/__init__.py`, `indykite/intent_agent/__init__.py`,
`test/__init__.py`) uses same `pkgutil.extend_path` namespace-package pattern (auto-generated
boilerplate, don't hand-edit or replace with real code) — for `indykite/*` this lets it be
re-opened by other separately-installed `indykite.*` packages instead of closed single-repo
package. Actual code always goes in sibling module (`config.py`, `agent.py`, a `test_*.py` file,
...), never in `__init__.py`.

Six modules carry whole service:

- **`config.py`** — `Settings(BaseSettings)` single source of runtime config, loaded from
  environment or `.env`. `gemini_api_key` has no default, so constructing `Settings()` fails
  fast (`pydantic.ValidationError`) if missing — startup gate, not runtime check. New config =
  typed field here; don't reach for `os.getenv` elsewhere.
- **`agent.py`** — the pipeline glue: `IntentAgent` class (owns LLM, discovered skills, schema
  cache, tool bindings, tool-call loop), A2A `IntentExecutor`. `build_app(settings) -> (Starlette
  app, IntentAgent)` = entrypoint into this module. Delegates skill discovery to `skills.py`, JWT
  auth to `auth.py`, schema fetch/cache to `schema.py`'s `SchemaClient`.
- **`skills.py`** — `SKILL.md` discovery/YAML-frontmatter parsing under `skills/`, builds the
  tier-1 system-prompt catalog (`skill_catalog_preamble.txt` + skill names/descriptions) and the
  A2A `AgentSkill` list for the agent card.
- **`schema.py`** — `SchemaClient`: fetches/caches the IndyKite graph schema (JGF) over `httpx2`.
- **`auth.py`** — pulls the Bearer token off the A2A request context and decodes the JWT `sub`
  claim (no signature verification).
- **`__main__.py`** — process entrypoint (`python -m indykite.intent_agent`). Wires `Settings()`,
  calls `build_app`, starts background health server on `HEALTH_PORT` (default 8001) using
  stdlib `http.server` — deliberately **not** part of main Starlette app, second listener —
  then `uvicorn.run`s main app on `INTENT_PORT` (default 6003).

**Skills / progressive disclosure.** `skills/<name>/SKILL.md` (YAML frontmatter + markdown body)
discovered by `skills.discover_skills()` relative to `skills.py`'s own directory, at `IntentAgent`
construction time. System prompt only lists skill names/descriptions; LLM calls `activate_skill`
to fetch skill's full body on demand, and for `extract-intent` specifically, cached IndyKite
schema + token's subject id get injected into that response. Currently one skill ships:
`extract-intent` — deliberately no separate `get-data-schema` skill; schema fetching = internal
infrastructure (`SchemaClient`, wired through `IntentAgent.load_data_schema`/
`get_cached_schema`), not LLM-facing capability. A `get_data_schema` tool is exposed to the LLM
too — separate from `activate_skill`'s auto-injection.

**Schema loading.** `SchemaClient.load()` fetches IndyKite graph schema (JGF) from
`{INDYKITE_BASE_URL}/data-schema/v1` using `IK_APP_AGENT_KEY` (as `X-IK-ClientKey` header), cached
once at startup via `IntentAgent.preload_schema()`. Either env var unset or fetch fails — agent
still starts, runs — schema tool just reports unavailable, LLM falls back to generic types.

**A2A transport quirks** (worth knowing before curling this service or writing integration
test): JSON-RPC method name is `SendMessage`, not `message/send`; requests need
`A2A-Version: 1.0` header; message `role` must be literal `ROLE_USER`; any non-empty
`Authorization: Bearer <token>` required — `sub` claim extracted from JWT payload direct, no
signature verification (hardening tracked separately, not implemented in this port).

## Docker

Multi-stage `Dockerfile` builds `python-deps` stage then copies into
`gcr.io/distroless/python3-debian13` runtime stage.

1. **Copy order in builder stage:** `pyproject.toml`/`uv.lock` copied and `uv sync` run *first*
   (deps only, so this layer caches across source-only changes), `indykite/` source copied in
   *after*, then `pyproject.toml`/`uv.lock` removed from the image.
2. **Runtime image has no shell, no `curl`.** `HEALTHCHECK` must be exec-form (JSON-array)
   `python3 -c "..."` command using stdlib `urllib.request` — shell pipeline or `curl`
   invocation fails every single healthcheck, marks container permanently "unhealthy" regardless
   whether app actually fine.

Build + smoke-test locally on Apple Silicon (base images pinned to `linux/amd64` digests, runs
under emulation):

```sh
docker buildx build --platform linux/amd64 --load -t intent-agent:local .
docker run --platform linux/amd64 -p 6003:6003 -p 8001:8001 --env-file .env intent-agent:local
```
