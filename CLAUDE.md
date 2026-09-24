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
- **New `Settings` (pydantic) fields need validation, not just a type annotation.** A bare
  `str`/`float` default silently accepts blank or malformed values until they cause a confusing
  failure deep in a request path instead of a clear one at startup. Match the existing pattern:
  required fields use `Field(min_length=1)` plus a `field_validator` rejecting whitespace-only
  values (`min_length` alone accepts one) — and reach for a more specific pydantic type where one
  exists (e.g. `HttpUrl` for URL fields, see `introspect_url`) rather than a plain `str` plus
  hand-rolled checks.
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
make evals                 # golden eval suite against the real Gemini model (needs GEMINI_API_KEY)
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

Eight modules carry whole service:

- **`config.py`** — `Settings(BaseSettings)` single source of runtime config, loaded from
  environment or `.env`. `gemini_api_key` has no default, so constructing `Settings()` fails
  fast (`pydantic.ValidationError`) if missing — startup gate, not runtime check. New config =
  typed field here; don't reach for `os.getenv` elsewhere.
- **`agent.py`** — the extraction pipeline: `IntentAgent` class (owns LLM, discovered skills,
  schema cache, tool bindings, tool-call loop, validation retries), plus the `RequestSubject` it
  takes and the `ExtractionResult` it returns. Delegates skill discovery to `skills.py`, schema
  fetch/cache to `schema.py`'s `SchemaClient`. Knows nothing of A2A transport or auditing.
- **`executor.py`** — the A2A leg: `IntentExecutor` authenticates the caller (`auth.py`), reads
  `subject_type`/`subject_id` off the message metadata, runs `IntentAgent.run_extraction`, emits
  the task events, and audits every request exactly once (`_RequestAudit`). Depends on
  `agent.py`, never the other way round.
- **`app.py`** — `build_app(settings) -> (Starlette app, IntentAgent)`: wires `IntentAgent`,
  `IntentExecutor`, the audit sender and the A2A routes together = entrypoint for `__main__.py`
  and the integration tests.
- **`skills.py`** — `SKILL.md` discovery/YAML-frontmatter parsing under `skills/`, builds the
  tier-1 system-prompt catalog (`skill_catalog_preamble.txt` + skill names/descriptions) and the
  A2A `AgentSkill` list for the agent card.
- **`schema.py`** — `SchemaClient`: fetches/caches the IndyKite graph schema (JGF) over `httpx2`;
  `build_schema_cache(settings)` picks the `dogpile.cache` region it caches into.
- **`policy.py`** — `PolicyClient`: fetches/caches the caller's subject type's ACTIVE KBAC
  policies (`GET /access/v1/policies?subject_type=`, X-IK-ClientKey, per-subject-type cache key,
  retry + short failure backoff, no cold-start preload since the subject type is per request);
  `get_cached(subject_type) -> (policies, fetch_failed)`. The entries (`{"policy": {subject,
  actions, resource, condition}, "tags"}`) are the second source of action names next to graph
  edges (PRD erratum 2026-09-04): `input_schema` widens the enum with them, `render_schema_summary`
  lists them as "Policy actions", `agent._policy_block` injects the raw list too, and
  `subject_binding._action_name_error` accepts a policy action only when the evaluation is marked
  `action_derivation_rule: "policy_based"` and the policy's resource type matches. A failed fetch
  never fails the request: extraction runs edge-only and `policy_fetch_failed` is set on the
  audit record and the response context. The eval runner serves fixture policies through
  `evals.runner.FixturePolicyClient` (same `get_cached(subject_type)` contract).
- **`auth.py`** — pulls the Bearer `access_token` off the A2A request context and validates it via
  RFC 7662 Token Introspection against the external IdP (no local JWT decode/signature check -
  the introspection call itself is what covers signature validity). Every introspection refusal
  is a `TokenValidationError` classified by whose fault it is (`TokenFailure`), the IAG's split:
  `CALLER` (blank or inactive token, 401), `SERVICE` (the request never got an answer, 500),
  `PROVIDER` (any status instead of an answer, 4xx and 5xx alike, or a body that isn't an
  RFC 7662 answer, 502) - `executor.py` maps the class to the status, so an IdP outage never
  reads as a bad token. A missing `Authorization` header never reaches introspection: `executor.py`
  answers that 401 itself.
- **`__main__.py`** — process entrypoint (`python -m indykite.intent_agent`). Wires `Settings()`,
  calls `build_app`, starts background health server on `HEALTH_PORT` (default 8001) using
  stdlib `http.server` — deliberately **not** part of main Starlette app, second listener —
  then `uvicorn.run`s main app on `INTENT_PORT` (default 6003).

**Skills / progressive disclosure.** `skills/<name>/SKILL.md` (YAML frontmatter + markdown body)
discovered by `skills.discover_skills()` relative to `skills.py`'s own directory, at `IntentAgent`
construction time. System prompt only lists skill names/descriptions; LLM calls `activate_skill`
to fetch skill's full body on demand, and for `extract-intent` specifically, the cached IndyKite
schema gets injected into that response as `input_schema.render_schema_summary`'s compact
node/edge vocabulary (a few KB; the raw JGF is tens of KB and used to be cut mid-JSON at
`SCHEMA_PREVIEW_CHARS`, which then sent the model re-fetching until the tool budget was gone). Currently one skill ships:
`extract-intent` — deliberately no separate `get-data-schema` skill; schema fetching = internal
infrastructure (`SchemaClient`, wired through `IntentAgent.load_data_schema`/
`get_cached_schema`), not LLM-facing capability. A `get_data_schema` tool is exposed to the LLM
too — separate from `activate_skill`'s auto-injection.

**Schema loading.** `SchemaClient.load()` fetches IndyKite graph schema (JGF) from
`{INDYKITE_BASE_URL}/data-schema/v1` using `IK_APP_AGENT_KEY` (as `X-IK-ClientKey` header),
best-effort-cached once at startup via `IntentAgent.preload_schema()`. Both `INDYKITE_BASE_URL`
and `IK_APP_AGENT_KEY` are required (`Settings` fails fast at startup if either is missing); the
fetch itself failing (network/HTTP error, or the platform not being up yet) does **not** prevent
cold start - the process keeps serving, and `run_extraction()` retries the fetch (via
`get_cached_schema()`) on every request until the schema is populated, returning
`schema_unavailable_error` for requests in the meantime. Once a schema is available, the
caller-supplied `subject_type` (see below) is validated against it on every request too
(`validate_subject_type`) - a mismatch fails that request with `subject_type_mismatch_error`
rather than crashing the process. `SCHEMA_REFRESH_TTL_MINUTES` controls how long a cached schema is served
before the next request triggers a refresh; a failed refresh after a cached schema has expired
keeps the stale schema instead of losing it.

**Validation loop (TDD 4.8/4.9).** After the LLM's final answer, `subject_binding.py`'s
`normalize_intent_response` checks every evaluation's `subject.type` / `action.name` /
`resource.type` against the cached schema (`validate_evaluations`; `action.name` against edges
plus the policy-granted actions, the same union `build_input_schema` puts in the enum and
`render_schema_summary` lists as "Policy actions") - belt-and-braces on top of
the structured-output enum, since Gemini treats the enum as a strong hint, not a guarantee - and
checks `context`'s shape the same way (`filters` a string, `parameters`/`qualifiers` objects;
null keys, a blank `filters` and the model's internal `subject_source` / `resource_source`
binding markers are dropped afterwards, a filter that only pins `resource.external_id == '<id>'`
on an id-less resource is lifted into `resource.id` - same single resource, the AuthZEN shape for
it - an evaluation whose `action.name` is present but blank is dropped as the model's "no action
fits" (retrying it was seen to force a match instead), a Subject Search over a node type the schema
doesn't mark DigitalTwin is dropped too (only DigitalTwin types are candidates; seen live as a
spurious second candidate next to the right one; an undefined type is still fed back), `resource_source: "caller"` binds the
caller's own type/id as the resource ("who can access me?"), a blank second-subject type is
completed from the gate evaluation, and everything else passes through verbatim as untrusted data) and enforces CEL on `filters` (`cel.py`, on `cel-python`): it must parse,
may reference only the `resource` variable, and may not call `dyn()`/`type()` - the same
posture as the platform's (blocked reflection functions, size limit). The
expression is never evaluated here; IBAC does that. An
off-schema value (or non-JSON output) raises `IntentValidationError`; `IntentAgent` feeds the
errors back to the LLM as a follow-up message (`validation_feedback_error`) in the same
conversation and asks again, up to `VALIDATION_MAX_RETRIES` extra times (default 1, `0`
disables retry), then returns `no_intent_error`. An empty `evaluations` array is the LLM's
legitimate "nothing coherent" answer and becomes the same `no_intent_error` without a retry.
Each LLM call is bounded by `LLM_TIMEOUT_SECONDS` (per call, like the other
`*_TIMEOUT_SECONDS`), and the whole extraction - every tool-call round and validation retry
of one request - by `REQUEST_TIMEOUT_SECONDS` (default 10, the PRD's end-to-end limit; the
per-call default is sized so the common activation + answer + retry path fits inside it).
Both surface as `llm_timeout_error`. Exhausting `TOOL_CALL_MAX_ITERATIONS`
after `extract-intent` was activated is also `no_intent_error`, never the free-text placeholder;
a repeat `activate_skill` for the already-active skill gets a short "answer now" tool message
instead of the skill body plus schema again (seen live: Gemini otherwise re-activates until the
budget is gone). An `{evaluations}` answer given *without* ever activating the skill (the bound
structured output yields that shape regardless) is never passed through as free text - it would
be unvalidated and carry whatever subject the model chose; `_activate_and_ask_again` injects the
skill content as the next message and the conversation is an extraction from there.
`resource.id` is a required string in the structured output (a required field is filled far
more reliably than an optional one; `""` = many resources, dropped again in normalisation). Every error - caller-facing or fed
back to the LLM - is the fixed `{prompt, error, reason, obligation}` shape built by
`errors.error_response`; new ones go in `errors.py` next to the existing ones, never as ad-hoc
dicts or strings elsewhere.

**Audit log (TDD 4.9).** Every request produces exactly one audit record - a successful
extraction, an `unmatched`/`validation_failed` one, a rejected token (audited as
`outcome = error` with the same reason the caller got, like the Agent Gateway audits every
rejection), and an unexpected exception alike. The record is the TDD's field list verbatim
(`audit/message.py`'s `AuditMessage`: snake_case, `outcome`/`message`, `prompt` and
`evaluations` recorded in full; `message` follows the IAG's `reason` rule and always equals the text the caller
received. Delivery is the IAG's, ported
one `Sender` per IAG sender under `audit/`: `file.py` (rotating `audit-<service>-<unixNano>.<fmt>`
files, json/csv/txt), `http.py` (one POST/PUT per record, mTLS/api-key/no-auth, failures logged
never raised), `async_sender.py` (bounded queue + worker, drop-on-full, drained by the Starlette
lifespan on shutdown), picked by `sender.build_sender` from the flat `AUDIT_*` `Settings` fields
that mirror the IAG's `audit.` YAML keys. There is no
no-op sender - `AUDIT_DELIVERY` is required, so a deployment can't silently run unaudited.
`IntentAgent.run_extraction` returns an `ExtractionResult` (response text plus outcome, message,
evaluations, retry count, schema hash) so `IntentExecutor` never parses its own output to
classify it; a new exit path from the pipeline must set its outcome there, not in the executor.

**Schema cache backend.** `CACHE_BACKEND` (`Settings`, default `"memory"`) picks the
`dogpile.cache` backend `build_schema_cache` (`schema.py`) configures: in-process `memory`, or
`redis` using `REDIS_URL` — required whenever `CACHE_BACKEND=redis` (`Settings` fails fast at
startup otherwise). Multi-instance deployments need `redis` so instances share one schema cache
entry (see the pre-merge multi-instance validation below).

**Golden eval suite (TDD 7.1/7.2).** `evals/` = fixtures (`<category>_test_<n>.json`, schemas
under `evals/schema/`) + the runner package (`fixtures.py` load/validate, `runner.py` drives
`IntentAgent.run_extraction` with the fixture schema served in place of the platform's,
`scoring.py` verdict/categories/gates, `report.py` text/Markdown/JSON, `__main__.py` CLI).
Not shipped in the image, not under `test/` - it calls the real Gemini model, so it's
`make evals` / `python -m evals`, never part of `make test`. Pass rule: response == `expected`,
exactly (evaluation order included). Gates: correctness ≥85%, every fixture of every
`adversarial ...` category passes (every such category must be present), no refusal fixture answered with evaluations; exit 1 on any.
`.github/workflows/eval-suite.yaml` runs it on PRs touching the service code (`indykite/**` - a
directory glob, not a file list, so a refactor can't drop a module out of the trigger), the
dependency pins (`pyproject.toml`, `uv.lock`) or `evals/**`, and on demand; needs the
`GEMINI_API_KEY` repo secret. Each data
model under `evals/schema/` has a `_schema.json` and, for policy-granted actions, a
`_policies.json` in the platform listing shape. The loader round-trips every golden answer
through `normalize_intent_response` as the fixture's caller and rejects one that comes back
different - the pipeline's own rules are the producibility check, not a copy of them. New fixture category
or gate = `fixtures.CATEGORIES` / `scoring.evaluate_gates`, not a parallel script. The fixtures
are the validated spec (they outrank the TDD's fixture shape): a red gate means the pipeline
doesn't produce the golden answer yet (e.g. context markers or action names the current skill /
schema validation don't support) - change the skill/pipeline, don't loosen the pass rule or
edit golden answers to fit.

**A2A transport quirks** (worth knowing before curling this service or writing integration
test): JSON-RPC method name is `SendMessage`, not `message/send`; requests need
`A2A-Version: 1.0` header; message `role` must be literal `ROLE_USER`; `Authorization: Bearer
<token>` required and validated via RFC 7662 introspection against the external IdP (real network
call - a valid-looking-but-fake token, like the old JWT-decode design accepted, is now rejected) -
this authenticates the calling client, not any subject, so `subject_type`/`subject_id` are no
longer derived from it. Both are required in the request's `message.metadata` instead (ENG-9810:
missing/blank is a 400 for either one now, since `subject_id` used to come reliably from the
token's own `sub` claim, so a missing one is a caller-side bug, not a legitimate anonymous-caller
request) - see `IntentExecutor._subject_params`. A Subject Search ("Who can do Y on Z?") is
unrelated to this: it is marked per evaluation by the LLM (`subject_binding._subject_search`),
not by the caller omitting `subject_id`.

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

## Pre-merge validation: full multi-instance stack

Before merging anything touching `schema.py`, cache/TTL behavior, or `agent.py`'s startup path,
run the real `docker-compose` stack (`intent-agent-1` + `intent-agent-2` + `redis` +
nginx `load-balancer`) - a single `uv run` process never exercises the actual multi-instance
concerns this service is deployed under (shared schema cache, no per-instance refresh collisions,
real load-balanced traffic). `make test`/`make build` alone don't cover this.

```sh
docker compose up --build -d
```

Then verify, don't just check it started:

- **Both instances started clean**: `docker logs intent-agent-intent-agent-1-1` /
  `...-2-1` - real schema fetch succeeded, no traceback, `Uvicorn running on ...:6003`.
- **They share one cache entry, not two**: `docker exec intent-agent-redis-1 redis-cli KEYS
  "ikg_schema*"` must show exactly one key (env-scoped, see `schema.py`'s `_cache_key`) - two
  keys means the cache-key-collision bug is back.
- **The load balancer actually distributes traffic**, not just accepts it: send a few requests
  through `http://localhost:6003/`, then confirm both `docker logs intent-agent-intent-agent-1-1
  | grep -c "POST /"` and the `-2-1` equivalent are nonzero.

`.env` needs container-network values for this, different from the bare `uv run` case above -
use `.env.docker-compose.example` (not the plain `.env.example`) as the starting point:
`REDIS_URL=redis://redis:6379` (not `localhost`), and if using a local mock introspection
endpoint on the host, `INTROSPECT_URL=http://host.docker.internal:9001/introspect` (macOS/Windows
Docker Desktop only - containers can't reach the host via `127.0.0.1`).
