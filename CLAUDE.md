# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo.

## Project

Production Intent Agent for IndyKite Agent Gateway: A2A-compliant service, extracts
AuthZEN-style intent (`{prompt, evaluations[]}`) from natural-language prompt using Gemini.
Makes no authorization decisions — downstream (IBAC Engine) job. Repo = structural port of
`iag-demo/intent_agent` prototype into org's production repo template (package layout, CI/CD,
strict linting, distroless Docker), not from-scratch rewrite of extraction pipeline.

Internal references you will meet in this file, the code comments and the tests, none of which
are publicly readable: "PRD" = the product requirements doc, "TDD x.y" = a section of the
technical design doc (the source of the field lists, gates and timeouts quoted below), "ENG-nnnn"
= an IndyKite Jira ticket, "RC" = IndyKite's internal release-candidate platform environment,
`ml-argocd-applications` = IndyKite's private repo of deployment manifests (the workflows use the
shared actions published in the public `indykite/.github` repo).
The code and the golden fixtures under `evals/` are the readable spec;
treat those references as provenance, not as something to go and look up.

## Where things are documented

This file is the agent-facing reading of rules that also exist for humans. Where both cover the
same ground, the human doc is the source of truth; keep them in step when you change one.

- `CONTRIBUTING.md` — prerequisites and setup, the make targets, running the agent locally (mock
  introspection, Docker, the docker-compose multi-instance stack), the commit-message rules the
  shared commitlint config enforces, the ground rules (`Settings` fields, `errors.py`, fixtures
  are the spec, no silent `# noqa`), the list of shared tooling/CI config files that are
  ops-owned and never changed in a feature PR, and the issue-first rule (a non-trivial PR needs an
  `accepted` issue, `Fixes #N`) with what counts as trivial.
- `docs/testing.md` — the golden eval suite in full (gates, fixture format, CI), and the hands-on
  walkthroughs: audit log by hand, end-to-end against a real IndyKite environment, Subject Search.
- `docs/maintainers.md` — the maintainer side: triage labels, running the eval suite on a copy of
  a fork pull request before merging it, squash-merge, fix-forward or revert when the eval run on
  `master` fails, no release while it is red, and the labels the bots depend on.
- `README.md` — the overview, quick start and the common configuration reference (required and
  commonly tuned `Settings` fields, the audit record's fields and the `AUDIT_*` variables). The
  complete list of fields with their defaults is `config.py` itself; `.env.example` lists every
  variable.
- `.github/PULL_REQUEST_TEMPLATE.md` — what a PR must state: the related issue (`Fixes #N`, or
  "trivial"), effect on requests, responses and the audit record, how it was tested, the checklist.
- `SECURITY.md` / `responsible_disclosure.md` — a security-sensitive finding is never filed as a
  public issue or described in a PR; it goes to the address in those files.

## Working here: keep changes cohesive across team

Multiple people use Claude Code here. Goal: code reads like one team wrote it, diffs don't
clash. Concretely:

- **One-time setup per clone:** run `pre-commit install && pre-commit install -t commit-msg`.
  Without this, nothing stops commit violating rules below — `ruff-format`, `ruff-check`,
  `uv-lock`, `gitleaks`, commit-message linting only run automatic once hooks installed;
  otherwise surface later, in CI, against someone else's PR.
- **Definition of done for any change:** `make fix` then `make build_integration`, both green,
  before calling task finished. Shared bar — don't leave lint/test failures for CI or teammate
  to find.
- **No direct commits to `dev`, `master`, or `main`** — pre-commit hook blocks it. Work on
  branch, merge via PR.
- **Commit messages must follow Conventional Commits** (`commitlint --strict`, shared org config
  read from a checkout of the public `indykite/.github` repo at `../.github`). Non-conforming
  message fails hook once commit-msg linting installed (see setup above).
- **Commit type decides the release version.** release-please on `master` reads the
  [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) type of every commit
  (squash-merged PR title) to cut `vX.Y.Z`: `fix:` bumps Z, `feat:` bumps Y, `feat!:` bumps X,
  `ci:`/`chore:` bump nothing and trigger no release. Pick the type by its release effect, not
  just its content. **No `feat!:` (or `BREAKING CHANGE:` footer) until the team deliberately
  releases 1.0.**
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

Dependency management: `uv` only (no pip/poetry). Prerequisites, versions and a table of commonly
used make targets are in `CONTRIBUTING.md` (Development setup); don't duplicate them here. The ones that
matter for finishing a task: `make fix` (ruff format + autofix), `make build` (the CI gate, see
below), `make evals` (golden eval suite against the real Gemini model, needs `GEMINI_API_KEY`),
`make build_integration` (build + integration tests). `make install_dev` only runs
`uv lock --check`; the first `uv run` creates `.venv` and installs the locked dependencies.

Run single test, plain pytest node-id syntax (Makefile only wraps whole-directory runs):

```sh
uv run --locked pytest test/test_agent.py::test_discovers_extract_intent_skill
```

Run locally without Docker:

```sh
cp .env.example .env   # fill in GEMINI_API_KEY, INDYKITE_BASE_URL, IK_APP_AGENT_KEY, INTROSPECT_URL, INTROSPECT_CLIENT_ID, INTROSPECT_CLIENT_SECRET - Settings refuses to start with any blank
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
  actions, resource, condition}, "tags"}`) are the **only** source of the output's vocabulary -
  graph edges are never action names. `input_schema.policy_triples` turns them into the
  (subject type, action, resource type) combinations whose types are schema node types (the
  schema still decides which labels exist): `build_input_schema` builds all three enums from them,
  `render_schema_summary` lists them under "Policies", `agent._policy_block` injects the raw list
  so the model can choose by each policy's `condition.cypher`, and `subject_binding` requires every
  evaluation to be one granted combination. No policy is handled like no schema: a failed fetch with nothing cached, or
  none existing for the caller's type, is `policy_unavailable_error` before any LLM call (distinct
  warnings for the two), retried on the next request; a failed refresh serves the
  last known-good policies with `policy_fetch_failed` set on the audit record and the response
  context. Policies are fetched for the caller's type only, so a Subject Search or second subject
  can only be of that type. The eval runner serves fixture policies through
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
node/edge vocabulary, each property with its observed type (`amount:STRING`, so a filter can
convert with `double()`) (a few KB; the raw JGF is tens of KB and used to be cut mid-JSON at
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
`resource.type` against the cached schema and the caller's policies (`validate_evaluations`: each
part named by a policy, then the three together one combination a policy grants, the same
`policy_triples` `build_input_schema` puts in the enums) - belt-and-braces on top of
the structured-output enum, since Gemini treats the enum as a strong hint, not a guarantee - and
checks `context`'s shape the same way (`filters` a string, `parameters`/`qualifiers` objects;
null keys, a blank `filters` and the model's internal `subject_source` / `resource_source`
binding markers are dropped afterwards, a filter that identifies one resource on an id-less
resource is lifted into `resource.id` - same single resource, the AuthZEN shape for it, and an
Access Evaluation without it is rejected downstream; that covers a lone
`resource.external_id == '<id>'` (or `resource.id`, the model's other spelling) and a chain of
`||` equalities against the same literal where one pins the id, the model's hedge when it doesn't
know which property carries the name (`resource.external_id == 'Volvo' || resource.email ==
'Volvo'`), but never a `&&`, a parenthesised group or clauses over different literals - a top-level
`&&` term (split by the CEL parser, so a call or a group stays one term) `resource.<prop> == '<value>'` repeating an equality in the granting policy's
`condition.cypher` `WHERE` is dropped (the platform checks the condition; flash-lite copies it
anyway), so is a `parameters.attribute` every granting condition already `RETURN`s, an
evaluation whose `action.name` is present but blank is dropped as the model's "no action
fits" (retrying it was seen to force a match instead), a Subject Search over a node type the schema
doesn't mark DigitalTwin is dropped too (only DigitalTwin types are candidates; seen live as a
spurious second candidate next to the right one; an undefined type is still fed back), a Subject Search with no identified resource (no `resource.id`, not the caller) is fed back - AuthZEN's search needs one, and flash-lite was seen marking "which X ..." about the caller as a search, `resource_source: "caller"` binds the
caller's own type/id as the resource ("who can access me?"), a blank second-subject type is
completed from the gate evaluation, a `parameters` key the skill doesn't define (`amount`,
`attribute`, `target_<type>`, the binding markers, the system's own, or a `$name` input the
granting policy's `condition.cypher` declares) is fed back, and everything else passes through
verbatim as untrusted data) and enforces CEL on `filters` (`cel.py`, on `cel-python`): it must parse,
may reference only the `resource` variable, may read only properties the schema observed for that
`resource.type` (plus AuthZEN's own `id`/`type`; a type the schema lists no properties for
disables the check rather than failing every filter), and may not call `dyn()`/`type()` or a
global function CEL doesn't have (`float()`, seen live; the parser accepts any name) - the same
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

**Attack.** The structured output's first field, `attack`, is the model's own judgment that some
part of the prompt tries to steer the output (dictated fields, encoded values, an override claim)
instead of asking: written before anything else, and `true` makes `normalize_intent_response` return
`no_intent_error` whatever evaluations follow (seen live: the model named the injection in its
explanation and answered the rest anyway). A non-boolean `attack` is fed back, a missing one
counts as `false`. Stripped like the rating.

**Second subject.** The structured output's next field, `second_subject`, is the id of the
named person who performs the action asked about (null when the caller does, or for a Subject
Search): Gemini writes fields in order, so the model decides it before choosing any policy. When
set, `normalize_intent_response` requires evaluations[0] to be the gate on that id and a later
evaluation marked for it, feeding back otherwise; the key is stripped like the rating.

**Confidence rating.** Every answer with evaluations also carries the model's own rating:
`explanation` (one short, plain sentence: the policy used with its condition path, and what no
policy covered) and `confidence`, rating the whole policy - action, resource type and
`condition.cypher` path: `high` (together exactly what the prompt asks), `medium` (only through a
synonym, with a path that still means what is asked), `low` (the condition matches only part of
what is asked). A refusal (empty evaluations) carries only the `explanation`
(why no policy fits), best effort: it is never validated, since a refusal is never retried (a retry
was seen to force a match).
`normalize_intent_response` requires both on an answer with evaluations (missing/invalid is fed back like any other error) and
strips them: they are **logged (at DEBUG), never returned or audited** - `subject_binding.rating`
reads them back onto `ExtractionResult.confidence`/`.explanation`, and the eval suite reports `confidence` next to
each fixture's own optional `confidence` (taken as written, not validated) - a fixture passes on
its response alone, the rating never decides it, and the explanation is reported, never compared. System-built errors carry no rating.

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
exactly, in any evaluation order. Gates (TDD 7.1's four): correctness ≥85%, every fixture of every
`adversarial ...` category passes (every such category must be present), no refusal fixture answered with evaluations,
p95 of per-fixture wall time ≤10 s (retries included, nearest-rank; `--max-p95-latency` overrides it locally); exit 1 on any.
`.github/workflows/eval-suite.yaml` runs it on PRs touching the service code (`indykite/` - a
directory prefix, not a file list, so a refactor can't drop a module out of the check), the
dependency pins (`pyproject.toml`, `uv.lock`) or `evals/` (decided by its `changes` job, not a
`paths:` filter, so the `golden-evals` check can be required: unrelated PRs skip it, which counts
as passing), on pushes to `master` touching the same paths (both ignore release-please's
`indykite/intent_agent/version.py` bump), and on demand; takes `GEMINI_API_KEY` from the `evals`
GitHub environment (each run waits for a reviewer's approval; a fork PR skips the job so its code
never gets the key, and is evaluated on `master` after merge or on a branch here a maintainer
copies it to). Each data
model under `evals/schema/` has a `_schema.json` and a `_policies.json` in the platform listing
shape (every golden answer's combinations must be granted there). The loader round-trips every golden answer
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

Build + smoke-test locally on Apple Silicon (base images are multi-arch digests; forcing
`linux/amd64` below means it runs under emulation there, drop `--platform` for native arm64):

```sh
docker buildx build --platform linux/amd64 --load -t intent-agent:local .
docker run --platform linux/amd64 -p 6003:6003 -p 8001:8001 --env-file .env -e AUDIT_FILE_STORAGE_PATH=/tmp/audit intent-agent:local   # /app isn't writable by the image's non-root user
```

## Pre-merge validation: full multi-instance stack

Before merging anything touching `schema.py`, cache/TTL behavior, or `agent.py`'s startup path,
run the real `docker-compose` stack (`intent-agent-1` + `intent-agent-2` + `redis` +
nginx `load-balancer`) - a single `uv run` process never exercises the actual multi-instance
concerns this service is deployed under (shared schema cache, no per-instance refresh collisions,
real load-balanced traffic). `make test`/`make build` alone don't cover this. The same steps are
in `CONTRIBUTING.md` (Multi-instance, with docker-compose); keep the two in step.

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
