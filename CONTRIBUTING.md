# Contributing

Thanks for your interest in improving the Intent Agent. A non-trivial change needs an issue that a
maintainer has labelled `accepted` before you open a pull request; ideas and questions start in
[Discussions](https://github.com/indykite/intent-agent/discussions). Pull requests for non-trivial
changes without an accepted issue may be closed. A short sanity check from a maintainer can save
you an afternoon of work.

## Before you start

- Read the [README](./README.md) to understand what the service does: it extracts an AuthZEN-style
  intent from a natural-language prompt and hands it on. It makes no authorization decisions.
- Check open issues and PRs to make sure the change isn't already in flight.
- Read [`responsible_disclosure.md`](./responsible_disclosure.md) if your contribution touches
  anything security-sensitive. Do **not** open a public issue for security reports.

### What counts as trivial

A trivial change needs no issue; anything else needs an accepted one (see above).

- **Trivial:** typos and docs wording, formatting or lint fixes, test additions under `test/` or
  `integration/`, an obvious one-line fix that changes no behaviour, response or error, docs-only
  changes outside `skills/`, dependency bumps by bots.
- **Not trivial:** any change in behaviour, response, status or error; prompts
  (`skills/**/SKILL.md`); the golden eval fixtures (`evals/*.json`) and the data models and
  policies under `evals/schema/` - they are the spec for the model's behaviour, so changing a
  golden answer changes what the agent must answer; the `Settings`, A2A or audit interfaces; new
  dependencies; cross-module refactors; anything touching authentication, audit or security; CI,
  Docker or the Helm chart.
- When in doubt, open an issue first.

## From issue to merge

1. **Issue.** Open it with the **Bug report** form; a new bug report is labelled `needs-triage`.
   Ideas and questions go to [Discussions](https://github.com/indykite/intent-agent/discussions),
   not to an issue - there is no feature request form. When a maintainer accepts an idea, they open
   the issue themselves from the discussion, labelled `enhancement` and `accepted`, so the
   conversation stays linked; you then continue from step 3.
2. **Triage.** A maintainer (someone with write access - the code owners in
   [`CODEOWNERS`](./.github/CODEOWNERS)) replaces `needs-triage` with one of:
    - `accepted`: the change is agreed; a pull request may reference it.
    - `help wanted` (together with `accepted`): open for an outside contributor to pick up;
      `good first issue` marks small, self-contained ones.
    - `needs-info`: waiting for an answer from the reporter; once it comes, the maintainer triages
      the issue again. It may be closed if none comes.
    - `wontfix`, `duplicate` or `invalid`: closed, with a short reason.

   This applies to bug reports too: confirming a bug is part of triage, so wait for `accepted`
   before opening a pull request that fixes it - unless the fix is
   [trivial](#what-counts-as-trivial).
3. **Take it.** Comment on the issue that you are working on it, so nobody else starts the same
   change. If you stop, say so there too.
4. **Pull request.** Make the change as described in [Making changes](#making-changes) and open the
   pull request as described in [Pull requests](#pull-requests), with `Fixes #N`.
5. **Review.** The code owners review it, and the build, tests and SAST Linting run. If the pull
   request changes the service code, the dependency pins, `evals/` or
   `.github/workflows/eval-suite.yaml`, a maintainer also runs the eval suite on a copy of your
   reviewed commit before merging (see Making changes, step 5).
6. **Merge.** A maintainer squash-merges the pull request: its title and description become the
   commit on `master`, the issue closes automatically, and the change ships in the next release.

## Ground rules

- **Extraction is the LLM's job, validation is the pipeline's.** New model-facing behaviour goes in
  a skill under `indykite/intent_agent/skills/<name>/SKILL.md`, not as prompt text bolted onto the
  agent. New checks on what the model returns go next to the existing ones in `subject_binding.py`.
- **Configuration is a typed `Settings` field** in `config.py`, with validation, never a bare
  `os.getenv` somewhere else. A blank or malformed value must fail at startup, not deep in a request.
- **Every error has the one fixed shape** built by `errors.error_response`. Add new errors in
  `errors.py`; don't return ad-hoc dicts or strings.
- **The golden fixtures under `evals/` are the spec.** If a change makes a fixture fail, fix the
  skill or the pipeline, don't edit the golden answer or loosen the pass rule.
- **No silent `# noqa`.** The linter runs with every rule enabled on purpose. A suppression is
  fine for a genuine false positive on one line, but it must name the rule and say why, e.g.
  `# noqa: S104 - reachable from outside container`; never a bare `# noqa` or a file-wide one.
  If a rule doesn't fit the codebase as a whole, that is a `pyproject.toml` change for the team
  to agree on; propose it in the PR and say why.
- **Don't change the shared tooling and CI configuration.** `Makefile`, `Dockerfile`, `/.github/`,
  `/chart/`, the tool sections of `pyproject.toml`, `.pre-commit-config.yaml`, the lint and scanner
  configs (`.checkov.yaml`, `.deepsource.toml`, `.markdownlint.yaml`, `.trivy*`, `.yamllint.yml`,
  `.yamlfmt.yml`, `.shellcheckrc`) and the supported Python version are maintained by the IndyKite
  teams and kept identical across IndyKite repos; the ones listed in
  [`CODEOWNERS`](./.github/CODEOWNERS) additionally require a review by the ops team. The
  linter will surface pre-existing findings in files you touch; fix what your change needs and
  raise the rest in an issue. If one of these files genuinely needs to change for your work, open
  an issue first so the owners can make the change in a PR of its own.

## Development setup

These instructions are written for macOS 15, but mostly are OS-agnostic.

Pre-requisites:

- Python 3.13 (`brew install python@3.13`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv`)
- [pre-commit](https://pre-commit.com/#install) (`brew install pre-commit`)
- GNU make `> 3.81` (`brew install make` on macOS)
    - **Warning: version that is included with macOS by default is too old (`GNU Make 3.81`)**
    - Check your version by running `/usr/bin/make --version`
    - If `make` is installed via Homebrew, it will be available as `gmake`. Make sure to use `gmake` instead of `make` in all the commands below.
    - Alternatively, set an alias or add the proper path to the `PATH`
        - for instance, add the following line to `~/.zshrc`: `export PATH="$HOMEBREW_PREFIX/opt/make/libexec/gnubin:$PATH"`

```sh
git clone https://github.com/YOUR-GITHUB-USERNAME/intent-agent.git   # your fork
cd intent-agent
git remote add upstream https://github.com/indykite/intent-agent.git

make install_dev     # verifies uv.lock is current and installs dependencies
make test            # runs the unit tests
```

Unit tests and the integration tests need no credentials or `.env` at all.

### Make targets

Although python execution is done through `uv`, all commands are defined in the `Makefile`.

| Command                          | Purpose                                               |
|----------------------------------|-------------------------------------------------------|
| `make build_integration`         | All checks: lockfile, lint, unit + integration tests  |
| `make fix`                       | Re-format files: imports, code style (Changes files)  |
| `make test`                      | Run unit tests                                        |
| `make test_coverage`             | Run code coverage and integration tests               |
| `make test_coverage_report_text` | Report code coverage as text in cli                   |
| `make test_coverage_report_html` | Report code coverage as an index.html file in htmlcov |
| `make evals`                     | Run the golden eval suite against Gemini              |

### Running the agent locally

`.env.example` leaves several settings blank that `Settings` requires at startup - set
`GEMINI_API_KEY`, `INDYKITE_BASE_URL`, `IK_APP_AGENT_KEY`, `INTROSPECT_URL`,
`INTROSPECT_CLIENT_ID` and `INTROSPECT_CLIENT_SECRET` before starting (the service refuses to
start with any of them blank; the [README](./README.md#configuration) explains each one).

For a purely local run, `INTROSPECT_URL` can point at the bundled mock introspection service, which
accepts any non-empty token. `make start_app` and `make start_mock_introspect_service` both run in
the foreground, so start each in its own terminal:

```sh
cp .env.example .env   # then fill in the required settings listed above

# terminal 1
make start_mock_introspect_service   # manual_testing/mock_introspect.py on port 9001

# terminal 2 - pass INTROSPECT_URL as a make variable (or set it in .env): the Makefile includes
# .env, and a makefile assignment overrides an exported shell variable, so `export` is ignored
make start_app INTROSPECT_URL=http://127.0.0.1:9001/introspect   # uv run python -m indykite.intent_agent

# terminal 3, once both are up
manual_testing/call_agent.sh         # sends a sample SendMessage request
```

This starts the A2A JSON-RPC server on `INTENT_PORT` (default `6003`) and a health endpoint on
`HEALTH_PORT` (default `8001`); the README's [quick start](./README.md#quick-start) has the health
check and a hand-written request. The schema and policies still come from the IndyKite
environment named by `INDYKITE_BASE_URL`; if it is unreachable the service starts anyway and
answers every request with `schema_unavailable_error` until the fetch succeeds. The walkthroughs in
[`docs/testing.md`](./docs/testing.md) build on this setup.

### With Docker

```sh
docker buildx build --platform linux/amd64 --load -t intent-agent:local .
docker run --platform linux/amd64 --rm -p 6003:6003 -p 8001:8001 --env-file .env \
  -e AUDIT_FILE_STORAGE_PATH=/tmp/audit intent-agent:local
```

The audit directory override is needed because the image runs as a non-root user and `/app` is
not writable, so `.env.example`'s relative `./.private/audit` cannot be created and the container
exits at startup (`.env.docker-compose.example` already uses `/tmp/audit` for the same reason).
The base images are pinned to multi-arch digests; the command above forces `linux/amd64`, so on
Apple Silicon it runs under emulation (drop `--platform` for a native arm64 image). The
`make build_docker_image` target builds both `linux/amd64` and `linux/arm64` with `--load`, which
only works with Docker's containerd image store; on the default store it fails with "docker
exporter does not currently support exporting manifest lists". `make start_docker_app` runs that
image with `.env` as-is, so set `AUDIT_FILE_STORAGE_PATH=/tmp/audit` in `.env` first.

### Multi-instance, with docker-compose

`docker-compose.yml` runs two agent instances behind an nginx load balancer, backed by a
shared Redis - the real deployment topology this service is designed for (shared schema
cache, no per-instance refresh collisions). A single `uv run` process never exercises this, so
run it before merging anything touching `schema.py`, cache/TTL behaviour or the startup path.

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
curl -X POST http://localhost:6003/ ...   # (the README's quick-start request), a few times
docker logs intent-agent-intent-agent-1-1 | grep -c "POST /"
docker logs intent-agent-intent-agent-2-1 | grep -c "POST /"   # both should be nonzero
```

### Pre-commit hooks

The same checks run on every pull request; installing the hooks catches failures before you push.
They go beyond the Python side that `make fix` and `make build_integration` cover: YAML,
Markdown, shell and GitHub Actions linting, Dockerfile and Helm checks, secret scanning with
gitleaks and trivy, checkov, and the commit-message rules below. The tools they call are listed on
the `brew install` line at the top of `.pre-commit-config.yaml`. The commit-message hook reads the
shared commitlint config from a checkout of the public `indykite/.github` repo next to this one:

```sh
git clone https://github.com/indykite/.github ../.github
pre-commit install
pre-commit install -t commit-msg
```

## Making changes

1. Fork the repository and branch off `master`.
2. Make your change, keeping it focused: one fix or feature per PR.
3. Add or update tests under `test/`. Integration tests that drive the real A2A endpoint live under
   `integration/`. A change to skill content, schema handling or the validation loop should come
   with a golden fixture under `evals/` when it changes what the agent answers.
4. Run the checks locally before pushing:

   ```sh
   make fix                # ruff format + ruff check --fix
   make build_integration  # lockfile check + ruff + unit and integration tests
   ```

5. If your change touches `evals/`, the skill, or the pipeline modules, run `make evals` if you
   have a Gemini key, and paste the per-category summary in the PR. Without a key, say so. The
   eval-suite workflow needs the project's Gemini key, which is never given to code from a fork,
   so its check is skipped on your PR (skipped counts as passing) and the suite runs on `master`
   once the change is merged. For pull requests from branches of this repository the suite is a
   required check whenever the PR touches the service code, the dependency pins, `evals/` or the
   eval workflow itself (`.github/workflows/eval-suite.yaml`). For a pull request from a fork
   that touches those files, a maintainer copies your reviewed commit
   to a branch of this repository, runs the suite there before merging, and posts the result.

## Pull requests

- Keep one fix or feature per PR; reviewers should not have to evaluate unrelated changes together.
- Reference the accepted issue in the PR description with `Fixes #N`, so it closes when the PR is
  merged. A [trivial change](#what-counts-as-trivial) or a bot PR doesn't need one; say so instead.
- Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/),
  e.g. `feat: accept policy-granted action names` or `fix: drop blank action instead of retrying`.
  Releases and the changelog are generated from them. The shared commitlint rules additionally
  require: a type from `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`,
  `style`, `test`; a subject of at most 50 characters (72 for the whole header and for each body
  line) with no trailing full stop; no quotes or backticks anywhere in the message; and, if a scope
  is used, one of `api`, `authn`, `authz`, `deps`, `docs`, `schema`, `test` (plus a few
  IndyKite-internal ones). No ticket reference is needed in the commit message: link the issue
  your PR addresses in its description (`Fixes #123`). IndyKite maintainers reference their
  internal ticket in the branch name instead (`ENG-1234-short-description`).
- The body should answer: what does this change do, why is it needed, and how was it tested?
- Breaking changes to the request or response shape, the audit record, or configuration names must
  be called out explicitly in the PR title and body.
- Don't reformat or re-align lines your change doesn't need. Large mixed-purpose diffs are hard to
  review and clash with other people's work.
- By submitting a PR you confirm you have the right to contribute the content under this repo's
  [Apache-2.0 LICENSE](./LICENSE).

## Reporting bugs

Use the **Bug report** form under "New issue"; it asks for everything below. A good bug report
contains:

1. The prompt, the `subject_type` and `subject_id` sent, the schema the agent was serving and the
   access policies active for that `subject_type` (or the `evals/schema/` data model's
   `<model>_schema.json` and `<model>_policies.json`) - every answer comes from those policies.
2. The response you got, the response you expected and why, and the relevant log lines or audit
   record if you have them.
3. The configuration that matters (model name, timeouts, retry settings) and the version or commit.

Please redact tokens, API keys and any real customer data before posting.

## Code of conduct

This project follows the [Contributor Covenant](./CODE_OF_CONDUCT.md). In short: assume good
faith, critique the work rather than the person, be specific in reviews, and welcome newcomers.
Harassment, discriminatory language, doxxing or sustained bad-faith argument are grounds for a ban.

To report a conduct issue privately, use the contact address in
[`responsible_disclosure.md`](./responsible_disclosure.md); reports are kept confidential among the
maintainers handling the case, and good-faith reports will never be held against you.
