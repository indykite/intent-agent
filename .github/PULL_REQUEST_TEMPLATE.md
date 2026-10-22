# Pull request

<!--
Thanks for opening a PR. Please fill in the sections below - reviewers use them
to triage and to write the eventual changelog entry. Empty PRs without context
get bounced.

Don't delete a section. If one doesn't apply (e.g. a doc-only change), write "None"
and why; only "Known gaps or follow-ups" may stay empty.
-->

## Related issue

<!-- `Fixes #N` for the accepted issue this PR addresses - it closes the issue on merge.
     A trivial change or a bot PR doesn't need one: write "trivial" (see
     CONTRIBUTING.md § What counts as trivial). CI, Docker and chart changes,
     cross-module refactors, prompts and the eval fixtures are never trivial. -->

## What this PR does

<!-- One sentence. Lead with the verb: "Add ...", "Fix ...", "Tighten ...".
     Name the affected stage where relevant (skill, subject binding, schema client,
     executor, audit). -->

## Effect on requests, responses and the audit record

<!-- Required for every PR. Which prompts or evaluation shapes are affected, and
     how does the response change? If nothing changes, write "None" and how you
     know (e.g. the existing tests pass unchanged). Call out explicitly:
     - any change to the request or response shape, or to an error reason
     - any change to the audit record's fields or values
     - any new or renamed configuration (Settings field / environment variable) -->

## How it was tested

<!-- Required for every PR. Be specific. List the tests added or updated, and
     paste relevant output. If the change touches the skill, the schema handling
     or the validation loop, run the golden eval suite and paste its per-category
     summary: `make evals` (needs a Gemini key). Which checks to run for which
     change: docs/testing.md § In a pull request. -->

- [ ] `make fix` then `make build_integration` pass
- [ ] Tests added/updated for the new behaviour (positive case + edge case), or none needed because no behaviour changes
- [ ] Agent started locally and verified end-to-end (see [`CONTRIBUTING.md` § Running the agent locally](https://github.com/indykite/intent-agent/blob/master/CONTRIBUTING.md#running-the-agent-locally) and [`docs/testing.md`](https://github.com/indykite/intent-agent/blob/master/docs/testing.md))
- [ ] `make evals` run (if pipeline-affecting) - result:

## Known gaps or follow-ups

<!-- Optional - leave empty if there are none. Anything intentionally out of
     scope: missing edge cases, unsupported prompt forms, features deferred to a
     follow-up PR. Better to surface than to surprise. -->

## Checklist

- [ ] References an accepted issue (`Fixes #N`), or the change is trivial - see [`CONTRIBUTING.md` § What counts as trivial](https://github.com/indykite/intent-agent/blob/master/CONTRIBUTING.md#what-counts-as-trivial).
- [ ] One fix or feature per PR - no unrelated changes bundled in.
- [ ] PR title follows the existing convention (e.g. `feat: accept policy-granted action names`, `fix: drop blank action instead of retrying`) - see [`CONTRIBUTING.md` § Pull requests](https://github.com/indykite/intent-agent/blob/master/CONTRIBUTING.md#pull-requests).
- [ ] No new entries in `pyproject.toml` dependencies unless agreed in an issue first; if there are, `uv.lock` is updated.
- [ ] Shared tooling and CI configuration untouched: nothing under `/.github/` (workflows included) or `/chart/`, and no change to the `Makefile`, `Dockerfile`, `.pre-commit-config.yaml`, the tool sections of `pyproject.toml` or the lint and scanner configs - see [`CONTRIBUTING.md` § Ground rules](https://github.com/indykite/intent-agent/blob/master/CONTRIBUTING.md#ground-rules).
  If one of them changes, leave this unticked, name the file and why in "What this PR does", and agree the change with its owners first.
- [ ] Breaking changes to the request/response shape, the audit record or configuration names are called out explicitly in the PR title and body.
- [ ] No secrets, internal URLs, or personal data committed.
- [ ] I have the right to contribute this content under the repo's [LICENSE](https://github.com/indykite/intent-agent/blob/master/LICENSE).
