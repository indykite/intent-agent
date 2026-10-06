# Pull request

<!--
Thanks for opening a PR. Please fill in the sections below - reviewers use them
to triage and to write the eventual changelog entry. Empty PRs without context
get bounced.

If anything here doesn't apply (e.g. doc-only change), say so explicitly rather
than deleting the section.
-->

## What this PR does

<!-- One sentence. Lead with the verb: "Add ...", "Fix ...", "Tighten ...".
     Name the affected stage where relevant (skill, subject binding, schema client,
     executor, audit). -->

## Effect on requests, responses and the audit record

<!-- Required for any change to the extraction pipeline. Which prompts or
     evaluation shapes are affected, and how does the response change?
     Call out explicitly:
     - any change to the request or response shape, or to an error reason
     - any change to the audit record's fields or values
     - any new or renamed configuration (Settings field / environment variable) -->

## How it was tested

<!-- Be specific. List the tests added or updated, and paste relevant output.
     If the change touches the skill, the schema handling or the validation
     loop, run the golden eval suite and paste its per-category summary:
     `make evals` (needs a Gemini key). -->

- [ ] `make fix` then `make build_integration` pass
- [ ] Tests added/updated for the new behaviour (positive case + edge case)
- [ ] Agent started locally and verified end-to-end (see [`CONTRIBUTING.md` § Running the agent locally](https://github.com/indykite/intent-agent/blob/master/CONTRIBUTING.md#running-the-agent-locally))
- [ ] `make evals` run (if pipeline-affecting) - result:

## Known gaps or follow-ups

<!-- Anything intentionally out of scope: missing edge cases, unsupported
     prompt forms, features deferred to a follow-up PR.
     Better to surface than to surprise. -->

## Checklist

- [ ] One fix or feature per PR - no unrelated changes bundled in.
- [ ] PR title follows the existing convention (e.g. `feat: accept policy-granted action names`, `fix: drop blank action instead of retrying`) - see [`CONTRIBUTING.md` § Pull requests](https://github.com/indykite/intent-agent/blob/master/CONTRIBUTING.md#pull-requests).
- [ ] No new entries in `pyproject.toml` dependencies unless agreed in an issue first; if there are, `uv.lock` is updated.
- [ ] Shared tooling and CI configuration untouched - see [`CONTRIBUTING.md` § Ground rules](https://github.com/indykite/intent-agent/blob/master/CONTRIBUTING.md#ground-rules); if one of those files genuinely needs to change, it was agreed with its owners first.
- [ ] Breaking changes to the request/response shape, the audit record or configuration names are called out explicitly in the PR title and body.
- [ ] No secrets, internal URLs, or personal data committed.
- [ ] I have the right to contribute this content under the repo's [LICENSE](https://github.com/indykite/intent-agent/blob/master/LICENSE).
