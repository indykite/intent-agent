# Maintaining the Intent Agent

For maintainers (write access or above): how issues are triaged, how pull requests from forks are
reviewed and evaluated, and what to check before a release. Contributors' side of the same flow is
in [`CONTRIBUTING.md`](../CONTRIBUTING.md#from-issue-to-merge).

## Triage

New issues arrive through the **Bug report** form, labelled `bug` and `needs-triage`. Ideas and
questions go to Discussions - there is no feature request form, and blank issues are limited to
maintainers. When an idea is accepted, open the issue yourself from the discussion ("Create issue
from discussion") so the conversation stays linked, and label it `enhancement` and `accepted`
(plus `help wanted` if an outside contributor may take it). Say so in the discussion.

Replace `needs-triage` with one of:

| Label | When |
| --- | --- |
| `accepted` | The change is agreed. A pull request may reference it with `Fixes #N`. |
| `help wanted` (with `accepted`) | An outside contributor may take it. Add `good first issue` if it is small and self-contained. |
| `needs-info` | You asked the reporter a question. Triage again when the answer comes; close the issue if none comes. |
| `wontfix`, `duplicate`, `invalid` | Close it, with a short reason. |

Work IndyKite schedules itself is tracked in IndyKite's internal tracker: create the ticket by
hand, link it to the GitHub issue, and keep internal details out of the public issue. The branch
is named after the ticket (`ENG-1234-short-description`).

A security-sensitive report must not stay public: don't discuss the details in the issue, ask the
reporter to use the private routes in [`SECURITY.md`](../SECURITY.md), and close or delete the
issue.

## Reviewing a pull request

Before approving, check the description against the
[pull request template](../.github/PULL_REQUEST_TEMPLATE.md):

- **Related issue:** `Fixes #N` for an `accepted` issue, or "trivial" when the change qualifies
  ([`CONTRIBUTING.md` § What counts as trivial](../CONTRIBUTING.md#what-counts-as-trivial)). CI,
  Docker and chart changes and cross-module refactors are never trivial.
- **"Effect on requests, responses and the audit record" and "How it was tested"** are filled in,
  with "None" and a reason when nothing changes. The checks match what the change touches
  ([`docs/testing.md` § In a pull request](testing.md#in-a-pull-request)).
- **The ticked boxes match the diff.** If anything under `/.github/` (workflows included),
  `/chart/` or another shared tooling file changes, "Shared tooling and CI configuration untouched"
  must be unticked and the change named in the description; CODEOWNERS asks its owners to review.

## Pull requests from forks

GitHub never gives repository secrets to a fork's workflows, so on a fork pull request the build,
tests and SAST Linting run, and the `golden-evals` check is skipped (skipped counts as passing).

- **Approve the workflow runs** when they wait for approval: on the pull request, **Files
  changed** → **Awaiting approval** → **Approve workflows to run**. Read the diff first.
- **Run the eval suite yourself** before merging any fork pull request that changes `indykite/`,
  `evals/`, `pyproject.toml`, `uv.lock` or `.github/workflows/eval-suite.yaml`. The check being
  skipped means GitHub won't stop the merge, so this step is yours:
    1. Review the diff. The suite runs the pull request's code with the Gemini key, including its
       version of `.github/workflows/eval-suite.yaml`.
    2. Copy the pull request's latest commit, unchanged, to a branch of this repository:

       ```sh
       git fetch origin pull/<N>/head:fork-pr-<N>
       git push origin fork-pr-<N>
       ```

    3. Run it: **Actions** → **Golden eval suite** → **Run workflow** on `fork-pr-<N>`.
    4. Post the result on the pull request. Merge only if all gates pass.
    5. Delete the `fork-pr-<N>` branch.

  If the contributor pushes again, the copy is out of date: compare the commit on the pull request
  with the one you copied and repeat from step 1.
- Never add a `pull_request_target` workflow or any other way of giving secrets to fork code.

## Merging

Squash-merge. The pull request's title becomes the commit subject on `master`, and release-please
derives the next version from its Conventional Commits type (`fix:` → patch, `feat:` → minor), so
check the title before merging. The issue named in `Fixes #N` closes automatically.

## After merge

A merge that changes `indykite/` (apart from the release version bump), `evals/`,
`pyproject.toml`, `uv.lock` or the eval workflow runs the eval suite on `master`. If that run
fails, the regression is already on `master` (and deployed to dev): fix it forward with a new pull
request, or revert the squash commit with `git revert <sha>` in a pull request.

## Releases

release-please keeps a release pull request open. Don't merge it while the latest eval run on
`master` is red: check **Actions** → **Golden eval suite**, filtered on the `master` branch.

## Labels used by automation

Don't delete or rename these; the bots look them up:

| Label | Used by |
| --- | --- |
| `autorelease: pending`, `autorelease: tagged` | release-please, to find its release pull request and tag the release |
| `dependencies`, `renovate`, `SemVer Major`, `SemVer Minor`, `SemVer Patch` | Renovate, on dependency pull requests |
| `ci`, `no-deploy` | Renovate, on CI configuration updates |
