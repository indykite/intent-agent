#!/usr/bin/env bash
set -euo pipefail

# Get list of changed files relative to main or HEAD~1 depending on CI or local
BASE_REF="${BASE_REF:-$(git rev-parse --abbrev-ref origin/HEAD 2>/dev/null || echo 'origin/master')}"

# Ensure BASE_REF is available (critical in shallow GitHub Actions checkouts)
if ! git rev-parse --verify "${BASE_REF}" >/dev/null 2>&1; then
    REMOTE="${BASE_REF%%/*}"
    BRANCH="${BASE_REF#*/}"
    git fetch --depth=1 "${REMOTE}" "${BRANCH}" 2>/dev/null || true
fi

# Include: committed changes (BASE_REF...HEAD), staged changes (--staged), unstaged changes (HEAD)
ALL_DIFF_FILES=$(printf '%s\n' \
    "$(git diff --name-only "${BASE_REF}"...HEAD || true)" \
    "$(git diff --name-only --staged || true)" \
    "$(git diff --name-only HEAD || true)" |
    grep -v '^$' | sort -u || true)

# Discover all Helm charts in the repo by presence of Chart.yaml
# shellcheck disable=SC2207
CHART_DIRS=($(find . -name "Chart.yaml" | sed 's|^\./||; s|/Chart.yaml$||' | sort))

if [[ ${#CHART_DIRS[@]} -eq 0 ]]; then
    echo "✅ No charts found in repository."
    exit 0
fi

echo "discovered charts:"
printf '  %s\n' "${CHART_DIRS[@]}"

declare -a CHART_FILES_CHANGED=()
declare -a VERSION_CHANGED=()

for chart_dir in "${CHART_DIRS[@]}"; do
    # Files relevant to this chart: Chart.yaml, .helmignore, values.yaml, values.schema.json, templates/
    chart_relevant=$(printf '%s\n' "${ALL_DIFF_FILES}" | grep -E "^${chart_dir}/(Chart\.yaml|\.helmignore|values\.yaml|values\.schema\.json|templates/)" || true)

    if [[ -z "${chart_relevant}" ]]; then
        continue
    fi

    CHART_FILES_CHANGED+=("${chart_dir}")

    # Check if Chart.yaml had a version bump
    chart_yaml="${chart_dir}/Chart.yaml"
    if git diff "${BASE_REF}"...HEAD -- "${chart_yaml}" | grep -q '^+version:' ||
        git diff --staged -- "${chart_yaml}" | grep -q '^+version:' ||
        git diff HEAD -- "${chart_yaml}" | grep -q '^+version:'; then
        VERSION_CHANGED+=("${chart_dir}")
    fi
done

[[ ${#CHART_FILES_CHANGED[@]} -gt 0 ]] && echo "detected changed chart dirs: ${CHART_FILES_CHANGED[*]}"
[[ ${#VERSION_CHANGED[@]} -gt 0 ]] && echo "updated versions in: ${VERSION_CHANGED[*]}"

if [[ ${#CHART_FILES_CHANGED[@]} -eq 0 ]]; then
    echo "✅ No changes in chart files detected."
    exit 0
fi

# Find charts with changes but WITHOUT version bump (set difference)
# shellcheck disable=SC2207
MISSING_VERSION_BUMP=($(comm -23 \
    <(printf '%s\n' "${CHART_FILES_CHANGED[@]+"${CHART_FILES_CHANGED[@]}"}" | sort || true) \
    <(printf '%s\n' "${VERSION_CHANGED[@]+"${VERSION_CHANGED[@]}"}" | sort || true)))

if (("${#MISSING_VERSION_BUMP[@]}" > 0)); then
    echo "❌ Chart file(s) changed but Chart.yaml version not updated! Bump the version for:"
    printf '%s\n' "${MISSING_VERSION_BUMP[@]+"${MISSING_VERSION_BUMP[@]}"}"
    exit 1
else
    echo "✅ Helm chart version check passed."
fi
