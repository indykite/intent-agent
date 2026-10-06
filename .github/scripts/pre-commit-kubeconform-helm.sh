#!/usr/bin/env bash
# Validate all Helm charts in the repo with kubeconform.
# Charts are discovered by locating Chart.yaml files (subcharts inside charts/ are excluded).
# Custom schemas are loaded from https://github.com/datreeio/CRDs-catalog
# https://github.com/yannh/kubeconform#overriding-schemas-location
#
# Usage: pre-commit-kubeconform-helm.sh [KUBECONFORM_ARGS ...]
#   Any extra arguments are forwarded directly to kubeconform, e.g.:
#     -skip apiextensions.k8s.io/v1/CustomResourceDefinition
#     -skip monitoring.coreos.com/v1/ServiceMonitor
set -uo pipefail

failed=0

while IFS= read -r chart_yaml; do
    chart_dir="$(dirname "${chart_yaml}")"
    echo "Validating chart: ${chart_dir}"
    values_args=()
    [[ -f "${chart_dir}/linter_values.yaml" ]] && values_args=(--values "${chart_dir}/linter_values.yaml")
    helm template dummy "${chart_dir}" ${values_args[@]+"${values_args[@]}"} |
        kubeconform \
            -summary \
            -strict \
            -verbose \
            -schema-location default \
            -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json' \
            "$@" ||
        failed=1
done < <(find . -name "Chart.yaml" -not -path "*/.git/*" -not -path "*/charts/*" || true)

exit "${failed}"
