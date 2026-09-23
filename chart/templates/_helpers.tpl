{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to 63 chars
(by the DNS naming spec), but container names will be longer (as they contain suffixes)
If release name contains chart name it will be used as a full name.
*/}}
{{- define "service.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "service.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "service.selectorLabels" -}}
app.kubernetes.io/name: {{ include "service.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name | trunc 63 | trimSuffix "-" }}
app: {{ .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "service.labels" -}}
helm.sh/chart: {{ include "service.chart" . }}
{{ include "service.selectorLabels" . }}
{{- /* An explicit image.tag override (values-dev.yaml, values-rc.yaml, a PR/master package's
     own suffixed tag) identifies the actually-deployed image more precisely than the chart's own
     version - prefer it. Otherwise fall back to Chart.AppVersion, not Chart.Version: Version
     carries whatever suffix the packaging step applied (`helm package --version`'s PR/dev suffix,
     or chart-publish's own "-chart" suffix on the Docker Hub package), none of which is ever part
     of the image's tag, while AppVersion always stays the plain release version - the same
     fallback templates/deployment.yaml uses for the image tag/pod label, so this label never
     reports a version the image was never tagged with. */}}
{{- if .Values.image.tag }}
app.kubernetes.io/version: {{ .Values.image.tag | quote }}
{{- else }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
owner: ai-ml
{{- with .Values.labels }}
{{- /* Drop keys already emitted above - a custom label reusing one (e.g. "app") would either
     duplicate a YAML key or, worse, override the value pods are selected by. */ -}}
{{- with (omit . "helm.sh/chart" "app.kubernetes.io/name" "app.kubernetes.io/instance" "app" "app.kubernetes.io/version" "app.kubernetes.io/managed-by" "owner") }}
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Fail fast if app.intentPort and app.healthPort collide - they back two distinct
Service ports/probes and silently sharing one port hides misconfiguration.
*/}}
{{- define "service.validatePorts" -}}
{{- if eq (.Values.app.intentPort | int) (.Values.app.healthPort | int) }}
{{- fail (printf "app.intentPort and app.healthPort must differ (both set to %d)" (.Values.app.intentPort | int)) }}
{{- end }}
{{- end }}

{{/*
Fail fast if app.llmTimeoutSeconds, app.schemaFetchTimeoutSeconds, app.policyFetchTimeoutSeconds
or app.redisSocketTimeoutSeconds exceeds app.requestTimeoutSeconds - mirrors config.py's Settings
._reject_llm_timeout_over_request_timeout and ._reject_fetch_timeouts_over_request_timeout, which
reject the same combinations at app startup. Checking it here too (review) catches a bad
values.yaml at `helm install`/`helm template` time instead of a pod that starts, fails Settings
validation, and crash-loops.
*/}}
{{- define "service.validateTimeouts" -}}
{{- if gt (.Values.app.llmTimeoutSeconds | float64) (.Values.app.requestTimeoutSeconds | float64) }}
{{- fail (printf "app.llmTimeoutSeconds (%v) must not exceed app.requestTimeoutSeconds (%v)" .Values.app.llmTimeoutSeconds .Values.app.requestTimeoutSeconds) }}
{{- end }}
{{- if gt (.Values.app.schemaFetchTimeoutSeconds | float64) (.Values.app.requestTimeoutSeconds | float64) }}
{{- fail (printf "app.schemaFetchTimeoutSeconds (%v) must not exceed app.requestTimeoutSeconds (%v)" .Values.app.schemaFetchTimeoutSeconds .Values.app.requestTimeoutSeconds) }}
{{- end }}
{{- if gt (.Values.app.policyFetchTimeoutSeconds | float64) (.Values.app.requestTimeoutSeconds | float64) }}
{{- fail (printf "app.policyFetchTimeoutSeconds (%v) must not exceed app.requestTimeoutSeconds (%v)" .Values.app.policyFetchTimeoutSeconds .Values.app.requestTimeoutSeconds) }}
{{- end }}
{{- if gt (.Values.app.redisSocketTimeoutSeconds | float64) (.Values.app.requestTimeoutSeconds | float64) }}
{{- fail (printf "app.redisSocketTimeoutSeconds (%v) must not exceed app.requestTimeoutSeconds (%v)" .Values.app.redisSocketTimeoutSeconds .Values.app.requestTimeoutSeconds) }}
{{- end }}
{{- if ge (.Values.app.redisQueueDequeueBlockSeconds | float64) (.Values.app.redisSocketTimeoutSeconds | float64) }}
{{- fail (printf "app.redisQueueDequeueBlockSeconds (%v) must be less than app.redisSocketTimeoutSeconds (%v)" .Values.app.redisQueueDequeueBlockSeconds .Values.app.redisSocketTimeoutSeconds) }}
{{- end }}
{{- end }}

{{/*
Fail fast if app.cacheBackend is "redis" but neither an external Redis (app.redisUrlSecretRef) nor
the chart-managed Redis (redis.enabled) is configured - mirrors config.py's Settings
._require_redis_url_for_redis_backend, which rejects this at app startup. name and key are
required together, in both directions, regardless of cacheBackend/redis.enabled - key set with
name blank would otherwise silently fall through to the redis.enabled-derived REDIS_URL (or, with
redis.enabled also false, get a less specific error further down), silently ignoring the secret
the caller thought they were pointing at.
Also fails the reverse mismatch: redisUrlSecretRef set but cacheBackend left at "memory" would
inject REDIS_URL while Settings.cache_backend stays "memory", silently defeating the shared cache.
Note: redis.enabled=true with cacheBackend="memory" is deliberately NOT rejected here, unlike the
redisUrlSecretRef case above - it's treated as a legitimate staging state (e.g. provisioning the
chart-managed Redis before cutting the app over to it), even though it also injects an unused
REDIS_URL. If that asymmetry ever needs closing, fold `((.Values.redis).enabled)` into the "else
if" condition below.
*/}}
{{- define "service.validateCacheBackend" -}}
{{- if and .Values.app.redisUrlSecretRef.key (not .Values.app.redisUrlSecretRef.name) }}
{{- fail "app.redisUrlSecretRef.name is required when app.redisUrlSecretRef.key is set" }}
{{- end }}
{{- if eq .Values.app.cacheBackend "redis" }}
{{- if .Values.app.redisUrlSecretRef.name }}
{{- if not .Values.app.redisUrlSecretRef.key }}
{{- fail "app.redisUrlSecretRef.key is required when app.redisUrlSecretRef.name is set" }}
{{- end }}
{{- else if not ((.Values.redis).enabled) }}
{{- fail "app.cacheBackend is \"redis\" but neither app.redisUrlSecretRef.name nor redis.enabled is set" }}
{{- end }}
{{- else if .Values.app.redisUrlSecretRef.name }}
{{- fail "app.redisUrlSecretRef.name is set but app.cacheBackend is not \"redis\" - set app.cacheBackend: \"redis\" too, or clear redisUrlSecretRef" }}
{{- end }}
{{- end }}

{{/*
Fail fast on the A2A task store's Redis config - mirrors validateCacheBackend above exactly,
same shape (app.a2aTaskStoreBackend/app.a2aTaskStoreUrlSecretRef vs. app.cacheBackend/
app.redisUrlSecretRef), since RedisTaskStore now has the same memory-mode fallback the schema
cache always had (the a2a-sdk's own InMemoryTaskStore) - app.a2aTaskStoreUrlSecretRef is only
required when app.a2aTaskStoreBackend is "redis", not unconditionally.
*/}}
{{- define "service.validateTaskStoreConfig" -}}
{{- if and .Values.app.a2aTaskStoreUrlSecretRef.key (not .Values.app.a2aTaskStoreUrlSecretRef.name) }}
{{- fail "app.a2aTaskStoreUrlSecretRef.name is required when app.a2aTaskStoreUrlSecretRef.key is set" }}
{{- end }}
{{- if eq .Values.app.a2aTaskStoreBackend "redis" }}
{{- if .Values.app.a2aTaskStoreUrlSecretRef.name }}
{{- if not .Values.app.a2aTaskStoreUrlSecretRef.key }}
{{- fail "app.a2aTaskStoreUrlSecretRef.key is required when app.a2aTaskStoreUrlSecretRef.name is set" }}
{{- end }}
{{- else if not ((.Values.redis).enabled) }}
{{- fail "app.a2aTaskStoreBackend is \"redis\" but neither app.a2aTaskStoreUrlSecretRef.name nor redis.enabled is set" }}
{{- end }}
{{- else if .Values.app.a2aTaskStoreUrlSecretRef.name }}
{{- fail "app.a2aTaskStoreUrlSecretRef.name is set but app.a2aTaskStoreBackend is not \"redis\" - set app.a2aTaskStoreBackend: \"redis\" too, or clear a2aTaskStoreUrlSecretRef" }}
{{- end }}
{{- end }}

{{/*
Fail fast on chart-managed Redis (redis.enabled) misconfigurations service.redisHost can't safely
paper over:
  - app.redisUrlSecretRef.name/key set alongside redis.enabled - the chart-managed Redis and an
    externally managed one are mutually exclusive, not layered: provisioning a Redis nobody points
    at (silently ignored in favor of the external secret) is exactly the kind of misconfiguration
    this file's validators exist to catch elsewhere. Clear redisUrlSecretRef to use the
    chart-managed Redis, or set redis.enabled: false to use the external one.
  - redis.architecture other than "standalone" - this chart only derives/tests the primary
    "<fullname>-master" Service name; replica-aware or sentinel-based discovery is out of scope.
    (Both "standalone" and "replication" architectures name the master Service the same way, so
    this restriction is about scope, not about service.redisHost's naming logic being wrong under
    "replication".)
  - redis.fullnameOverride or redis.nameOverride set - service.redisHost replicates the bitnami
    subchart's default naming algorithm and does not account for these overrides; supporting them
    would mean duplicating more of that logic, so for now this just fails loudly instead of
    silently deriving a wrong hostname.
  - redis.auth.enabled - the derived REDIS_URL (service.redisHost) never embeds a password, so a
    password-protected chart-managed Redis is unreachable via that derived URL. (There's no
    "point app.redisUrlSecretRef at an authenticated DSN instead" escape hatch here, unlike the
    other validators in this file - redisUrlSecretRef is required empty by the mutual-exclusivity
    check above whenever redis.enabled is true, so the only fix is redis.auth.enabled: false.)
  - a release name long enough that the derived "<fullname>-master" hostname exceeds Kubernetes'
    63-char DNS label limit - the bitnami subchart's own master/service.yaml template appends
    "-master" to its (already 63-char-truncated) fullname with no further truncation, so past this
    length its generated Service name is itself invalid and `helm install`/`kubectl apply` would
    reject it; failing here instead points at the actual cause (the release name) rather than a
    confusing error surfaced from deep inside the vendored subchart's own Service manifest.
*/}}
{{- define "service.validateRedisConfig" -}}
{{- if ((.Values.redis).enabled) }}
{{- if or .Values.app.redisUrlSecretRef.name .Values.app.redisUrlSecretRef.key }}
{{- fail "app.redisUrlSecretRef.name/key must be empty when redis.enabled is true - the chart-managed Redis and an external Redis can't both be configured; clear redisUrlSecretRef, or set redis.enabled: false to use the external secret instead" }}
{{- end }}
{{- if ne (((.Values.redis).architecture) | default "standalone") "standalone" }}
{{- fail (printf "redis.architecture %q is not supported - only \"standalone\" is, when redis.enabled is true" ((.Values.redis).architecture)) }}
{{- end }}
{{- if or ((.Values.redis).fullnameOverride) ((.Values.redis).nameOverride) }}
{{- fail "redis.fullnameOverride/redis.nameOverride are not supported alongside redis.enabled - service.redisHost cannot derive the correct hostname when these are set" }}
{{- end }}
{{- if ((.Values.redis).auth).enabled }}
{{- fail "redis.auth.enabled is true but the chart-managed Redis's derived REDIS_URL (service.redisHost) never carries a password - set redis.auth.enabled: false" }}
{{- end }}
{{- if gt (len (include "service.redisHost" .)) 63 }}
{{- fail (printf "the chart-managed Redis's derived hostname %q exceeds Kubernetes' 63-char DNS label limit - shorten the Helm release name" (include "service.redisHost" .)) }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Replicates the bitnami/redis subchart's own "common.names.fullname" algorithm (confirmed against
bitnami/charts@redis/28.0.12's common chart's templates/_names.tpl) bit-for-bit, INCLUDING its
63-char truncation - the same 63-char limit applies here, before service.redisHost appends
"-master", so the two must agree on where truncation happens for the derived hostname to actually
match the subchart's real Service name. Kept separate from service.redisHost so
service.validateRedisConfig can check the final "<fullname>-master" length independently.
*/}}
{{- define "service.redisFullname" -}}
{{- $releaseName := regexReplaceAll "(-?[^a-z0-9-])+-?" (lower .Release.Name) "-" -}}
{{- if contains "redis" $releaseName -}}
{{- $releaseName | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-redis" $releaseName | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/*
Derive the chart-managed Redis's in-cluster hostname without depending on the bitnami/redis
subchart's own named templates (coupling to "common.names.fullname" et al. across bitnami chart
version bumps is fragile) - replicates that algorithm (confirmed against
bitnami/charts@redis/28.0.12's common/templates/_names.tpl and redis/templates/master/service.yaml)
for the specific case service.validateRedisConfig enforces: architecture "standalone", no
redis.fullnameOverride/nameOverride set. Mirrors master/service.yaml's own
`printf "%s-master" (include "common.names.fullname" .)` exactly - including appending "-master"
AFTER truncation, with no further length limit - see service.validateRedisConfig's length check
for why that isn't silently allowed to produce an invalid/mismatched hostname instead.
*/}}
{{- define "service.redisHost" -}}
{{- printf "%s-master" (include "service.redisFullname" .) -}}
{{- end -}}

{{/*
Renders one Redis-DSN-shaped env var entry (12-space indented, matching deployment.yaml's env
list) - the shape REDIS_URL and A2A_TASK_STORE_URL both need: a Secret reference when the given
secretRef.name is set, the chart-managed Redis's derived DSN when redis.enabled, or nothing at
all otherwise. Usage: {{ include "service.redisUrlEnvVar" (list . "REDIS_URL" .Values.app.redisUrlSecretRef) }}
*/}}
{{- define "service.redisUrlEnvVar" -}}
{{- $root := index . 0 }}
{{- $envVarName := index . 1 }}
{{- $secretRef := index . 2 }}
{{- if $secretRef.name }}
            - name: {{ $envVarName }}
              valueFrom:
                secretKeyRef:
                  name: {{ $secretRef.name }}
                  key: {{ $secretRef.key }}
{{- else if (($root.Values.redis).enabled) }}
            - name: {{ $envVarName }}
              value: {{ printf "redis://%s:6379" (include "service.redisHost" $root) | quote }}
{{- end }}
{{- end }}

{{/*
Fail fast if autoscaling.minReplicas exceeds maxReplicas - the schema validates each field
independently, so this combination passes Helm validation but Kubernetes rejects the HPA.
*/}}
{{- define "service.validateAutoscaling" -}}
{{- if gt (.Values.autoscaling.minReplicas | int) (.Values.autoscaling.maxReplicas | int) }}
{{- fail (printf "autoscaling.minReplicas (%d) must not exceed autoscaling.maxReplicas (%d)" (.Values.autoscaling.minReplicas | int) (.Values.autoscaling.maxReplicas | int)) }}
{{- end }}
{{- end }}

{{/*
Suffix service.fullname while staying within Kubernetes' 63-char DNS-label limit - truncating
the base name AFTER appending the suffix (rather than the already-63-char fullname) would
silently drop the suffix chars, so reserve their room on the base first.
Usage: {{ include "service.suffixedName" (list . "-metrics") }}
*/}}
{{- define "service.suffixedName" -}}
{{- $suffix := index . 1 }}
{{- $base := include "service.fullname" (index . 0) | trunc (sub 63 (len $suffix) | int) | trimSuffix "-" }}
{{- printf "%s%s" $base $suffix }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "service.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "service.suffixedName" (list . "-sa")) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Common annotations
*/}}
{{- define "service.annotations" -}}
{{- toYaml .Values.annotations }}
{{- end }}

{{/*
AuthorizationPolicy base resource (metadata + selector + ALLOW action).
Used by authorizationpolicy.yaml to avoid duplicating the resource envelope.
*/}}
{{- define "service.authorizationPolicy" -}}
---
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: {{ include "service.suffixedName" (list . "-allow") }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "service.labels" . | nindent 4 }}
  annotations:
    argocd.argoproj.io/sync-wave: "1"
    {{- with .Values.annotations }}
    {{- include "service.annotations" $ | nindent 4 }}
    {{- end }}
spec:
  selector:
    matchLabels:
      {{- include "service.selectorLabels" . | nindent 6 }}
  action: ALLOW
{{- end }}
