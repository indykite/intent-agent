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
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- else }}
app.kubernetes.io/version: {{ .Values.image.tag | quote }}
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
Fail fast if app.cacheBackend is "redis" but neither an external Redis (app.redisUrlSecretRef) nor
the chart-managed Redis (redis.enabled) is configured - mirrors config.py's Settings
._require_redis_url_for_redis_backend, which rejects this at app startup. When redisUrlSecretRef
IS used, name and key are still required together, same as before.
Also fails the reverse mismatch: redisUrlSecretRef set but cacheBackend left at "memory" would
inject REDIS_URL while Settings.cache_backend stays "memory", silently defeating the shared cache.
Note: redis.enabled=true with cacheBackend="memory" is deliberately NOT rejected here, unlike the
redisUrlSecretRef case above - it's treated as a legitimate staging state (e.g. provisioning the
chart-managed Redis before cutting the app over to it), even though it also injects an unused
REDIS_URL. If that asymmetry ever needs closing, fold `((.Values.redis).enabled)` into the "else
if" condition below.
*/}}
{{- define "service.validateCacheBackend" -}}
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
Fail fast on chart-managed Redis (redis.enabled) misconfigurations service.redisHost can't safely
paper over:
  - redis.architecture other than "standalone" - this chart only derives/tests the primary
    "<fullname>-master" Service name; replica-aware or sentinel-based discovery is out of scope.
    (Both "standalone" and "replication" architectures name the master Service the same way, so
    this restriction is about scope, not about service.redisHost's naming logic being wrong under
    "replication".)
  - redis.fullnameOverride or redis.nameOverride set - service.redisHost replicates the bitnami
    subchart's default naming algorithm and does not account for these overrides; supporting them
    would mean duplicating more of that logic, so for now this just fails loudly instead of
    silently deriving a wrong hostname.
  - redis.auth.enabled with no app.redisUrlSecretRef.name - the derived REDIS_URL
    (service.redisHost) never embeds a password, so a password-protected chart-managed Redis is
    unreachable via that derived URL; either set redis.auth.enabled: false, or point
    app.redisUrlSecretRef at a secret containing a fully-composed authenticated DSN instead of
    relying on derivation.
*/}}
{{- define "service.validateRedisConfig" -}}
{{- if ((.Values.redis).enabled) }}
{{- if ne (((.Values.redis).architecture) | default "standalone") "standalone" }}
{{- fail (printf "redis.architecture %q is not supported - only \"standalone\" is, when redis.enabled is true" ((.Values.redis).architecture)) }}
{{- end }}
{{- if or ((.Values.redis).fullnameOverride) ((.Values.redis).nameOverride) }}
{{- fail "redis.fullnameOverride/redis.nameOverride are not supported alongside redis.enabled - service.redisHost cannot derive the correct hostname when these are set" }}
{{- end }}
{{- if and (not .Values.app.redisUrlSecretRef.name) (((.Values.redis).auth).enabled) }}
{{- fail "redis.auth.enabled is true but app.redisUrlSecretRef.name is not set - the chart-managed Redis's derived URL never carries a password, so set redis.auth.enabled: false, or supply app.redisUrlSecretRef pointing at a full authenticated REDIS_URL" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Derive the chart-managed Redis's in-cluster hostname without depending on the bitnami/redis
subchart's own named templates (coupling to "common.names.fullname" et al. across bitnami chart
version bumps is fragile) - replicates that algorithm (confirmed against
bitnami/charts@redis/23.1.1's common/templates/_names.tpl and redis/templates/master/service.yaml)
for the specific case service.validateRedisConfig enforces: architecture "standalone", no
redis.fullnameOverride/nameOverride set.
*/}}
{{- define "service.redisHost" -}}
{{- $releaseName := regexReplaceAll "(-?[^a-z0-9-])+-?" (lower .Release.Name) "-" -}}
{{- if contains "redis" $releaseName -}}
{{- printf "%s-master" $releaseName -}}
{{- else -}}
{{- printf "%s-redis-master" $releaseName -}}
{{- end -}}
{{- end -}}

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
