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
{{- range $k, $v := . }}
{{- if ne $k "owner" }}
{{ $k }}: {{ $v | quote }}
{{- end }}
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
Fail fast if app.cacheBackend is "redis" but no redisUrlSecretRef is configured - mirrors
config.py's Settings._require_redis_url_for_redis_backend, which rejects this at app startup.
*/}}
{{- define "service.validateCacheBackend" -}}
{{- if and (eq .Values.app.cacheBackend "redis") (not .Values.app.redisUrlSecretRef.name) }}
{{- fail "app.redisUrlSecretRef.name is required when app.cacheBackend is \"redis\"" }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "service.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (printf "%s-sa" (include "service.fullname" .)) .Values.serviceAccount.name }}
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
  name: {{ include "service.fullname" . }}-allow
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
