---
name: Bug report
about: A wrong, missing or invalid extraction, or a service error
title: 'bug: '
labels: bug
---

<!--
Security-sensitive reports (authentication bypass, prompt injection that changes
the subject, leaked credentials) go to the private channel in
responsible_disclosure.md - please do NOT open a public issue for those.

Redact tokens, API keys and real customer data before posting.
-->

## Minimal reproduction

<!-- The prompt sent, the subject_type / subject_id in the message metadata, and
     the schema the agent was serving. If one of the data models under
     evals/schema/ reproduces it, name that instead of pasting a schema. -->

```json
```

## Actual response

<!-- The response you got: the {prompt, evaluations} or the error shape, and the
     relevant log lines or audit record. -->

```json
```

## Expected response

<!-- What you expected instead and why: the skill rule, the schema edge or the
     policy that should have applied. -->

```json
```

## Environment

- `intent-agent` version / commit:
- Model (`GEMINI_MODEL`):
- Relevant settings (timeouts, retries, cache backend):
- How it was run (uv, Docker, docker-compose, Helm):
