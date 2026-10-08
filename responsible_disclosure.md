# Responsible Disclosure

Thank you for taking the time to help keep this project, and the systems that put it in front of an
authorization engine, safe.

## Why this service needs a disclosure process

The Intent Agent sits between an AI agent and an authorization decision. It authenticates callers
with bearer tokens (validated through RFC 7662 token introspection against an external identity
provider), sends natural-language prompts, the graph schema and the fetched policy definitions to
an external LLM, and turns the result into AuthZEN-style evaluations that a downstream engine
decides on. It also writes an audit
record for every request. A bug here can therefore be more than a crash: a crafted prompt could
produce an evaluation that misrepresents who is asking or what they are asking for, a token-handling
flaw could let an unauthenticated caller in, and an audit gap could hide either.

## In scope

- Bypassing or weakening caller authentication: accepting an inactive, malformed or wrong-audience
  token, or any path that reaches extraction without a validated token.
- Prompt injection or other inputs that make the extracted intent carry a subject, action or
  resource the caller did not ask for (a `subject` other than the authenticated one, a resource
  outside the schema, a CEL filter that escapes the `resource` variable).
- Leaking secrets or internal configuration (API keys, introspection credentials, audit webhook
  credentials) in responses, logs, audit records or error messages.
- Server-side request forgery or path handling in the schema, policy, introspection or audit
  webhook clients.
- Inputs that hang the service, exhaust memory or CPU, or defeat the per-request timeouts.
- Audit records that are lost, duplicated or altered for a request that was served.

## Out of scope

- Vulnerabilities in the LLM provider, the identity provider, Redis, Python or other third-party
  components: report those upstream.
- The authorization decision itself. This service makes none; a wrong decision on a correct
  evaluation belongs to the engine that made it.
- Issues that require an already-compromised host, a malicious operator or leaked deployment
  credentials.
- Theoretical concerns without a demonstrable trigger.
- Wrong-but-honest extractions for unusual prompts. That is a regular bug, so open an issue.

## How to report

**Contact:** <responsible-disclosure@indykite.com>

When reporting, include:

1. The affected component (module or file) and commit hash or release version.
2. A minimal reproduction: the request, the schema or policies served, and the configuration needed.
3. The impact: what an attacker could achieve, and under what assumptions.
4. Any suggested mitigation, if you have one.

Please **do not** open a public GitHub issue for security-sensitive reports.

## What to expect

- **Acknowledgement** within 5 business days.
- **Triage and severity assessment** within 10 business days.
- **Fix or mitigation** on a timeline proportional to severity.
- **Credit** in the changelog or release notes if you would like it. Let us know your preferred
  name or handle.

## Safe harbor

We will not pursue legal action against researchers who:

- Make a good-faith effort to avoid privacy violations, data destruction and service disruption
  while testing.
- Test only against instances they operate themselves, never against IndyKite's or a customer's
  deployment.
- Report findings privately and give a reasonable window for remediation before public disclosure.
- Do not exploit a finding beyond what is necessary to demonstrate it.

If in doubt about whether your testing falls within these guidelines, ask first.
