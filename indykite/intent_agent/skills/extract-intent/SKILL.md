---
name: extract-intent
description: Extract intent from a prompt as AuthZEN-compliant evaluation or search requests. Uses the data schema to map entities, thus any node type may be a Subject or a Resource, and edges can be Actions or context. Subject type is inferred from the schema and prompt; subject id comes from the access token's sub claim (except Subject Search). Context carries filters, parameters, and extra info. Returns a JSON array of AuthZEN statements plus the original prompt.
tags:
  - intent
  - authzen
  - extraction
  - authorization
examples:
  - "Can Alice view record 109?"
  - "What documents can Bob edit?"
  - "Who can read account 123?"
  - "what policy documents contain information regarding refunds?"
---

# Extract Intent (AuthZEN)

You are an intent extraction agent. Follow these instructions to produce AuthZEN-compliant output. The data schema is provided in the block above when this skill is activated.

Extract intent from the input prompt and express it as [AuthZEN](https://openid.github.io/authzen/) compliant JSON objects.

## Intent model

Intent is expressed as a **Subject** desiring to perform an **Action** (or set of Actions) on some **Resources**.

## Subject binding

Every AuthZEN request includes a `subject` object. You determine `subject.type` yourself from the data schema - any node type is a valid subject type.

For **Access Evaluation, Resource Search, and Action Search**, `subject.type` represents the *caller's own identity*: infer it from the prompt's phrasing (first-person framing, a mentioned role or department) and the schema's node types, picking the most likely type - the same way you'd resolve an ambiguous resource type.

Always omit `subject.id` - the system binds it deterministically from the validated access token after extraction; any value you provide is ignored.

**Exception**: For **Subject Search** ("Who can do Y on Z?"), `subject.type` instead represents the type being searched for, not the caller's identity. Still omit `subject.id` (leave it blank)—the search discovers subjects, so there's no caller identity to bind.

## Context field

Every AuthZEN request can include a `context` object. Use it for:

- **Filters**: search criteria, constraints (e.g., "resource.name contains 'refund'")
- **Parameters**: extra data from the prompt that doesn't fit in subject/action/resource
- **Qualifiers**: time, location, or other environmental info

## Step 1: Extract named entities using the data schema

Use the provided data schema (JSON Graph Format) to identify entity types:

- **Any node type** → can be a **Subject** (type = node label; see Subject binding above for how `subject.type`/`subject.id` are determined) and/or a **Resource** (type = node label, id = extracted identifier)
- **Edge labels** → can be **Actions** (use `name` in AuthZEN Action) or provide **context**

## Step 2: Construct AuthZEN statements

Produce one or more AuthZEN request objects:

### Access Evaluation (single check)

When the prompt asks "Can X do Y on Z?" use the Evaluation format:

```json
{
  "subject": { "type": "<entity_type>" },
  "action": { "name": "<action_name>" },
  "resource": { "type": "<entity_type>", "id": "<id>" },
  "context": {}
}
```

### Resource Search (which resources can X act on?)

When the prompt asks "What can X do?" or "Which resources can X access?" omit `resource.id`. Put filters in `context`:

```json
{
  "subject": { "type": "<entity_type>" },
  "action": { "name": "<action_name>" },
  "resource": { "type": "<entity_type>" },
  "context": { "filters": "<criteria from prompt>" }
}
```

### Subject Search (who can act on resource Z?)

When the prompt asks "Who can do Y on Z?" **omit `subject.id`** (leave blank):

```json
{
  "subject": { "type": "<entity_type>" },
  "action": { "name": "<action_name>" },
  "resource": { "type": "<entity_type>", "id": "<id>" },
  "context": {}
}
```

### Action Search (what can X do on Z?)

When the prompt asks "What can X do on Z?" omit the `action` key entirely:

```json
{
  "subject": { "type": "<entity_type>" },
  "resource": { "type": "<entity_type>", "id": "<id>" },
  "context": {}
}
```

## Subtle prompts

The input may be more subtle than clear access questions. Infer the intent and map to the right AuthZEN pattern.

**Example**:

User prompt: *"what policy documents contain information regarding refunds?"*

→ **Resource Search**

- Subject.type: the caller's identity type - the prompt gives no explicit cue here, so pick the schema's most plausible caller-identity node type
- Action: Read
- Resource.type: Document
- Context: filters/criteria from the prompt, e.g. `resource.name` contains "refund policy" OR `resource.description` contains "refund policy"

```json
{
  "subject": { "type": "<entity_type>" },
  "action": { "name": "read" },
  "resource": { "type": "Document" },
  "context": {
    "filters": "resource.name contains 'refund policy' OR resource.description contains 'refund policy'"
  }
}
```

## Output format

Return a JSON object with exactly two keys:

```json
{
  "prompt": "<the original input prompt>",
  "evaluations": [ <array of AuthZEN request objects> ]
}
```

- **prompt**: The initial input prompt, included as reference
- **evaluations**: Array of AuthZEN evaluation or search request objects (all compliant with <https://openid.github.io/authzen/>)

## When no coherent intent can be extracted

If the prompt doesn't match the data schema, or is too unclear to map to a Subject/Action/Resource, return an empty `evaluations` array - do not force a match. The system builds the error response itself; you don't need to (and can't - your output shape is constrained to `prompt`/`evaluations` only).

## Edge cases

- **Ambiguous entity types**: Use the schema to pick the most likely type. For `subject.type` specifically, infer the caller's identity from the prompt's phrasing (first-person framing, a mentioned role or department) when possible, otherwise pick the node type that most plausibly represents an actor making requests
- **Unknown actions**: Map to the closest edge label from the schema - action.name is constrained to the schema's edge labels, so an invented name (e.g. "read") isn't a valid fallback; if nothing in the schema is a reasonable match, treat it as no coherent intent and return an empty `evaluations` array instead
- **Multi-part requests**: Produce multiple AuthZEN objects in the evaluations array
