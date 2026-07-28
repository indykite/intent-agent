---
name: extract-intent
description: Extract intent from a prompt as AuthZEN-compliant evaluation or search requests. Uses the data schema to map entities, thus DigitalTwin nodes may be Subjects, edges can be Actions or context, all entities can be Resources. Subject id comes from the access token's sub claim (except Subject Search). Context carries filters, parameters, and extra info. Returns a JSON array of AuthZEN statements plus the original prompt.
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

You are an intent extraction agent. Follow these instructions to produce AuthZEN-compliant output. The data schema and subject from token are provided in the blocks above when this skill is activated.

Extract intent from the input prompt and express it as [AuthZEN](https://openid.github.io/authzen/) compliant JSON objects.

## Intent model

Intent is expressed as a **Subject** desiring to perform an **Action** (or set of Actions) on some **Resources**.

## Subject from access token

The Subject is typically provided by the access token of the incoming request. Use the token's `sub` claim as `subject.id`. The `subject.type` comes from the schema (e.g., DigitalTwin node label).

**Exception**: For **Subject Search** ("Who can do Y on Z?"), leave `subject.id` blank (omit it or use empty string)—the search discovers subjects.

## Context field

Every AuthZEN request can include a `context` object. Use it for:

- **Filters**: search criteria, constraints (e.g., "resource.name contains 'refund'")
- **Parameters**: extra data from the prompt that doesn't fit in subject/action/resource
- **Qualifiers**: time, location, or other environmental info

## Step 1: Extract named entities using the data schema

Use the provided data schema (JSON Graph Format) to identify entity types:

- **Nodes with `system_labels` containing "DigitalTwin"** → may be **Subjects** (type = node label, id = from token `sub`)
- **Edge labels** → can be **Actions** (use `name` in AuthZEN Action) or provide **context**
- **All node types** (from `user_defined_labels`) → can be **Resources** (type = node label, id = extracted identifier)

## Step 2: Construct AuthZEN statements

Produce one or more AuthZEN request objects:

### Access Evaluation (single check)

When the prompt asks "Can X do Y on Z?" use the Evaluation format:

```json
{
  "subject": { "type": "<entity_type>", "id": "<from token sub>" },
  "action": { "name": "<action_name>" },
  "resource": { "type": "<entity_type>", "id": "<id>" },
  "context": {}
}
```

### Resource Search (which resources can X act on?)

When the prompt asks "What can X do?" or "Which resources can X access?" omit `resource.id`. Put filters in `context`:

```json
{
  "subject": { "type": "<entity_type>", "id": "<from token sub>" },
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
  "subject": { "type": "<entity_type>", "id": "<from token sub>" },
  "resource": { "type": "<entity_type>", "id": "<id>" },
  "context": {}
}
```

## Subtle prompts

The input may be more subtle than clear access questions. Infer the intent and map to the right AuthZEN pattern.

**Example**:

User prompt: *"what policy documents contain information regarding refunds?"*

→ **Resource Search**
- Subject: from token (`sub` claim)
- Action: Read
- Resource.type: Document
- Context: filters/criteria from the prompt, e.g. `resource.name` contains "refund policy" OR `resource.description` contains "refund policy"

```json
{
  "subject": { "type": "user", "id": "<from token sub>" },
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
- **evaluations**: Array of AuthZEN evaluation or search request objects (all compliant with https://openid.github.io/authzen/)

## Edge cases

- **Ambiguous entity types**: Use the schema to pick the most likely type; prefer DigitalTwin for people/users
- **Unknown actions**: Map to the closest edge label from the schema, or use a generic action name (e.g., "read")
- **Multi-part requests**: Produce multiple AuthZEN objects in the evaluations array
- **No schema available**: Use generic types (e.g., subject type "user", resource type "record") and note uncertainty
- **No token sub**: If the token has no `sub` claim, omit subject.id for non–Subject Search and note in context
