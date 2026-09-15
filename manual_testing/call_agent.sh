#!/usr/bin/env sh

# Any non-empty token works against the mock introspection service - override with a real one
# for a real IdP: INTENT_AGENT_TOKEN=<token> ./manual_testing/call_agent.sh
TOKEN="${INTENT_AGENT_TOKEN:-manual-test-token}"
SUBJECT_TYPE="${INTENT_AGENT_SUBJECT_TYPE:-Customer}"
SUBJECT_ID="${INTENT_AGENT_SUBJECT_ID:-manual-test-subject}"

curl -s -X POST http://127.0.0.1:6003/ \
    -H "A2A-Version: 1.0" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "SendMessage",
    "params": {
      "message": {
        "role": "ROLE_USER",
        "parts": [{"text": "Can I buy stocks with my account?"}],
        "messageId": "manual-test-1",
        "metadata": {
          "subject_type": "'"${SUBJECT_TYPE}"'",
          "subject_id": "'"${SUBJECT_ID}"'"
        }
      }
    }
  }'
