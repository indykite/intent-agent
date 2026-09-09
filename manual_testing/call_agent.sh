#!/usr/bin/env sh

# Any non-empty token works against the mock introspection service - override with a real one
# for a real IdP: INTENT_AGENT_TOKEN=<token> ./manual_testing/call_agent.sh
TOKEN="${INTENT_AGENT_TOKEN:-manual-test-token}"
# Only needed when the agent runs with ITS_* configured (X-IK-Token is then required); sent raw,
# no Bearer prefix: INTENT_AGENT_IK_TOKEN=<ik_token> ./manual_testing/call_agent.sh
IK_TOKEN="${INTENT_AGENT_IK_TOKEN:-}"

curl -s -X POST http://127.0.0.1:6003/ \
    -H "A2A-Version: 1.0" \
    -H "Authorization: Bearer ${TOKEN}" \
    ${IK_TOKEN:+-H "X-IK-Token: ${IK_TOKEN}"} \
    -H "Content-Type: application/json" \
    -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "SendMessage",
    "params": {
      "message": {
        "role": "ROLE_USER",
        "parts": [{"text": "Can I buy stocks with my account?"}],
        "messageId": "manual-test-1"
      }
    }
  }'
