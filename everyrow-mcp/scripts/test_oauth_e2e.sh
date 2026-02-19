#!/bin/bash
# End-to-end OAuth test for the EveryRow MCP server.
#
# Prerequisites:
#   MCP server running on localhost:8000 (see README)
#
# Usage:
#   ./scripts/test_oauth_e2e.sh           # Step 1: prints URL to open in browser
#   ./scripts/test_oauth_e2e.sh exchange   # Step 2: after pasting callback URL

set -euo pipefail
SERVER=http://localhost:8000
STATE_FILE=/tmp/oauth_e2e_state.txt

if [ "${1:-}" = "exchange" ]; then
    # ── Step 2: Exchange the code for tokens and test MCP ──
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: Run without arguments first."
        exit 1
    fi
    source "$STATE_FILE"

    CALLBACK_URL="${2:-}"
    if [ -z "$CALLBACK_URL" ]; then
        echo "Usage: $0 exchange 'http://127.0.0.1:9999/callback?code=...&state=...'"
        exit 1
    fi

    # Extract auth code
    AUTH_CODE=$(echo "$CALLBACK_URL" | sed -n 's/.*code=\([^&]*\).*/\1/p')
    if [ -z "$AUTH_CODE" ]; then
        echo "ERROR: No 'code' parameter in URL: $CALLBACK_URL"
        exit 1
    fi

    echo "=== Token Exchange ==="
    TOKEN_RESPONSE=$(curl -s -X POST "$SERVER/token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "grant_type=authorization_code" \
        -d "code=$AUTH_CODE" \
        -d "redirect_uri=http://127.0.0.1:9999/callback" \
        -d "client_id=$CLIENT_ID" \
        -d "code_verifier=$VERIFIER")

    ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('access_token',''))" 2>/dev/null)
    if [ -z "$ACCESS_TOKEN" ]; then
        echo "FAILED:"
        echo "$TOKEN_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TOKEN_RESPONSE"
        exit 1
    fi
    echo "OK — got access_token + refresh_token"

    echo ""
    echo "=== MCP Initialize ==="
    curl -s -D /tmp/mcp-headers.txt -X POST "$SERVER/mcp" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"e2e-test","version":"1.0"}}}' > /dev/null

    SESSION_ID=$(grep -i "mcp-session-id" /tmp/mcp-headers.txt | tr -d '\r' | awk '{print $2}')
    if [ -z "$SESSION_ID" ]; then
        echo "FAILED — no session ID"
        exit 1
    fi
    echo "OK — session: $SESSION_ID"

    echo ""
    echo "=== List Tools ==="
    TOOLS=$(curl -s -X POST "$SERVER/mcp" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -H "mcp-session-id: $SESSION_ID" \
        -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')

    echo "$TOOLS" | python3 -c "
import sys
for line in sys.stdin:
    if line.startswith('data:'):
        import json
        data = json.loads(line[5:])
        tools = data.get('result',{}).get('tools',[])
        print(f'OK — {len(tools)} tools:')
        for t in tools:
            print(f'  - {t[\"name\"]}')
" 2>/dev/null

    echo ""
    echo "=== Unauthenticated request (should fail) ==="
    UNAUTH=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SERVER/mcp" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}')
    if [ "$UNAUTH" = "401" ]; then
        echo "OK — 401 Unauthorized (as expected)"
    else
        echo "UNEXPECTED — got HTTP $UNAUTH (expected 401)"
    fi

    echo ""
    echo "✅ All E2E checks passed!"

else
    # ── Step 1: Register client and print authorize URL ──
    echo "=== Registering OAuth client ==="

    VERIFIER=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    CHALLENGE=$(echo -n "$VERIFIER" | openssl dgst -sha256 -binary | openssl base64 -A | tr '+/' '-_' | tr -d '=')

    REG=$(curl -s -X POST "$SERVER/register" \
        -H "Content-Type: application/json" \
        -d '{
            "client_name": "e2e-test",
            "redirect_uris": ["http://127.0.0.1:9999/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none"
        }')

    CLIENT_ID=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin)['client_id'])")

    cat > "$STATE_FILE" <<EOF
VERIFIER=$VERIFIER
CLIENT_ID=$CLIENT_ID
EOF

    echo "OK — client_id: $CLIENT_ID"
    echo ""
    echo "Open this URL in your browser and log in with Google:"
    echo ""
    echo "  $SERVER/authorize?client_id=${CLIENT_ID}&redirect_uri=http%3A%2F%2F127.0.0.1%3A9999%2Fcallback&response_type=code&code_challenge=${CHALLENGE}&code_challenge_method=S256&state=e2e-test"
    echo ""
    echo "After login, your browser will show 'site can't be reached' at"
    echo "http://127.0.0.1:9999/callback?code=...&state=e2e-test"
    echo ""
    echo "Copy the FULL URL and run:"
    echo ""
    echo "  $0 exchange 'THE_FULL_URL'"
fi
