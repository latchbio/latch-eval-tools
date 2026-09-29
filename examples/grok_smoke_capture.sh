#!/usr/bin/env bash
# Capture grok-build's real headless streaming-json output so we can verify the
# parser assumptions in NOTES_grokbuild.md (token-field names, session-id
# location, -p arg vs stdin). Run from the repo root.
#
#   export XAI_API_KEY=xai-...
#   bash examples/grok_smoke_capture.sh
#
# Writes the raw event stream to examples/grok_smoke_traj.jsonl.
set -euo pipefail

if [[ -z "${XAI_API_KEY:-}" ]]; then
  echo "ERROR: XAI_API_KEY is not set. export XAI_API_KEY=xai-... first." >&2
  exit 1
fi

IMAGE=grok-smoke
OUT=examples/grok_smoke_traj.jsonl

echo "== building $IMAGE (grok CLI only) =="
docker build -f examples/Dockerfile.grok-smoke -t "$IMAGE" .

echo "== running grok headless (prompt via -p, ACP streaming-json) =="
# Mirrors exactly what _build_agent_command emits for grokbuild.
docker run --rm -i \
  -e "XAI_API_KEY=${XAI_API_KEY}" \
  -w /workspace \
  "$IMAGE" \
  bash -lc 'cd /workspace && git init -q 2>/dev/null || true; \
    grok --no-auto-update --no-alt-screen --always-approve --cwd /workspace \
         --output-format streaming-json -m grok-4.6 \
         -p "Create a file hello.txt containing the single word hi, then stop."' \
  | tee "$OUT"

echo
echo "== distinct sessionUpdate discriminators seen =="
jq -rc 'try (.params.update.sessionUpdate // .sessionUpdate) catch empty' "$OUT" 2>/dev/null | sort | uniq -c || true
echo "== any object containing a usage block (check token field names!) =="
grep -o '"usage":[^}]*}' "$OUT" | head || echo "(no 'usage' key found -- token accounting may live elsewhere)"
echo "== where the session id appears =="
grep -o '"sessionId":"[^"]*"' "$OUT" | head || echo "(no top-level/params sessionId -- check nesting)"
echo
echo "Raw stream saved to $OUT. Reconcile against _GROK_USAGE_ALIASES and"
echo "AGENT_IDENTIFIER_KEYS per NOTES_grokbuild.md."
