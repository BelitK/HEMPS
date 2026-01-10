#!/usr/bin/env bash
set -euo pipefail

HOST="0.0.0.0"

MANGO_PORT="8000"
LLM_PORT="9001"

MANGO_LOG="${MANGO_LOG:-mango_server.log}"
LLM_LOG="${LLM_LOG:-llm_controller.log}"

LOG_LEVEL="debug"

# Activate venv
source .venv/bin/activate

wait_for_mango() {
  local url="http://127.0.0.1:${MANGO_PORT}/topology"
  echo "Waiting for Mango server to become available..."
  for i in {1..40}; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✔ Mango is up."
      return 0
    fi
    sleep 0.5
  done
  echo "✖ Mango did not start in time."
  exit 1
}

echo "========================================"
echo "Starting Mango Runtime Server"
echo "  App: mango_server:app"
echo "  Host: $HOST"
echo "  Port: $MANGO_PORT"
echo "  Log:  $MANGO_LOG"
echo "========================================"

uvicorn mango_server:app \
  --reload \
  --host "$HOST" \
  --port "$MANGO_PORT" \
  --log-level "$LOG_LEVEL" \
  --access-log \
  2>&1 | tee -a "$MANGO_LOG" &

MANGO_PID=$!

# 🔒 wait until Mango is actually ready
wait_for_mango

echo
echo "========================================"
echo "Starting LLM Service"
echo "  App: llm_controller:app"
echo "  Host: $HOST"
echo "  Port: $LLM_PORT"
echo "  Log:  $LLM_LOG"
echo "========================================"

uvicorn llm_controller:app \
  --reload \
  --host "$HOST" \
  --port "$LLM_PORT" \
  --log-level "$LOG_LEVEL" \
  --access-log \
  2>&1 | tee -a "$LLM_LOG" &

LLM_PID=$!

echo
echo "Both servers are running."
echo "  Mango PID: $MANGO_PID"
echo "  LLM   PID: $LLM_PID"
echo
echo "Press Ctrl+C to stop everything."
echo

cleanup() {
  echo
  echo "Shutting down servers..."
  kill "$MANGO_PID" "$LLM_PID" 2>/dev/null || true
  wait "$MANGO_PID" "$LLM_PID" 2>/dev/null || true
  echo "Shutdown complete."
}

trap cleanup SIGINT SIGTERM

wait
