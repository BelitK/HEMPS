#!/usr/bin/env bash
set -euo pipefail

HOST="0.0.0.0"

MANGO_PORT="8000"
LLM_PORT="9001"
UI_PORT="${UI_PORT:-8501}"

MANGO_LOG="${MANGO_LOG:-mango_server.log}"
LLM_LOG="${LLM_LOG:-llm_controller.log}"
UI_LOG="${UI_LOG:-streamlit_ui.log}"

LOG_LEVEL="debug"

# Activate venv
source .venv/bin/activate

wait_for_http() {
  local url="$1"
  local name="$2"
  echo "Waiting for ${name} to become available..."
  for i in {1..60}; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✔ ${name} is up."
      return 0
    fi
    sleep 0.5
  done
  echo "✖ ${name} did not start in time."
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

# Wait until Mango is ready
wait_for_http "http://127.0.0.1:${MANGO_PORT}/topology" "Mango"

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

# Wait until LLM is ready
wait_for_http "http://127.0.0.1:${LLM_PORT}/health" "LLM"

echo
echo "========================================"
echo "Starting Streamlit UI"
echo "  App: ui.py"
echo "  Host: $HOST"
echo "  Port: $UI_PORT"
echo "  Log:  $UI_LOG"
echo "========================================"

# Streamlit binds separately with --server.address/--server.port
# Note: --server.headless=true prevents Streamlit from trying to open a browser automatically.
streamlit run ui.py \
  --server.address "$HOST" \
  --server.port "$UI_PORT" \
  --server.headless true \
  2>&1 | tee -a "$UI_LOG" &

UI_PID=$!

echo
echo "All services are running."
echo "  Mango PID:     $MANGO_PID"
echo "  LLM PID:       $LLM_PID"
echo "  Streamlit PID: $UI_PID"
echo
echo "UI should be available at:"
echo "  http://127.0.0.1:${UI_PORT}"
echo
echo "Press Ctrl+C to stop everything."
echo

cleanup() {
  echo
  echo "Shutting down services..."
  # kill in reverse order (UI -> LLM -> Mango) so UI stops calling APIs first
  kill "$UI_PID" "$LLM_PID" "$MANGO_PID" 2>/dev/null || true
  wait "$UI_PID" "$LLM_PID" "$MANGO_PID" 2>/dev/null || true
  echo "Shutdown complete."
}

trap cleanup SIGINT SIGTERM

wait
