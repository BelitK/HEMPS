#!/usr/bin/env bash
set -e

HOST="127.0.0.1"
PORT="8000"
SERVER_LOG="${SERVER_LOG:-server.log}"
LOG_LEVEL="debug"

source .venv/bin/activate

echo "Starting uvicorn (verbose)"
echo "  App: mango_server:app"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Log level: $LOG_LEVEL"
echo "  Reload: enabled"
echo "  Access log: enabled"
echo "  Logging to: $SERVER_LOG"
echo

# Show logs in terminal AND write to file
uvicorn mango_server:app \
  --reload \
  --host "$HOST" \
  --port "$PORT" \
  --log-level "$LOG_LEVEL" \
  --access-log \
  2>&1 | tee -a "$SERVER_LOG"
