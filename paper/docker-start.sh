#!/usr/bin/env bash
set -e

# Default config
DATA_PATH="${DATA_PATH:-/app/data/audit_input_clean.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/audit_results}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-3018}"
TEST_MODE="${TEST_MODE:-false}"

echo "========================================"
echo "NPC Audit App — Next.js + FastAPI"
echo "========================================"

# Start backend
echo "[1/2] Starting FastAPI backend on ${BACKEND_HOST}:${BACKEND_PORT} ..."
cd /app/backend
python main.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" \
  ${TEST_MODE:+--test} &
BACKEND_PID=$!

# Wait a moment for backend to be ready
sleep 3

# Start frontend
echo "[2/2] Starting Next.js frontend on ${FRONTEND_HOST}:${FRONTEND_PORT} ..."
cd /app/frontend
export PORT="$FRONTEND_PORT"
export HOSTNAME="$FRONTEND_HOST"
node server.js &
FRONTEND_PID=$!

echo ""
echo "App ready!"
echo "  Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "  Backend API: http://${BACKEND_HOST}:${BACKEND_PORT}"
echo ""
echo "Press Ctrl+C to stop both servers."

# Graceful shutdown
cleanup() {
  echo ""
  echo "Shutting down..."
  kill $FRONTEND_PID 2>/dev/null || true
  kill $BACKEND_PID 2>/dev/null || true
  wait
  exit 0
}
trap cleanup SIGINT SIGTERM

wait
