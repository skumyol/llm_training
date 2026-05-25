#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Check if servers are already running
if [ "$1" = "--check" ]; then
  BACKEND_OK=false
  FRONTEND_OK=false

  if curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "Backend running on http://127.0.0.1:8000"
    BACKEND_OK=true
  else
    echo "Backend NOT running"
  fi

  if curl -s http://127.0.0.1:3018 >/dev/null 2>&1; then
    echo "Frontend running on http://127.0.0.1:3018"
    FRONTEND_OK=true
  else
    echo "Frontend NOT running"
  fi

  if [ "$BACKEND_OK" = true ] && [ "$FRONTEND_OK" = true ]; then
    echo "Both servers are running."
    exit 0
  else
    exit 1
  fi
fi

echo "Building frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
  pnpm install
fi
pnpm build
cd ..

echo "Starting backend..."
cd backend
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  uv venv .venv
fi
source .venv/bin/activate
uv pip install -r requirements.txt >/dev/null 2>&1

python main.py --data ../audit_input_clean.jsonl --output ../audit_results &
BACKEND_PID=$!
cd ..

echo "Backend started on http://127.0.0.1:8000 (PID: $BACKEND_PID)"
sleep 2

echo "Starting frontend (production)..."
cd frontend
pnpm start &
FRONTEND_PID=$!
cd ..

echo "Frontend started on http://127.0.0.1:3018 (PID: $FRONTEND_PID)"
echo ""
echo "Open http://localhost:3018 in your browser"
echo "Press Ctrl+C to stop both servers"

wait
