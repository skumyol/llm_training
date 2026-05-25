# NPC Audit App — Next.js + FastAPI (replaces Gradio)
#
# Build:
#   docker build -t human-audit .
#
# Run:
#   docker run -d \
#     -p 3018:3018 \
#     -v $(pwd)/audit_input_clean.jsonl:/app/data/audit_input_clean.jsonl:ro \
#     -v $(pwd)/audit_results:/app/audit_results \
#     human-audit
#
# Or use docker-compose.yml for one-command deployment.

# ---------------------------------------------------------------------------
# Stage 1: Build Next.js frontend
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies and build
COPY frontend/package.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python runtime + run both backend and frontend
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Install Node.js (needed to run the Next.js standalone server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python backend dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy built frontend (standalone output)
COPY --from=frontend-builder /app/frontend/.next/standalone ./frontend/
COPY --from=frontend-builder /app/frontend/.next/static ./frontend/.next/static

# Create directories for data and outputs
RUN mkdir -p /app/data /app/audit_results

# Copy startup script
COPY docker-start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose frontend port (backend is internal via rewrite proxy)
EXPOSE 3018

ENTRYPOINT ["/app/start.sh"]
