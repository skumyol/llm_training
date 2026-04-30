# =============================================================================
# NPC Dialogue Model Serving
# =============================================================================
# Uses the lightweight PyTorch-native serving path (no vLLM dependency needed
# for LoRA adapters — vLLM is optional for high-throughput use).
#
# Build:  docker build -t npc-dialogue .
# Run:    docker run -p 8765:8765 -v ./checkpoints:/app/checkpoints npc-dialogue
# =============================================================================

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Source code
COPY llm_finetuning/src/   llm_finetuning/src/
COPY llm_finetuning/scripts/ llm_finetuning/scripts/
COPY slm_training/src/     slm_training/src/
COPY scripts/serve.py      scripts/serve.py
COPY scripts/serve.sh      scripts/serve.sh

# Checkpoints (mounted at runtime for portability)
# COPY checkpoints/ checkpoints/

ENV PYTHONPATH=/app/llm_finetuning:/app/slm_training:/app/slm_training/src/train

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8765/health || exit 1

ENTRYPOINT ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8765"]
CMD ["--system", "llm"]
