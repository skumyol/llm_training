Docker Deployment
==================

Production deployment using Docker and docker-compose.

Quick Start
-----------

Build and run the LLM backend::

   docker build -t npc-dialogue .
   docker run -p 8765:8765 -v ./checkpoints:/app/checkpoints npc-dialogue

Or use docker-compose for full stack::

   docker compose up -d              # Start LLM backend
   docker compose up -d slm          # Start SLM backend
   docker compose up -d vllm         # Start vLLM backend (high throughput)

Verify::

   curl http://localhost:8765/v1/models
   curl http://localhost:8765/health

Services
--------

llm (port 8765)
   Native PyTorch backend serving Qwen3 + LoRA adapter.
   Best for: Development, smaller scale, full latent state prediction.

slm (port 8766)
   Small language models trained from scratch (GRU, GPT, etc.).
   Best for: Resource-constrained environments, ablation studies.

vllm (port 8000)
   High-throughput serving via vLLM with LoRA support.
   Best for: Production, high concurrent load.
   Requires: vllm/vllm-openai:latest image.

Environment Variables
----------------------

=================== =================================================
Variable            Description
=================== =================================================
PYTHONPATH          /app/llm_finetuning:/app/slm_training:/app/slm_training/src/train
PYTORCH_CUDA_ALLOC_CONF  expandable_segments:True
=================== =================================================

Volumes
-------

=================== ========= =================================================
Host Path           Container Purpose
=================== ========= =================================================
./checkpoints       /app/checkpoints    Model weights (read-only)
./data              /app/data           Scenarios and world contexts
./llm_finetuning/configs /app/llm_finetuning/configs  Config files
~/.cache/huggingface  /root/.cache/huggingface  HuggingFace cache (vLLM)
=================== ========= =================================================

Dockerfile Details
------------------

Base image::

   pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

Build steps:

1. Install system deps (build-essential, git, curl)
2. Install Python requirements + fastapi, uvicorn
3. Copy source code from llm_finetuning/, slm_training/
4. Expose port 8765
5. Healthcheck: curl http://localhost:8765/health
6. Entrypoint: python scripts/serve.py

Profiles
--------

Use docker-compose profiles to select services::

   docker compose --profile gpu-heavy up -d vllm    # GPU-intensive only

Available profiles:

- (default): llm, slm services
- gpu-heavy: vllm service

Notes
-----

- Checkpoints are mounted at runtime, not copied into image (portability)
- GPU reservations require NVIDIA Container Toolkit
- vLLM service uses profiles to avoid accidental resource consumption
