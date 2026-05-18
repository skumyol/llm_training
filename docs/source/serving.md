Model Serving & Deployment
===========================

Serve trained NPC dialogue models behind an OpenAI-compatible API so you can chat with them using any ChatGPT-compatible frontend.

The commands below assume trained checkpoints have been synced into the paths shown. This checkout does not include the generated split data or trained model weights.

Quick Start
-----------

.. code-block:: bash

   # LLM fine-tuned model (Qwen3 + LoRA)
   ./scripts/serve.sh llm

   # SLM from scratch (GPT, GRU, etc.)
   ./scripts/serve.sh slm --arch gpt

   # Both systems on one port
   ./scripts/serve.sh both

   # Test with curl
   curl http://localhost:8765/health
   curl http://localhost:8765/v1/models

   # Chat
   curl http://localhost:8765/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "llm",
       "messages": [{"role": "user", "content": "Have you caught the spy yet?"}]
     }'

API Endpoints
-------------

.. list-table::
   :header-rows: 1

   * - Method
     - Path
     - Description
   * - GET
     - ``/health``
     - Health check, lists available backends
   * - GET
     - ``/v1/models``
     - List available models (OpenAI format)
   * - POST
     - ``/v1/chat/completions``
     - Chat completion (OpenAI-compatible)
   * - GET
     - ``/docs``
     - Swagger UI (auto-generated)

Serving Backends
-----------------

Three options depending on throughput needs:

1. Native PyTorch (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Good for: single-user testing, low latency, no extra dependencies.

.. code-block:: bash

   ./scripts/serve.sh llm --backend native
   ./scripts/serve.sh slm --backend native

Uses the model's native ``generate()`` method with PyTorch inference.

2. vLLM (high throughput)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Good for: multi-user production, batching, continuous batching.

.. code-block:: bash

   # Requires vLLM: pip install vllm
   ./scripts/serve.sh llm --backend vllm --checkpoint checkpoints/response_generator_best

   # Or via docker-compose
   docker compose --profile gpu-heavy up vllm

vLLM serves the merged LoRA adapter as an OpenAI-compatible endpoint on port 8000.

3. Docker Deployment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build and run
   docker compose up -d llm        # LLM backend on :8765
   docker compose up -d slm        # SLM backend on :8766
   docker compose up -d vllm       # vLLM on :8000 (needs GPU)

   # Check
   curl http://localhost:8765/health

Docker Compose Services
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Service
     - Port
     - Backend
     - GPU
   * - ``llm``
     - 8765
     - Native PyTorch (Qwen3 4-bit)
     - Yes
   * - ``slm``
     - 8766
     - Native PyTorch (small LM)
     - Optional
   * - ``vllm``
     - 8000
     - vLLM (high throughput)
     - Yes

SLM Chat (Terminal REPL)
--------------------------

For interactive terminal-based NPC chat:

.. code-block:: bash

   cd slm_training
   PYTHONPATH=. python -m src.infer.chat

In-session commands:

.. list-table::
   :header-rows: 1

   * - Command
     - Action
   * - ``/npc <id> <profile>``
     - Register & switch NPC
   * - ``/load <world.yaml>``
     - Load all NPCs from world context
   * - ``/switch <id>``
     - Switch active NPC
   * - ``/state``
     - Show personality + affect vectors
   * - ``/temp <0.0-2.0>``
     - Set generation temperature
   * - ``/tokens <int>``
     - Set max new tokens
   * - ``/save <path>``
     - Save conversation log
   * - ``/quit``
     - Exit

LLM Interactive Chat
---------------------

.. code-block:: bash

   cd llm_finetuning
   PYTHONPATH=. python src/inference/interactive.py \
       --checkpoint ../checkpoints/joint_model_best/ \
       --base_model Qwen/Qwen3-1.7B \
       --scenario ../data/scenario_bank/secret_extraction.yaml \
       --npc guard_captain

SLM FastAPI Server (Full NPC Backend)
--------------------------------------

The SLM has a full NPC backend with persistent memory:

.. code-block:: bash

   cd slm_training
   PYTHONPATH=. uvicorn src.api.server:app --reload --port 8765

Additional endpoints:

.. list-table::
   :header-rows: 1

   * - Method
     - Path
     - Description
   * - GET
     - ``/api/npcs``
     - List registered NPCs
   * - POST
     - ``/api/npcs``
     - Register new NPC (encodes personality)
   * - DELETE
     - ``/api/npcs/{id}``
     - Remove NPC
   * - POST
     - ``/api/chat/{id}``
     - Chat turn → response + full state
   * - POST
     - ``/api/reset/{id}``
     - Clear conversation history
   * - GET
     - ``/api/eval``
     - Encoder + dialogue metrics

Connecting Frontends
--------------------

Any OpenAI-compatible frontend works:

- **Continue.dev / Cursor / Windsurf:** Point to ``http://host:8765/v1`` as custom provider
- **Open WebUI:** Add as OpenAI-compatible endpoint
- **ChatGPT UI clones:** Set ``OPENAI_API_BASE=http://host:8765/v1``
- **Custom apps:** Use ``openai`` Python library with ``base_url`` parameter

.. code-block:: python

   from openai import OpenAI

   client = OpenAI(base_url="http://localhost:8765/v1", api_key="not-needed")

   response = client.chat.completions.create(
       model="llm",
       messages=[{"role": "user", "content": "Who goes there?"}],
   )
   print(response.choices[0].message.content)
