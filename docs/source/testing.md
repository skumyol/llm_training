Testing Guide
=============

Running tests and validation for the NPC dialogue pipeline.

Quick Test
----------

Run all tests via root script::

   ./scripts/test_all.sh

Or run specific test suites.

LLM Fine-Tuning Tests
---------------------

Located in ``llm_finetuning/tests/``.

test_cleaning.py
~~~~~~~~~~~~~~~~

Tests data cleaning utilities::

   cd llm_finetuning
   python -m pytest tests/test_cleaning.py -v

Validates:

- Label format validation
- Invalid character detection
- JSON parsing edge cases
- Missing field handling

test_cleaning_response.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests response generation cleaning::

   python -m pytest tests/test_cleaning_response.py -v

Validates:

- Response text normalization
- Speaker prefix removal ("NPC:", "Player:")
- Empty response handling
- Special token stripping

test_training_debug.py
~~~~~~~~~~~~~~~~~~~~~~

Debug utilities for training issues::

   python -m pytest tests/test_training_debug.py -v

Covers:

- Gradient flow verification
- Loss computation checks
- Model output shape validation
- Device placement (CPU/GPU/MPS)

SLM Training Tests
------------------

Located in ``slm_training/tests/``.

test_comprehensive_training.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full training pipeline validation::

   cd slm_training
   python -m pytest tests/test_comprehensive_training.py -v

Tests:

- Data loading (Personality, Affect, Dialogue)
- Model initialization (all 6 architectures)
- Forward/backward pass
- Checkpoint saving/loading
- Evaluation metrics computation
- MLflow integration

Smoke Tests
-----------

Quick validation without full training.

SLM Smoke Test
~~~~~~~~~~~~~~~

::

   cd slm_training
   bash smoke_test.sh

Runs:

- DistilBERT personality encoder (1 epoch)
- DistilGPT2 small LM (1 epoch)

Validates environment setup and basic functionality.

External Corpus Test
~~~~~~~~~~~~~~~~~~~~

::

   bash smoke_test_external.sh

Tests on BookCorpus + OpenSubtitles subset.

Integration Tests
-----------------

End-to-end pipeline validation.

data_gen Dry Run
~~~~~~~~~~~~~~~~

Test data generation without API calls::

   cd llm_finetuning
   PYTHONPATH=. python run_data_gen.py --config configs/data_gen.yaml --dry-run --n-episodes 20

Pipeline Stages
~~~~~~~~~~~~~~~

Test individual stages::

   # Stage 1: Latent predictor (debug mode, 10 steps)
   ./scripts/pipeline.sh train latent --debug

   # Stage 2: Response generator
   ./scripts/pipeline.sh train response --debug

   # Stage 3: Joint training
   ./scripts/pipeline.sh train joint --debug

Debug Mode
----------

All training scripts support ``--debug``::

   python run_train.py --config configs/train_latent.yaml --debug

Effects:

- Reduces epochs to 1
- Limits steps per epoch to 10
- Uses smaller batch size
- Disables wandb/mlflow (or uses local file only)
- Saves checkpoints to ``checkpoints/debug/``

Manual Testing
--------------

Interactive Inference
~~~~~~~~~~~~~~~~~~~~~

Test trained model manually::

   cd llm_finetuning
   PYTHONPATH=. python src/inference/interactive.py \
       --checkpoint ../checkpoints/joint_model_best \
       --base_model Qwen/Qwen3-4B \
       --scenario ../data/world_contexts/oakhaven_siege.yaml \
       --npc commander_vance

API Health Check
~~~~~~~~~~~~~~~~

Test serving endpoint::

   curl http://localhost:8765/health
   curl http://localhost:8765/v1/models

Expected response::

   {"status": "healthy", "service_ready": true}

Frontend Dev Mode
~~~~~~~~~~~~~~~~~

Test UI without full build::

   cd slm_training/npc_frontend
   pnpm dev

Opens dev server with hot reload.

Common Issues
-------------

Out of Memory
~~~~~~~~~~~~~

Reduce batch size in config::

   batch_size: 4                    # From 16
   gradient_accumulation_steps: 8   # Increase to maintain effective batch

Or use quantization::

   quantization: 4bit

CUDA Errors
~~~~~~~~~~~

Clear cache and retry::

   rm -rf ~/.cache/huggingface/hub
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Import Errors
~~~~~~~~~~~~~

Ensure PYTHONPATH is set::

   export PYTHONPATH=/path/to/llm_finetuning:/path/to/slm_training

Or use the provided scripts which set it automatically.

MLflow Not Found
~~~~~~~~~~~~~~~~

Install optional dependency::

   pip install mlflow

Or training proceeds without logging (graceful degradation).

Validation Checklist
--------------------

Before committing changes:

- [ ] Unit tests pass: ``pytest llm_finetuning/tests/ slm_training/tests/``
- [ ] Smoke test passes: ``bash slm_training/smoke_test.sh``
- [ ] Data gen dry-run works: ``--dry-run --n-episodes 20``
- [ ] Config validates: YAML syntax check
- [ ] Docker builds: ``docker build -t npc-dialogue .``
- [ ] Serving starts: ``curl /health`` returns 200

Performance Benchmarks
----------------------

Reference times (RTX 4090):

============================ ============= =============================
Task                         Time          Command
============================ ============= =============================
Data generation (100 ep)   ~30 min       run_data_gen.py
Stage 1 training (Qwen3-4B)  ~2 hours      train_latent.yaml
Stage 2 training             ~1.5 hours    train_response.yaml
Stage 3 training             ~1 hour       train_joint.yaml
SLM HPO (20 trials × 6)    ~8 hours      train_small_lms.sh
SLM final (30 ep × 3 seed) ~4 hours      sequential_training_orchestrator.py
============================ ============= =============================

CI/CD Considerations
--------------------

For automated testing:

1. Use CPU-only runners for unit tests
2. Use ``--debug`` mode for integration tests
3. Mock API calls in data generation tests
4. Cache HuggingFace models between runs
5. Run smoke tests on PR, full training on nightly
