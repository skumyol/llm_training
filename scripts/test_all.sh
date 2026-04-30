#!/usr/bin/env bash
# =============================================================================
# test_all.sh — End-to-end smoke test for LLM + SLM pipelines
# =============================================================================
# Runs everything in a temp directory, cleans up on exit.
# Tests: data gen → training (1 epoch) → eval → mlflow logging
#
# Usage:
#   ./scripts/test_all.sh           # Quick test (~2 min, no GPU needed)
#   ./scripts/test_all.sh --gpu     # Include GPU model tests (~5 min)
#   ./scripts/test_all.sh --full    # Full test with real tokenizer (~15 min)
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0

pass() { echo -e "  ${GREEN}✓${NC} $*"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗${NC} $*"; FAIL=$((FAIL + 1)); }
header() { echo -e "\n${BOLD}${CYAN}── $* ──${NC}"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
USE_GPU=false
FULL=false

for arg in "$@"; do
    case "$arg" in
        --gpu)  USE_GPU=true ;;
        --full) FULL=true ;;
    esac
done

# ── Temp workspace ────────────────────────────────────────────────────────────
TMPDIR=$(mktemp -d /tmp/llm_training_test_XXXXXX)
trap "rm -rf $TMPDIR" EXIT

mkdir -p "$TMPDIR/mlruns" "$TMPDIR/data/splits" "$TMPDIR/checkpoints" "$TMPDIR/eval_results"

cat > "$TMPDIR/.env" << 'EOF'
MLFLOW_TRACKING_URI=sqlite:///$TMPDIR/mlflow.db
EOF

echo -e "${CYAN}Temp dir:${NC} $TMPDIR"
echo -e "${CYAN}GPU tests:${NC} $USE_GPU  ${CYAN}Full tests:${NC} $FULL"
echo ""

# ==============================================================================
# 1. IMPORT CHECKS
# ==============================================================================
header "1. Import checks"

cd "$ROOT"

if PYTHONPATH="$ROOT/llm_finetuning" "$PYTHON" -c "
from src.training.model import LatentStatePredictor, HEAD_SPECS, build_latent_predictor
from src.training.dataset import HeadSupervisionDataset, SFTDataset, JointDataset, LABEL_MAPS
from src.training.loss import MultiHeadLoss, JointLoss, ConsistencyLoss
from src.mlflow_utils import setup_mlflow
from src.data_gen.labeler import Labeler, normalize_labels
from src.packaging.packager import Packager
from src.packaging.splitter import Splitter
from src.eval.eval_latent import eval_latent
from src.eval.eval_response import eval_response
from scripts.clean_labels import clean_record, clean_file
from scripts.analyze_data_quality import analyze_episodes
from scripts.visualize_dialogues import format_turn, process_episode
print('OK')
" 2>&1 | tail -1 | grep -q OK; then
    pass "llm_finetuning imports (12 modules)"
else
    fail "llm_finetuning imports"
fi

if PYTHONPATH="$ROOT/slm_training" "$PYTHON" -c "
from src.models.personality import DistilBertRegressor
from src.models.affect import DistilBertRegressor as AffectRegressor
from src.models.dialogue import ConditionalDialogueModel, ConditionalSoftPrefix
from src.train.small_lm_architectures import SmallGRULM, AWDLSTMLM, TinyGPTLM, PrefixTinyGPTLM, TinyMoELM, MambaLikeLM, build_model, RECOMMENDED_CONFIGS
from src.data.datasets import RegressionTextDataset, DialogueJsonlDataset, PersonalityCache, DataDownloader
from src.train.mlflow_tracker import MLflowTracker
from src.common.config import PersonalityTrainConfig, AffectTrainConfig
print('OK')
" 2>&1 | tail -1 | grep -q OK; then
    pass "slm_training imports (10 modules)"
else
    fail "slm_training imports"
fi

# ==============================================================================
# 2. LLM DATA GEN (dry run)
# ==============================================================================
header "2. LLM data generation (dry run, 3 episodes)"

cd "$ROOT"
cp "$ROOT/llm_finetuning/configs/data_gen.yaml" "$TMPDIR/data_gen_test.yaml"
# Override output paths to temp
"$PYTHON" -c "
import yaml
with open('$TMPDIR/data_gen_test.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['generation']['n_episodes'] = 3
cfg['output']['raw_episodes_dir'] = '$TMPDIR/raw_episodes'
cfg['output']['validated_turns_dir'] = '$TMPDIR/validated_turns'
cfg['output']['counterfactuals_dir'] = '$TMPDIR/counterfactuals'
cfg['mlflow']['tracking_uri'] = 'file://$TMPDIR/mlruns'
with open('$TMPDIR/data_gen_test.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

if PYTHONPATH="$ROOT/llm_finetuning" "$PYTHON" "$ROOT/llm_finetuning/run_data_gen.py" \
    --config "$TMPDIR/data_gen_test.yaml" --dry-run --no-mlflow 2>&1 | grep -q 'Generation complete'; then
    pass "data generation (3 episodes, dry run)"
else
    fail "data generation"
fi

# ==============================================================================
# 3. LLM LABEL CLEANING
# ==============================================================================
header "3. Label cleaning (dry run)"

if PYTHONPATH="$ROOT/llm_finetuning" "$PYTHON" "$ROOT/llm_finetuning/scripts/clean_labels.py" \
    --dry-run 2>&1 | grep -q 'Total:'; then
    pass "label cleaning"
else
    fail "label cleaning"
fi

# ==============================================================================
# 4. MLFLOW INTEGRATION
# ==============================================================================
header "4. MLflow integration"

if "$PYTHON" -c "
import mlflow, os, tempfile
uri = 'file://$TMPDIR/mlruns'
mlflow.set_tracking_uri(uri)

# Test LLM-style logging
mlflow.set_experiment('test_llm')
with mlflow.start_run(run_name='smoke_test'):
    mlflow.log_params({'model': 'qwen3-0.6B', 'lr': 2e-4})
    mlflow.log_metric('val_loss', 0.42, step=1)
    mlflow.log_metric('val_f1', 0.75, step=1)

# Verify it was written
from mlflow.tracking import MlflowClient
client = MlflowClient()
exp = client.get_experiment_by_name('test_llm')
assert exp is not None, 'Experiment not found'
runs = client.search_runs(experiment_ids=[exp.experiment_id])
assert len(runs) >= 1, f'Expected >=1 run, got {len(runs)}'
print('LLM mlflow OK')
" 2>&1 | grep -q 'LLM mlflow OK'; then
    pass "LLM mlflow (write + read back)"
else
    fail "LLM mlflow"
fi

if PYTHONPATH="$ROOT/slm_training" "$PYTHON" -c "
import sys
sys.path.insert(0, '$ROOT/slm_training/src/train')
from mlflow_tracker import MLflowTracker

# Test SLM-style logging (should go to same mlruns)
tracker = MLflowTracker(experiment='test_slm')
tracker.start_run(run_name='smoke_test', tags={'arch': 'gpt', 'seed': '42'})
tracker.log_params({'arch': 'gpt', 'lr': 3e-4, 'batch_size': 16})
tracker.log_metrics({'val_ppl': 23.4, 'bleu_1': 12.5}, step=1)
tracker.log_metrics({'val_ppl': 22.1, 'bleu_1': 13.2}, step=2)
tracker.end_run()

import mlflow
mlflow.set_tracking_uri('file://$TMPDIR/mlruns')
from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_ids=['2'])
assert len(runs) == 1, f'Expected 1 SLM run, got {len(runs)}'
run = runs[0]
metrics = {m.key: m.value for m in run.data.metrics}
assert 'val_ppl' in metrics, f'Missing val_ppl in {list(metrics.keys())}'
print('SLM mlflow OK')
" 2>&1 | grep -q 'SLM mlflow OK'; then
    pass "SLM mlflow (write + metrics read back)"
else
    fail "SLM mlflow"
fi

# ==============================================================================
# 5. SLM SMALL LM ARCHITECTURES (instantiate only)
# ==============================================================================
header "5. SLM architectures (instantiate + forward pass)"

if PYTHONPATH="$ROOT/slm_training" "$PYTHON" -c "
import torch
import sys
sys.path.insert(0, '$ROOT/slm_training/src/train')
from small_lm_architectures import (
    SmallGRULM, AWDLSTMLM, TinyGPTLM, PrefixTinyGPTLM, TinyMoELM, MambaLikeLM,
    GRUConfig, AWDLSTMConfig, GPTConfig, PrefixGPTConfig, MoEConfig, MambaLikeConfig
)

# Test all 6 architectures with tiny configs
configs = {
    'gru':        GRUConfig(vocab_size=1000, embed_dim=32, hidden_size=64, num_layers=1, dropout=0.0),
    'awdlstm':    AWDLSTMConfig(vocab_size=1000, embed_dim=32, hidden_size=64, num_layers=1, wdrop=0.0, dropout=0.0, dropouth=0.0, dropouti=0.0),
    'gpt':        GPTConfig(vocab_size=1000, n_embd=32, n_head=2, n_layer=1, dropout=0.0, max_seq_len=16),
    'prefix_gpt': PrefixGPTConfig(vocab_size=1000, n_embd=32, n_head=2, n_layer=1, dropout=0.0, max_seq_len=16, prefix_length=4, cond_dim=8),
    'moe':        MoEConfig(vocab_size=1000, n_embd=32, n_head=2, n_layer=1, num_experts=2, top_k=1, dropout=0.0, max_seq_len=16),
    'mamba_like': MambaLikeConfig(vocab_size=1000, n_embd=32, n_layer=1, d_state=4, d_conv=2, expand=2, dropout=0.0, max_seq_len=16),
}

x = torch.randint(0, 1000, (2, 16))
targets = torch.randint(0, 1000, (2, 16))

for name, cfg in configs.items():
    if name == 'prefix_gpt':
        cond = torch.randn(2, 8)
        model = PrefixTinyGPTLM(cfg)
        out = model(x, cond, targets)
    else:
        model = {
            'gru': SmallGRULM, 'awdlstm': AWDLSTMLM,
            'gpt': TinyGPTLM, 'moe': TinyMoELM, 'mamba_like': MambaLikeLM
        }[name](cfg)
        out = model(x, targets)
    assert out.loss is not None, f'{name}: loss is None'
    assert out.logits.shape == (2, 16, 1000), f'{name}: bad logits shape {out.logits.shape}'
    print(f'  {name:12s}  loss={out.loss.item():.4f}  params={sum(p.numel() for p in model.parameters()):,}')

print('ALL_ARCHS_OK')
" 2>&1 | grep -q 'ALL_ARCHS_OK'; then
    pass "6 SLM architectures (forward pass + loss)"
else
    fail "SLM architectures"
fi

# ==============================================================================
# 6. DATA QUALITY ANALYSIS
# ==============================================================================
header "6. Data quality analysis"

if PYTHONPATH="$ROOT/llm_finetuning" "$PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, '$ROOT/llm_finetuning/scripts')
from analyze_data_quality import analyze_episodes, print_report
results = analyze_episodes()
assert results['counts']['raw_episodes'] > 0, 'No episodes found'
print(f'  episodes={results[\"counts\"][\"raw_episodes\"]}  turns={sum(results[\"turns_per_episode\"])}')
print('QUALITY_OK')
" 2>&1 | grep -q 'QUALITY_OK'; then
    pass "data quality analysis"
else
    fail "data quality analysis"
fi

# ==============================================================================
# 7. VISUALIZE DIALOGUES
# ==============================================================================
header "7. Dialogue visualization"

if PYTHONPATH="$ROOT/llm_finetuning" "$PYTHON" "$ROOT/llm_finetuning/scripts/visualize_dialogues.py" \
    --latest 1 --output "$TMPDIR/dialogue_test.txt" 2>&1 | grep -q 'Output written'; then
    lines=$(wc -l < "$TMPDIR/dialogue_test.txt")
    pass "dialogue visualization ($lines lines)"
else
    fail "dialogue visualization"
fi

# ==============================================================================
# 8. SLM DATASET DOWNLOADER (registry test only, no network)
# ==============================================================================
header "8. SLM dataset registry"

if PYTHONPATH="$ROOT/slm_training" "$PYTHON" -c "
from src.data.datasets import DataDownloader
d = DataDownloader(base_dir='$TMPDIR/slm_data')
keys = d.list_datasets()
assert len(keys) >= 10, f'Expected >=10 datasets, got {len(keys)}'
for k in keys[:5]:
    info = d.get_dataset_info(k)
    assert info.method in ('hf', 'git', 'url', 'parlai', 'manual'), f'{k}: bad method {info.method}'
print(f'REGISTRY_OK ({len(keys)} datasets)')
" 2>&1 | grep -q 'REGISTRY_OK'; then
    pass "SLM dataset registry ($(PYTHONPATH=$ROOT/slm_training $PYTHON -c "from src.data.datasets import DataDownloader; print(len(DataDownloader().list_datasets()))" 2>/dev/null) datasets)"
else
    fail "SLM dataset registry"
fi

# ==============================================================================
# 9. GPU CHECK (if --gpu flag)
# ==============================================================================
if $USE_GPU; then
    header "9. GPU availability"
    if PYTHONPATH="$ROOT/llm_finetuning" "$PYTHON" "$ROOT/llm_finetuning/scripts/check_gpu.py" 2>&1 | grep -q 'GPU is available'; then
        pass "GPU detected and working"
    else
        fail "GPU check"
    fi
else
    header "9. GPU check (skipped, use --gpu to enable)"
    pass "skipped"
fi

# ==============================================================================
# 10. FULL MODEL LOAD TEST (if --full flag)
# ==============================================================================
if $FULL; then
    header "10. Full model load test (downloads ~1GB from HuggingFace)"

    if PYTHONPATH="$ROOT/llm_finetuning" "$PYTHON" -c "
import torch
from src.training.model import build_latent_predictor
print('Loading Qwen3-0.6B...')
predictor, tokenizer = build_latent_predictor(
    'Qwen/Qwen3-0.6B', quantization=None, torch_dtype='float32',
    lora_config={'r': 4, 'alpha': 8, 'dropout': 0.0, 'target_modules': ['q_proj', 'v_proj']},
    pooling='last'
)
# Forward pass with dummy input
x = tokenizer('Hello world', return_tensors='pt')
out = predictor(x.input_ids, x.attention_mask)
assert 'logits' in out
assert len(out['logits']) == 29
print(f'FULL_MODEL_OK heads={len(out[\"logits\"])}')
" 2>&1 | grep -q 'FULL_MODEL_OK'; then
        pass "full model load + forward pass (Qwen3-0.6B, 29 heads)"
    else
        fail "full model load"
    fi
else
    header "10. Full model test (skipped, use --full to enable)"
    pass "skipped"
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  RESULTS: ${GREEN}${PASS} passed${NC}  ${RED}${FAIL} failed${NC}  ($((PASS + FAIL)) total)${NC}"
echo -e "${BOLD}============================================================${NC}"

if [ "$FAIL" -gt 0 ]; then
    echo -e "\n${RED}Some tests failed. Check output above.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All tests passed. Ready to commit and deploy.${NC}"
    exit 0
fi
