# Pipeline Progress

## Status: Week 3–4 of 12-week plan

---

## Background Processes (started Mar 15 2026)

| Process | PID | Log | Status |
|---------|-----|-----|--------|
| 500-episode data generation | 1000233 | `data_gen_500.log` | Running |
| Qwen3-4B download | 1003192 | `qwen3_4b_download.log` | Running |

Check generation: `grep "valid=" data_gen_500.log | tail -5`  
Check download: `tail -3 qwen3_4b_download.log`

---

## Data Generation — COMPLETE (pipeline ready)

All 7 scenario types × 5 templates = 35 scenarios.  
Last verified run: 10 episodes → 67 valid + 108 CF = 175 records, 0 validation errors.

Target after 500-episode run: ~3,350 valid + ~5,400 CF ≈ 8,750 records  
_(still short of 40k target — need 5000 episodes total for full training)_

---

## Training Pipeline — VERIFIED (smoke test 20/20)

All components tested with mock backbone, no real model load needed.

### Bugs fixed this session
- `dataset.py`: Added `probe`, `negotiate` to `PLAYER_INTENT_LABELS` (now 9 classes)
- `dataset.py`: Added `clarify` to `RESPONSE_POLICY_LABELS` (now 10 classes)
- `model.py`: Updated `HEAD_SPECS` n_classes to match (player_intent=9, response_policy=10)
- `model.py`: Cast `pooled` to `float32` before heads (bfloat16/float32 mismatch fix)
- `train_joint.py`: Replaced `SFTDataset` with new `JointDataset` (head labels were never in batch)
- `dataset.py`: Added `JointDataset` + `collate_joint_batch`
- `configs/train_joint.yaml`: Added `debug_model`, `heads_max_seq_len`

---

## Next Steps (in order)

### 1. When 500-episode generation finishes
```bash
# Verify output
grep "Generation complete" data_gen_500.log
wc -l data/splits/train_heads.jsonl

# If ~8k records → run Stage 1 debug training first
python run_train.py --stage latent --config configs/train_latent.yaml --debug
```

### 2. Scale to 5000 episodes (after Qwen3-4B confirmed working)
```bash
# Clear old data first
rm -rf data/raw_episodes data/validated_turns data/counterfactuals data/merged_validated data/packaged data/splits
nohup .venv/bin/python run_data_gen.py --config configs/data_gen.yaml --n-episodes 5000 --no-mlflow --stage all > data_gen_5000.log 2>&1 &
```

### 3. Stage 1 training (latent-state predictor)
```bash
python run_train.py --stage latent --config configs/train_latent.yaml
# Target: response_policy_f1 >= 0.75, stance_delta_accuracy >= 0.70
```

### 4. Stage 1 eval
```bash
python run_eval.py --stage latent --config configs/eval.yaml
```

### 5. Stage 2 training (response generator, gold state)
```bash
python run_train.py --stage response --config configs/train_response.yaml
```

### 6. Stage 3 joint fine-tuning
```bash
python run_train.py --stage joint --config configs/train_joint.yaml
```

---

## Known Limitations / Future Work

- Validation is 100% (19/19 diagnostic turns) but tested on only 3 scenarios
- `eval.yaml` `base_model` defaults to `Qwen/Qwen3-4B` inside `eval_latent.py` — fine
- `eval_response.py` SECRECY_KEYWORDS list is hardcoded; should derive from scenario bank
- Stage 4 (selective router) not yet implemented in training code
- MLflow tracking disabled (`--no-mlflow`) during data generation; enable for training
