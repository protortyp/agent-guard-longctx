# Reproduction Guide

## 1. Install

ONNX-only (`christianengel/agent-guard-longctx`):

```bash
uv pip install -e ".[cpu]"      # or ".[cuda]"
```

Train/export/dev:

```bash
uv pip install -e ".[train]" --group dev
```

## 2. Tests

Normalizer, ONNX FP32/FP16/INT8 parity, PyTorch-free `onnxruntime` (`torch=None`, `transformers=None`), and adapters:

```bash
pytest -q
```

## 3. Build v3 dataset

Uses `datasets/held_out/` exclusion hashes (`0/0/0/0/0` overlap):

```bash
python -m agent_guard.data_recipe --version v3
```

## 4. Train (`exp06_v3_ce_ls0_4096`)

```bash
python -m agent_guard.train \
  --run-name exp06_v3_ce_ls0_4096 \
  --train-file datasets/train_v3.jsonl \
  --dev-file datasets/dev_internal_v3.jsonl \
  --output-dir checkpoints/exp06_v3_ce_ls0_4096 \
  --max-length 4096 \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --lr 3e-5 \
  --loss-type ce \
  --label-smoothing 0.0 \
  --seed 11
```

## 5. Export ONNX (`fp32`, `fp16`, `int8`)

```bash
python -m agent_guard.export_onnx \
  --checkpoint-dir checkpoints/exp06_v3_ce_ls0_4096 \
  --output-dir artifacts/agent-guard-longctx-release \
  --opset 17 \
  --exclude-classifier \
  --nodes-to-exclude /mlp/Wo/ \
  --threshold 0.9539660811424255 \
  --threshold-external-core-fpr-1pct 0.9820051130822474
```

## 6. Locked eval

```bash
# PyTorch FP32 (oracle + dev-calibrated tau=0.9540 + latency)
python -m agent_guard.eval_locked \
  --backend hf \
  --model checkpoints/exp06_v3_ce_ls0_4096 \
  --max-length 4096 \
  --batch-size 16 \
  --device cuda \
  --normalize-unicode agent_guard.normalizer:normalize_text \
  --calibrated-threshold 0.9539660811424255 \
  --output results/final_pytorch_fp32_eval_dev_tau05.json

# ONNX INT8 CPU (no PyTorch inference)
python -m agent_guard.eval_locked \
  --backend onnx \
  --onnx-path artifacts/agent-guard-longctx-release/onnx/model_quantized.onnx \
  --tokenizer-dir artifacts/agent-guard-longctx-release \
  --max-length 4096 \
  --batch-size 8 \
  --device cpu \
  --normalize-unicode agent_guard.normalizer:normalize_text \
  --calibrated-threshold 0.9539660811424255 \
  --output results/final_onnx_int8_cpu_eval.json
```
