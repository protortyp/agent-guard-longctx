# ONNX Export & Transformers.js v4

[`agent_guard/export_onnx.py`](../agent_guard/export_onnx.py) exports opset-17 ONNX graphs for `onnxruntime`, `onnxruntime-web`, and `@huggingface/transformers` v4:

- `onnx/model.onnx` (`571.31 MiB` / `599,066,373` bytes) — FP32
- `onnx/model_fp16.onnx` (`286.24 MiB` / `300,142,664` bytes) — Mixed FP16
- `onnx/model_quantized.onnx` (`311.74 MiB` / `326,878,449` bytes) — `QInt8`

Exports force `attn_implementation="eager"` and `reference_compile=False` for opset-17 WASM/WebGPU compatibility.

## 1. FP16 RoPE Fix

ModernBERT alternates local attention (`rope_theta = 10000.0`) with global attention every 3 layers (`rope_theta = 160000.0`), where `inv_freq` reaches `~9e-6`. Naive FP16 casting truncates `inv_freq` and accumulates RoPE phase error over `512–4096` tokens.

`_convert_to_fp16_preserving_rope` in [`agent_guard/export_onnx.py`](../agent_guard/export_onnx.py):

1. Blocks `LayerNormalization`, `Softmax`, `Range`, `Sin`, `Cos` (`DEFAULT_FP16_BLOCK_OPS`), plus `"rotary"` nodes.
2. Restores FP32 `Constant` `inv_freq` tensors (`orig_rotary_consts`) post-conversion.
3. Collapses `Cast(FP32->FP16)->Cast(FP16->FP32)` round-trips (`_remove_fp16_roundtrips`) and prunes dead casts.
4. Re-toposorts via `topo_sort_graph` (Kahn's algorithm).

Parity (`export_manifest.json`):
- `short_128`: `max_abs_err = 0.006657`, `cosine_similarity = 0.9999989`, `top1_agreement = 1.0`.
- `long_2048`: `max_abs_err = 0.014349`, `cosine_similarity = 0.9999770`, `top1_agreement = 1.0`.
- `external_core`: `86.75%` Macro-F1 (**100.211%** retention vs PyTorch FP32).

## 2. INT8 GeGLU `/mlp/Wo/` Fix

ModernBERT GeGLU MLPs (`Wi` -> split -> `GeLU * gate` -> `Wo`) produce heavy activation outliers entering `Wo`. Uniform `QInt8` `quantize_dynamic` saturates `/mlp/Wo/MatMul` scales across all 22 layers.

To avoid unsupported per-channel/static QDQ in browser WASM/WebGPU, release export excludes:

- `/model/model/layers.{0..21}/mlp/Wo/MatMul` (22 layers)
- `/model/head/dense/MatMul`, `/model/classifier/Gemm`

Release command:

```bash
python -m agent_guard.export_onnx \
  --checkpoint-dir checkpoints/exp06_v3_ce_ls0_4096 \
  --output-dir artifacts/agent-guard-longctx-release \
  --exclude-classifier \
  --nodes-to-exclude /mlp/Wo/ \
  --threshold 0.9539660811424255 \
  --threshold-external-core-fpr-1pct 0.9820051130822474
```

Locked eval ([`final_onnx_int8_cpu_eval.json`](../results/final_onnx_int8_cpu_eval.json)):
- `model_quantized.onnx`: `311.74 MiB` (`54.6%` of FP32).
- `external_core`: `86.69%` Macro-F1 (**100.140%** oracle retention; `87.40%` / **100.004%** @ `tau=0.9540`).
- `longctx_indirect`: `76.67%` recall @ `0.00%` FPR (`78.67%` @ `tau=0.9540`).
- `512` CPU p50 (Ryzen 9 5950X): `626.81 ms` -> **`323.59 ms`**.

## 3. `Transformers.js v4`

`{ device: "webgpu", dtype: "q8" }` loads `config.json`, `tokenizer.json`, and `onnx/model_quantized.onnx` (auto-fallback to `"wasm"`). Run the JS port of [`normalizer.py`](../agent_guard/normalizer.py) ([WebGPU Space](https://huggingface.co/spaces/christianengel/agent-guard-demo)) pre-tokenization.
