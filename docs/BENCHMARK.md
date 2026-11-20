# Benchmarks

## Setup

- **Base**: [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) @ `8949b909ec900327062f0ebf497f51aef5e6f0c8` (`149,606,402` params, `max_position_embeddings=8192`, `max_length=4096`).
- **Baseline**: [`protectai/deberta-v3-base-prompt-injection-v2`](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) @ `90c9989b1a342275dd0d1a95aad283c04e075671` (`184,423,682` params, `max_length=512`).
- **Held-out** (`held_out_manifest_sha256: 70bf12d6c0ec24ed616f92488a9a3179d61043e000bb39bb939623bfa5abf045`):
  - `external_core.jsonl`: `2,401` (`736` pos / `1,665` neg, 9 sources)
  - `longctx_indirect.jsonl`: `300` (`150` pos / `150` neg READMEs)
  - `unicode_obfuscated.jsonl`: `300` (`150` Tags/VS/ZW / `150` neg)
- **Overlap** ([`split_overlap_audit_v3.json`](../results/split_overlap_audit_v3.json)): `67,588` train, `5,749` dev; `0/0/0/0/0` overlap.
- **Hardware**: RTX 3090, Ryzen 9 5950X (eval); Vast.ai RTX 5090 (`exp04`–`exp06`, `$1.84`).

## Trajectory (`exp01`–`exp06`)

| Run | Recipe & Key Change | Max Len | `external_core` AUROC | `external_core` R@1%FPR | `external_core` Macro-F1 | `longctx` R / FPR | `unicode` R / FPR | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | `protectai/deberta-v3-base-prompt-injection-v2` | 512 | `0.8363` | `66.71%` | `85.83%` | `1.33%` / `2.67%` | `11.33%` / `0.00%` | Reference |
| `exp01` | v1 data (`44,391` rows), `weighted_ce`/`ce` | 2048 | `0.8258` | `18.07%` | `57.13%` | `0.00%` / `0.67%` | `0.67%` / `0.00%` | Discarded (pooling/data bug fixed) |
| `exp02` | v2 data (`48,902` rows), `cls` pooling, FP32 master, CE `ls=0.0` | 3584 | `0.9802` | `71.06%` | `87.77%` | `42.67%` / `0.00%` | `36.00%` / `0.00%` | Kept (`external_core` strong, weak longctx/unicode) |
| `exp03` | v3 data (`67,588` rows, zero-truncation 4K + conversational HN), CE `ls=0.05` | 4096 | `0.9611` | `62.50%` | `83.89%` | `71.33%` / `0.00%` | `81.33%` / `15.33%` | Discarded (`ls=0.05` caused unicode benign FPs) |
| `exp04` | v3 data, Focal loss `gamma=2.0`, `ls=0.03` (RTX 5090) | 4096 | `0.9788` | `54.62%` | `80.04%` | `67.33%` / `0.00%` | `84.00%` / `2.00%` | Discarded (lower R@1%FPR) |
| `exp05` | v3 data, `weighted_ce` `neg_weight_hard=1.5`, `lr=2e-5`, `ls=0.05` (RTX 5090) | 4096 | `0.9665` | `61.68%` | `83.50%` | `74.67%` / `0.00%` | `83.33%` / `8.67%` | Discarded (`unicode` FPR `8.67%`) |
| `soup_02_04_05` | Equal linear soup `0.33*exp02 + 0.33*exp04 + 0.34*exp05` | 4096 | `0.9783` | `62.77%` | `84.01%` | `60.67%` / `0.00%` | `72.00%` / `0.00%` | Rejected (dilutes `longctx` recall) |
| `soup_02_06` | `0.50*exp02 + 0.50*exp06` / `0.35*exp02 + 0.65*exp06` / `35/45/20` | 4096 | `0.9778` | `67.53%` | `86.20%` | `51.33–63.33%` / `0.00%` | `69.33–75.33%` / `0.67%` | Rejected (`exp02` pulls `longctx` < 65% gate) |
| **`exp06`** | **v3 data, pure CE `ls=0.0`, `neg_weight_hard=1.0`, `lr=3e-5` (RTX 5090)** | **4096** | **`0.9744`** | **`68.34%`** | **`86.57%`** | **`74.67%`** / **`0.00%`** | **`83.33%`** / **`1.33%`** | **Selected (passes all 6 release gates)** |

## Suite metrics & 95% bootstrap CIs

| Metric | DeBERTa-v3 (`0.9945`) | Agent Guard Oracle (`0.9820`) | Agent Guard `tau=0.9540` (`config.json`) | Agent Guard `tau=0.4751` (Dev 1% FPR) |
| --- | ---: | ---: | ---: | ---: |
| `external_core` AUROC | `0.8363` `[0.8140, 0.8592]` | **`0.9744`** `[0.9687, 0.9797]` | `0.9744` | `0.9744` |
| `external_core` AUPRC | `0.8436` `[0.8234, 0.8649]` | **`0.9446`** `[0.9304, 0.9580]` | `0.9446` | `0.9446` |
| `external_core` Recall | `66.71%` `[61.82, 71.20]` | `68.34%` `[50.81, 73.23]` | `70.52%` | `77.72%` |
| `external_core` FPR | `0.96%` `[0.48, 1.44]` | `0.96%` `[0.54, 1.44]` | `1.14%` (`19/1665`) | `1.98%` (`33/1665`) |
| `external_core` Precision | `96.84%` `[95.20, 98.41]` | `96.92%` `[95.40, 98.30]` | `96.47%` | `94.55%` |
| `external_core` Macro-F1 | `85.83%` `[84.24, 87.49]` | `86.57%` `[85.01, 88.15]` | `87.39%` | `89.81%` |
| `longctx_indirect` Recall / FPR | `1.33%` / `2.67%` | **`74.67%`** / **`0.00%`** | **`74.67%`** / **`0.00%`** | `79.33%` / `0.00%` |
| `longctx_indirect` AUROC / Macro-F1 | `0.5226` / `34.16%` | **`0.9510`** / **`87.13%`** | `0.9510` / `87.13%` | `0.9510` / `89.56%` |
| `unicode_obfuscated` Recall / FPR | `11.33%` / `0.00%` | **`83.33%`** / `1.33%` | **`85.33%`** / `1.33%` | `89.33%` / `1.33%` |
| `unicode_obfuscated` AUROC / Macro-F1 | `0.7450` / `44.82%` | **`0.9624`** / **`90.95%`** | `0.9624` / `91.96%` | `0.9624` / `93.99%` |

## Per-source breakdown

| Source | Pos / Neg | DeBERTa-v3 (`0.9945`) | PyTorch FP32 (`0.9820`) | PyTorch FP32 (`0.9540`) | ONNX INT8 CPU (`0.9913`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cve` (benign) | `0 / 250` | FPR `0.00%` | FPR `0.00%` | FPR `0.00%` | FPR `0.00%` |
| `github_readmes` (benign) | `0 / 247` | FPR `1.62%` | FPR `0.00%` | FPR `0.00%` | FPR `0.00%` |
| `sql` (benign) | `0 / 250` | FPR `0.00%` | FPR `0.00%` | FPR `0.00%` | FPR `0.00%` |
| `system_prompts` (benign) | `0 / 200` | FPR `1.00%` | FPR `0.00%` | FPR `0.00%` | FPR `0.00%` |
| `yaml` (benign) | `0 / 163` | FPR `2.45%` | FPR `0.00%` | FPR `0.00%` | FPR `0.00%` |
| `cyberec` | `552 / 390` | R `79.71%` · FPR `1.28%` | R `78.08%` · FPR `3.33%` | R `79.53%` · FPR `3.59%` | R `78.44%` · FPR `3.33%` |
| `deepset` | `60 / 56` | R `30.00%` · FPR `0.00%` | R `28.33%` · FPR `3.57%` | R `33.33%` · FPR `5.36%` | R `28.33%` · FPR `3.57%` |
| `jbb` | `84 / 78` | R `0.00%` · FPR `0.00%` | R `25.00%` · FPR `1.28%` | R `29.76%` · FPR `2.56%` | R `23.81%` · FPR `1.28%` |
| `slfgod` | `40 / 31` | R `82.50%` · FPR `3.23%` | R `85.00%` · FPR `0.00%` | R `87.50%` · FPR `0.00%` | R `87.50%` · FPR `0.00%` |
| `longctx_indirect` | `150 / 0` | R `1.33%` | R `74.67%` | R `74.67%` | R `76.67%` |
| `github_readmes_longctx` | `0 / 150` | FPR `2.67%` | FPR `0.00%` | FPR `0.00%` | FPR `0.00%` |
| `unicode_unicode_tags` | `50 / 0` | R `2.00%` | R `80.00%` | R `80.00%` | R `82.00%` |
| `unicode_variation_selectors` | `50 / 0` | R `24.00%` | R `86.00%` | R `86.00%` | R `84.00%` |
| `unicode_zero_width` | `50 / 0` | R `8.00%` | R `84.00%` | R `84.00%` | R `82.00%` |
| `unicode_benign_dev` | `0 / 150` | FPR `0.00%` | FPR `1.33%` | FPR `1.33%` | FPR `1.33%` |

## Parity & latency

- **ONNX FP32 CUDA** ([json](../results/final_onnx_fp32_cuda_eval.json)): `86.57%` Macro-F1 (`100.000%`), `0.9744` AUROC.
- **ONNX FP16 CUDA** ([json](../results/final_onnx_fp16_cuda_eval.json)): `86.75%` Macro-F1 (`100.211%`), `0.9744` AUROC.
- **ONNX INT8 CPU** ([json](../results/final_onnx_int8_cpu_eval.json)): `86.69%` Macro-F1 (`100.140%` @ `tau=0.9913`; `87.40%` / `100.004%` @ `tau=0.9540`); Ryzen 9 5950X `323.59 ms` (`512`/`674` tok, `2.10 docs/s` b16), `4,437.35 ms` (`2048`/`2,438` tok, `0.20 docs/s`) p50.
- **PyTorch FP32** ([json](../results/final_pytorch_fp32_eval.json)): RTX 3090 `18.91 ms` (`512`) / `86.15 ms` (`2048`) p50 (`2,332.3 MiB` VRAM, `1,471.6 MiB` RSS); Ryzen 9 5950X `626.81 ms` / `3,083.12 ms` p50 (`2,787.4 MiB` RSS).
