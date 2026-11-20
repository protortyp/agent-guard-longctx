#!/usr/bin/env python3
"""Zero-cost Model Soup: linear weight averaging of compatible FP32 checkpoints.

Averages FP32 state_dicts of ModernBertForSequenceClassification checkpoints
(trained from the same answerdotai/ModernBERT-base@8949b909 base with pooling=cls)
in FP64 -> FP32, saves model.safetensors + config.json + tokenizer, verifies
disk reload, and calibrates thresholds on dev_internal_v3.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_guard.train import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    BucketedBatchSampler,
    TokenizedGuardDataset,
    evaluate_dev,
    load_jsonl,
    make_collate_fn,
)

COMPAT_CONFIG_KEYS = (
    "architectures",
    "model_type",
    "classifier_pooling",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "vocab_size",
    "id2label",
    "label2id",
)


def validate_and_load_configs(ckpt_dirs: list[Path]) -> dict[str, Any]:
    """Ensure all checkpoints share compatible architecture, pooling, and labels."""
    base_cfg: dict[str, Any] | None = None
    for ckpt in ckpt_dirs:
        cfg_path = ckpt / "config.json"
        sf_path = ckpt / "model.safetensors"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {ckpt}")
        if not sf_path.is_file():
            raise FileNotFoundError(f"Missing model.safetensors in {ckpt}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if base_cfg is None:
            base_cfg = cfg
        else:
            for key in COMPAT_CONFIG_KEYS:
                if cfg.get(key) != base_cfg.get(key):
                    raise ValueError(
                        f"Incompatible config key '{key}' in {ckpt}: "
                        f"{cfg.get(key)!r} != {base_cfg.get(key)!r}"
                    )
    assert base_cfg is not None
    return base_cfg


def merge_state_dicts_fp64(
    ckpt_dirs: list[Path],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Load FP32 state_dicts on CPU and compute weighted linear average in FP64 -> FP32."""
    state_dicts = [load_file(str(ckpt / "model.safetensors"), device="cpu") for ckpt in ckpt_dirs]
    ref_keys = sorted(state_dicts[0].keys())
    for ckpt, sd in zip(ckpt_dirs[1:], state_dicts[1:]):
        keys = sorted(sd.keys())
        if keys != ref_keys:
            raise ValueError(f"State dict key mismatch in {ckpt}")
        for k in ref_keys:
            if sd[k].shape != state_dicts[0][k].shape:
                raise ValueError(
                    f"Shape mismatch for '{k}' in {ckpt}: {sd[k].shape} != {state_dicts[0][k].shape}"
                )
            if sd[k].dtype != state_dicts[0][k].dtype:
                raise ValueError(
                    f"Dtype mismatch for '{k}' in {ckpt}: {sd[k].dtype} != {state_dicts[0][k].dtype}"
                )

    merged: dict[str, torch.Tensor] = {}
    for k in ref_keys:
        t0 = state_dicts[0][k]
        if t0.is_floating_point():
            if t0.dtype != torch.float32:
                raise ValueError(f"Expected float32 master tensor for '{k}', got {t0.dtype}")
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for w, sd in zip(weights, state_dicts):
                acc.add_(sd[k].to(torch.float64), alpha=float(w))
            merged[k] = acc.to(torch.float32).contiguous()
        else:
            for ckpt, sd in zip(ckpt_dirs[1:], state_dicts[1:]):
                if not torch.equal(sd[k], t0):
                    raise ValueError(f"Non-floating tensor '{k}' differs in {ckpt}")
            merged[k] = t0.clone().contiguous()
    return merged


def save_soup_checkpoint(
    merged_sd: dict[str, torch.Tensor],
    ref_ckpt: Path,
    output_dir: Path,
) -> None:
    """Save merged safetensors, config.json, and tokenizer to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged_sd, str(output_dir / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy2(ref_ckpt / "config.json", output_dir / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(ref_ckpt))
    tokenizer.save_pretrained(str(output_dir))


def calibrate_on_dev(
    checkpoint_dir: Path,
    dev_file: Path,
    max_length: int,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, Any]:
    """Reload checkpoint from disk, verify FP32, and evaluate on dev_internal_v3.jsonl."""
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir),
        dtype=torch.float32,
    )
    first_param = next(model.parameters())
    assert first_param.dtype == torch.float32, f"Expected float32, got {first_param.dtype}"
    model.to(device)
    model.eval()

    dev_rows = load_jsonl(dev_file)
    dev_ds = TokenizedGuardDataset(dev_rows, tokenizer, max_length=max_length, neg_weight_hard=1.0)
    dev_sampler = BucketedBatchSampler(dev_ds.lengths, batch_size=batch_size, shuffle=False, seed=42)
    collate_fn = make_collate_fn(tokenizer.pad_token_id, max_length=max_length, pad_to_multiple_of=8)
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=dev_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    dev_metrics = evaluate_dev(model, dev_loader, len(dev_ds), device, use_bf16=use_bf16)
    dev_metrics["checkpoint_tag"] = "soup_merge"
    dev_metrics["dev_rows"] = len(dev_ds)
    return dev_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Linear weight averaging (Model Soup) for FP32 ModernBERT checkpoints.")
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True, help="Two or more compatible checkpoint dirs.")
    p.add_argument("--weights", nargs="+", type=float, required=True, help="Mixing weights summing to 1.0.")
    p.add_argument("--output-dir", type=Path, required=True, help="Destination directory for souped checkpoint.")
    p.add_argument("--dev-file", type=Path, default=HERE / "datasets" / "dev_internal_v3.jsonl")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true", help="Use bf16 autocast during dev calibration (default: FP32).")
    p.add_argument("--skip-calibration", action="store_true", help="Skip dev calibration (for fast unit testing).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if len(args.checkpoints) < 2:
        raise ValueError("Need at least 2 checkpoints to form a soup.")
    if len(args.checkpoints) != len(args.weights):
        raise ValueError(
            f"Count mismatch: {len(args.checkpoints)} checkpoints vs {len(args.weights)} weights."
        )
    if any(w < 0.0 for w in args.weights):
        raise ValueError(f"Weights must be non-negative, got {args.weights}")
    weight_sum = sum(args.weights)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0 (got {weight_sum:.9f})")

    t0 = time.perf_counter()
    base_cfg = validate_and_load_configs(args.checkpoints)
    merged_sd = merge_state_dicts_fp64(args.checkpoints, args.weights)
    save_soup_checkpoint(merged_sd, args.checkpoints[0], args.output_dir)

    reloaded_sd = load_file(str(args.output_dir / "model.safetensors"), device="cpu")
    assert sorted(reloaded_sd.keys()) == sorted(merged_sd.keys())
    for k, v in merged_sd.items():
        assert torch.equal(reloaded_sd[k], v), f"Reload mismatch on tensor '{k}'"

    cal_payload: dict[str, Any] = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "pooling": base_cfg.get("classifier_pooling", "cls"),
        "master_dtype": "float32",
        "max_length": args.max_length,
        "normalizer": "agent_guard.normalizer:normalize_text",
        "best_checkpoint_tag": "soup_merge",
        "soup_sources": [
            {"checkpoint": str(ckpt), "weight": float(w)}
            for ckpt, w in zip(args.checkpoints, args.weights)
        ],
        "dev_file": str(args.dev_file),
    }

    if not args.skip_calibration:
        device = torch.device(args.device)
        dev_metrics = calibrate_on_dev(
            checkpoint_dir=args.output_dir,
            dev_file=args.dev_file,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            device=device,
            use_bf16=args.bf16,
        )
        cal_payload["recommended_threshold_fpr_1pct"] = dev_metrics["tau_fpr_1pct"]
        cal_payload["recommended_threshold_fpr_0_5pct"] = dev_metrics["tau_fpr_0_5pct"]
        cal_payload["tau_dev_fpr1pct"] = dev_metrics["tau_fpr_1pct"]
        cal_payload["tau_dev_fpr0_5pct"] = dev_metrics["tau_fpr_0_5pct"]
        cal_payload["dev_metrics"] = dev_metrics
        print(
            f"[soup calibrated] {args.output_dir.name} | "
            f"dev AUROC={dev_metrics['auroc']:.4f} | "
            f"R@1%={dev_metrics['recall_at_fpr_1pct']:.4f} (tau={dev_metrics['tau_fpr_1pct']:.6f}) | "
            f"R@0.5%={dev_metrics['recall_at_fpr_0_5pct']:.4f} (tau={dev_metrics['tau_fpr_0_5pct']:.6f})",
            flush=True,
        )

    (args.output_dir / "calibration.json").write_text(
        json.dumps(cal_payload, indent=2) + "\n", encoding="utf-8"
    )
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"Saved soup checkpoint to {args.output_dir} in {elapsed}s", flush=True)


if __name__ == "__main__":
    main()
