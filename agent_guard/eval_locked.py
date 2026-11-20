#!/usr/bin/env python3
"""Immutable locked evaluation harness for Track A (agent-guard-longctx).

Supports HuggingFace AutoModelForSequenceClassification (--backend hf) and
ONNX Runtime (--backend onnx) classifiers on the held-out splits in
agent_guard/datasets/held_out/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
DEFAULT_HELD_OUT_DIR = HERE / "datasets" / "held_out"

INVISIBLE_CODEPOINTS = frozenset(
    [
        0x00AD,  # soft hyphen
        0x034F,  # combining grapheme joiner
        0x061C,  # arabic letter mark
        0x180E,  # mongolian vowel separator
        0x200B,  # zero width space
        0x200C,  # zero width non-joiner
        0x200D,  # zero width joiner
        0x200E,  # left-to-right mark
        0x200F,  # right-to-left mark
        0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embeddings/overrides
        0x2060, 0x2061, 0x2062, 0x2063, 0x2064,  # word joiner & invisible operators
        0x2066, 0x2067, 0x2068, 0x2069,  # bidi isolates
        0xFEFF,  # BOM / zero width no-break space
    ]
)


def builtin_normalize_unicode(text: str) -> str:
    """Safe reference normalizer: NFKC + decode/strip tag chars & invisible controls."""
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if 0xE0020 <= cp <= 0xE007E:
            out.append(chr(cp - 0xE0000))
        elif cp in (0xE0001, 0xE007F) or (0xE0000 <= cp <= 0xE007F):
            continue
        elif (0xFE00 <= cp <= 0xFE0F) or (0xE0100 <= cp <= 0xE01EF):
            continue
        elif cp in INVISIBLE_CODEPOINTS:
            continue
        else:
            out.append(ch)
    return unicodedata.normalize("NFKC", "".join(out))


def load_normalizer(spec: str | None) -> Callable[[str], str] | None:
    if spec is None:
        return None
    if spec in ("builtin", "nfkc_strip_invisible"):
        return builtin_normalize_unicode
    if ":" in spec:
        mod_ref, fn_name = spec.split(":", 1)
    else:
        mod_ref, fn_name = spec, "normalize_text"
    mod_path = Path(mod_ref)
    if mod_path.is_file():
        module_spec = importlib.util.spec_from_file_location("custom_normalizer", mod_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot load normalizer module from {mod_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_ref)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_load_held_out(held_out_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = held_out_dir / "held_out_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets: dict[str, list[dict[str, Any]]] = {}
    for fname in ("external_core.jsonl", "longctx_indirect.jsonl", "unicode_obfuscated.jsonl"):
        fpath = held_out_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing held-out split: {fpath}")
        expected_sha = manifest["files"][fname]["sha256"]
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            raise ValueError(f"SHA256 mismatch for {fpath}: expected {expected_sha}, got {actual_sha}")
        rows = [json.loads(line) for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        datasets[fname] = rows
    return manifest, datasets


def logits_to_pos_probs(logits: np.ndarray, pos_index: int) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        return 1.0 / (1.0 + np.exp(-np.clip(arr, -500.0, 500.0)))
    if arr.shape[1] == 1:
        z = np.clip(arr[:, 0], -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    return probs[:, pos_index]


def infer_positive_label_index(id2label: dict[Any, str] | None, num_labels: int, override: int | None) -> int:
    if override is not None:
        return override
    if num_labels == 1:
        return 0
    if id2label:
        norm_map = {int(k): str(v).strip().upper() for k, v in id2label.items()}
        pos_keywords = ("INJECTION", "UNSAFE", "MALICIOUS", "JAILBREAK", "ATTACK", "POSITIVE")
        neg_keywords = ("SAFE", "BENIGN", "LEGIT", "CLEAN", "NEGATIVE")
        for idx, name in sorted(norm_map.items()):
            if any(kw in name for kw in pos_keywords):
                return idx
        for idx, name in sorted(norm_map.items()):
            if any(kw in name for kw in neg_keywords) and num_labels == 2:
                return 1 - idx
    return 1 if num_labels >= 2 else 0


class ClassifierRunner:
    """Unified inference interface for HF and ONNX backends."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.backend = args.backend
        self.normalizer = load_normalizer(args.normalize_unicode)
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.device_str = args.device
        self.metadata: dict[str, Any] = {
            "backend": args.backend,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "device": args.device,
            "dtype": args.dtype,
            "normalize_unicode": args.normalize_unicode,
        }

        if args.backend == "hf":
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(args.threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
            self.torch = torch
            self.device = torch.device(args.device)
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map[args.dtype]
            model_kwargs: dict[str, Any] = {}
            if args.revision:
                model_kwargs["revision"] = args.revision
            cfg = AutoConfig.from_pretrained(args.model, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, **model_kwargs)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                args.model, dtype=torch_dtype, **model_kwargs
            ).to(self.device)
            self.model.eval()
            self.pos_index = infer_positive_label_index(
                getattr(cfg, "id2label", None), getattr(cfg, "num_labels", 2), args.label_index
            )
            self.metadata.update(
                {
                    "model": args.model,
                    "revision": args.revision,
                    "parameters": sum(p.numel() for p in self.model.parameters()),
                    "model_max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
                    "id2label": getattr(cfg, "id2label", None),
                    "positive_label_index": self.pos_index,
                }
            )
        elif args.backend == "onnx":
            import onnxruntime as ort
            from transformers import AutoConfig, AutoTokenizer

            if not args.onnx_path or not args.tokenizer_dir:
                raise ValueError("--backend onnx requires --onnx-path and --tokenizer-dir")
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = args.threads
            sess_opts.inter_op_num_threads = 1
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if args.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(str(args.onnx_path), sess_options=sess_opts, providers=providers)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
            id2label = None
            num_labels = 2
            cfg_path = Path(args.tokenizer_dir) / "config.json"
            if cfg_path.is_file():
                cfg = AutoConfig.from_pretrained(args.tokenizer_dir)
                id2label = getattr(cfg, "id2label", None)
                num_labels = getattr(cfg, "num_labels", 2)
            self.pos_index = infer_positive_label_index(id2label, num_labels, args.label_index)
            self.metadata.update(
                {
                    "onnx_path": str(args.onnx_path),
                    "onnx_size_mib": round(Path(args.onnx_path).stat().st_size / (1024**2), 3),
                    "tokenizer_dir": str(args.tokenizer_dir),
                    "onnx_providers": self.session.get_providers(),
                    "id2label": id2label,
                    "positive_label_index": self.pos_index,
                }
            )
        else:
            raise ValueError(f"Unsupported backend: {args.backend}")

    def _preprocess(self, texts: list[str]) -> list[str]:
        if self.normalizer is None:
            return texts
        return [self.normalizer(t) for t in texts]

    def sync(self) -> None:
        if self.backend == "hf" and self.device_str == "cuda":
            self.torch.cuda.synchronize(self.device)

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        proc = self._preprocess(texts)
        if self.backend == "hf":
            enc = self.tokenizer(
                proc,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self.torch.inference_mode():
                out = self.model(**enc)
                logits = out.logits.detach().cpu().numpy()
            return logits_to_pos_probs(logits, self.pos_index)

        enc = self.tokenizer(
            proc,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed = {k: np.asarray(enc[k], dtype=np.int64) for k in self.input_names if k in enc}
        logits = self.session.run(None, feed)[0]
        return logits_to_pos_probs(logits, self.pos_index)

    def score_texts(self, texts: list[str]) -> np.ndarray:
        all_probs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_probs.append(self.predict_batch(batch))
        self.sync()
        return np.concatenate(all_probs, axis=0) if all_probs else np.empty((0,), dtype=np.float64)


def find_max_recall_at_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.01) -> tuple[float, float, float]:
    """Return (tau, recall, actual_fpr) maximizing Recall subject to FPR <= max_fpr."""
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative examples to calibrate threshold")

    order = np.argsort(-s, kind="mergesort")
    s_desc = s[order]
    y_desc = y[order]

    is_block_end = np.empty(len(s_desc), dtype=bool)
    is_block_end[:-1] = s_desc[:-1] != s_desc[1:]
    is_block_end[-1] = True

    tp_cum = np.cumsum(y_desc == 1)[is_block_end]
    fp_cum = np.cumsum(y_desc == 0)[is_block_end]
    threshs = s_desc[is_block_end]

    fprs = fp_cum / n_neg
    valid = np.where(fprs <= max_fpr + 1e-15)[0]
    if len(valid) == 0:
        return float(s_desc[0]) + 1e-6, 0.0, 0.0
    best_idx = int(valid[-1])
    return float(threshs[best_idx]), float(tp_cum[best_idx] / n_pos), float(fprs[best_idx])


def compute_threshold_metrics(y: np.ndarray, s: np.ndarray, tau: float) -> dict[str, Any]:
    pred = (s >= tau).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    n_pos = tp + fn
    n_neg = fp + tn

    recall = float(tp / n_pos) if n_pos > 0 else None
    fpr = float(fp / n_neg) if n_neg > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (0.0 if n_pos > 0 else None)
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0)) if (n_pos > 0 and n_neg > 0) else None
    auroc = float(roc_auc_score(y, s)) if (n_pos > 0 and n_neg > 0) else None
    auprc = float(average_precision_score(y, s)) if (n_pos > 0 and n_neg > 0) else None

    return {
        "threshold": float(tau),
        "n_total": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_positives": fp,
        "total_benign": n_neg,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "auprc": auprc,
    }


def stratified_bootstrap_cis(
    y: np.ndarray, s: np.ndarray, tau: float, n_boot: int = 1000, seed: int = 42
) -> dict[str, dict[str, float]]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)

    aurocs = np.empty(n_boot, dtype=np.float64)
    auprcs = np.empty(n_boot, dtype=np.float64)
    recalls_1pct = np.empty(n_boot, dtype=np.float64)
    macro_f1s = np.empty(n_boot, dtype=np.float64)
    recalls_tau = np.empty(n_boot, dtype=np.float64)
    precisions_tau = np.empty(n_boot, dtype=np.float64)
    fprs_tau = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        p_draw = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n_draw = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p_draw, n_draw])
        yb = y[idx]
        sb = s[idx]

        aurocs[b] = roc_auc_score(yb, sb)
        auprcs[b] = average_precision_score(yb, sb)
        _, r1, _ = find_max_recall_at_fpr(yb, sb, max_fpr=0.01)
        recalls_1pct[b] = r1

        pred = (sb >= tau).astype(np.int64)
        tp = int(np.sum((pred == 1) & (yb == 1)))
        fp = int(np.sum((pred == 1) & (yb == 0)))
        recalls_tau[b] = tp / len(pos_idx)
        fprs_tau[b] = fp / len(neg_idx)
        precisions_tau[b] = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        macro_f1s[b] = f1_score(yb, pred, average="macro", zero_division=0)

    def summarize(arr: np.ndarray) -> dict[str, float]:
        q025, q975 = np.quantile(arr, [0.025, 0.975])
        return {"ci95_low": float(q025), "ci95_high": float(q975), "std": float(np.std(arr))}

    return {
        "auroc": summarize(aurocs),
        "auprc": summarize(auprcs),
        "recall_at_fpr_1pct": summarize(recalls_1pct),
        "macro_f1_at_tau": summarize(macro_f1s),
        "recall_at_tau": summarize(recalls_tau),
        "precision_at_tau": summarize(precisions_tau),
        "fpr_at_tau": summarize(fprs_tau),
    }


def breakdown_by_key(
    rows: list[dict[str, Any]], y: np.ndarray, s: np.ndarray, tau: float, key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[str(r.get(key, "unknown"))].append(i)
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(groups):
        idx = np.asarray(groups[name], dtype=np.int64)
        out[name] = compute_threshold_metrics(y[idx], s[idx], tau)
    return out


def make_rep_text(target_tokens: int) -> str:
    unit = (
        "## Service Configuration & API Notes\n"
        "Use `POST /v1/authorize` with TLS 1.3 client certificates and verify SHA-256 checksums "
        "before deploying Kubernetes YAML manifests into production clusters.\n"
    )
    # ~40 tokens per unit
    reps = max(1, (target_tokens // 36) + 2)
    return unit * reps


def benchmark_latency(runner: ClassifierRunner) -> dict[str, Any]:
    if runner.backend == "hf" and runner.device_str == "cuda":
        runner.torch.cuda.reset_peak_memory_stats(runner.device)

    profiles: dict[str, Any] = {}
    for label, target_toks in (("512_tokens", 512), ("2048_tokens", 2048)):
        text = make_rep_text(target_toks)
        tok_len = len(runner.tokenizer(text, truncation=True, max_length=runner.max_length)["input_ids"])
        for _ in range(5):
            runner.predict_batch([text])
        runner.sync()

        single_ms: list[float] = []
        for _ in range(25):
            t0 = time.perf_counter()
            runner.predict_batch([text])
            runner.sync()
            single_ms.append((time.perf_counter() - t0) * 1000.0)

        batch16 = [text] * 16
        for _ in range(2):
            runner.predict_batch(batch16)
        runner.sync()
        t0 = time.perf_counter()
        for _ in range(5):
            runner.predict_batch(batch16)
        runner.sync()
        batch_dur = time.perf_counter() - t0

        profiles[label] = {
            "effective_input_tokens": tok_len,
            "single_p50_ms": float(np.median(single_ms)),
            "single_p95_ms": float(np.quantile(single_ms, 0.95)),
            "single_mean_ms": float(np.mean(single_ms)),
            "batch16_docs_per_sec": float((16 * 5) / batch_dur),
        }

    return {
        "device": runner.device_str,
        "gpu": runner.torch.cuda.get_device_name(runner.device)
        if (runner.backend == "hf" and runner.device_str == "cuda")
        else None,
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        "platform": platform.platform(),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "cuda_peak_allocated_mib": float(runner.torch.cuda.max_memory_allocated(runner.device) / (1024**2))
        if (runner.backend == "hf" and runner.device_str == "cuda")
        else None,
        "profiles": profiles,
    }


def measure_isolated_cpu_latency(args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend",
        args.backend,
        "--max-length",
        str(args.max_length),
        "--batch-size",
        str(args.batch_size),
        "--device",
        "cpu",
        "--dtype",
        "float32" if args.backend == "hf" else args.dtype,
        "--threads",
        str(args.threads),
        "--held-out-dir",
        str(args.held_out_dir),
        "--profile-only",
        "--output-json",
        "-",
    ]
    if args.backend == "hf":
        cmd.extend(["--model", args.model])
        if args.revision:
            cmd.extend(["--revision", args.revision])
    else:
        cmd.extend(["--onnx-path", str(args.onnx_path), "--tokenizer-dir", str(args.tokenizer_dir)])
    if args.normalize_unicode:
        cmd.extend(["--normalize-unicode", args.normalize_unicode])
    if args.label_index is not None:
        cmd.extend(["--label-index", str(args.label_index)])

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout)
    return payload["latency"]


def evaluate_all(args: argparse.Namespace) -> dict[str, Any]:
    runner = ClassifierRunner(args)
    if args.profile_only:
        return {"metadata": runner.metadata, "latency": benchmark_latency(runner)}

    held_out_dir = Path(args.held_out_dir)
    manifest, datasets = verify_and_load_held_out(held_out_dir)

    core_rows = datasets["external_core.jsonl"]
    long_rows = datasets["longctx_indirect.jsonl"]
    uni_rows = datasets["unicode_obfuscated.jsonl"]

    core_y = np.asarray([int(r["label"]) for r in core_rows], dtype=np.int64)
    core_s = runner.score_texts([r["text"] for r in core_rows])

    tau_oracle, recall_1pct, fpr_1pct = find_max_recall_at_fpr(core_y, core_s, max_fpr=0.01)
    oracle_core_metrics = compute_threshold_metrics(core_y, core_s, tau_oracle)
    oracle_core_metrics["recall_at_fpr_1pct"] = recall_1pct
    oracle_core_metrics["actual_fpr_at_tau"] = fpr_1pct

    bootstrap_cis = stratified_bootstrap_cis(
        core_y, core_s, tau_oracle, n_boot=args.bootstrap_n, seed=args.bootstrap_seed
    )
    per_source_oracle = breakdown_by_key(core_rows, core_y, core_s, tau_oracle, key="source")

    long_y = np.asarray([int(r["label"]) for r in long_rows], dtype=np.int64)
    long_s = runner.score_texts([r["text"] for r in long_rows])
    long_oracle = compute_threshold_metrics(long_y, long_s, tau_oracle)
    long_by_variant = breakdown_by_key(long_rows, long_y, long_s, tau_oracle, key="source")

    uni_y = np.asarray([int(r["label"]) for r in uni_rows], dtype=np.int64)
    uni_s = runner.score_texts([r["text"] for r in uni_rows])
    uni_oracle = compute_threshold_metrics(uni_y, uni_s, tau_oracle)
    uni_by_scheme = breakdown_by_key(uni_rows, uni_y, uni_s, tau_oracle, key="source")

    result: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": runner.metadata,
        "held_out_manifest_sha256": sha256_file(held_out_dir / "held_out_manifest.json"),
        "held_out_split_shas": {k: v["sha256"] for k, v in manifest["files"].items()},
        "external_core": {
            "oracle_global_threshold": tau_oracle,
            "metrics": oracle_core_metrics,
            "bootstrap_ci95": bootstrap_cis,
            "per_source": per_source_oracle,
        },
        "longctx_indirect": {
            "at_oracle_global_threshold": long_oracle,
            "per_source": long_by_variant,
        },
        "unicode_obfuscated": {
            "at_oracle_global_threshold": uni_oracle,
            "per_source": uni_by_scheme,
        },
    }

    if args.threshold is not None:
        tau_cal = float(args.threshold)
        result["calibrated_threshold_eval"] = {
            "threshold": tau_cal,
            "external_core": compute_threshold_metrics(core_y, core_s, tau_cal),
            "external_core_per_source": breakdown_by_key(core_rows, core_y, core_s, tau_cal, key="source"),
            "longctx_indirect": compute_threshold_metrics(long_y, long_s, tau_cal),
            "unicode_obfuscated": compute_threshold_metrics(uni_y, uni_s, tau_cal),
        }

    if not args.skip_latency:
        result["latency"] = benchmark_latency(runner)
        if args.measure_cpu_latency and args.device != "cpu":
            result["cpu_latency"] = measure_isolated_cpu_latency(args)

    return result


def log_to_trackio(run_name: str, res: dict[str, Any]) -> None:
    import trackio

    core = res["external_core"]["metrics"]
    longm = res["longctx_indirect"]["at_oracle_global_threshold"]
    unim = res["unicode_obfuscated"]["at_oracle_global_threshold"]
    trackio.init(project="agent_guard_eval", name=run_name, config=res["metadata"])
    payload = {
        "core_recall_at_fpr_1pct": core["recall_at_fpr_1pct"],
        "core_macro_f1": core["macro_f1"],
        "core_precision": core["precision"],
        "core_fpr": core["fpr"],
        "core_auroc": core["auroc"],
        "core_auprc": core["auprc"],
        "core_tau": res["external_core"]["oracle_global_threshold"],
        "longctx_recall": longm["recall"],
        "longctx_fpr": longm["fpr"],
        "longctx_macro_f1": longm["macro_f1"],
        "longctx_auroc": longm["auroc"],
        "unicode_recall": unim["recall"],
        "unicode_fpr": unim["fpr"],
        "unicode_macro_f1": unim["macro_f1"],
        "unicode_auroc": unim["auroc"],
    }
    trackio.log(payload)
    trackio.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Locked Track A evaluation harness.")
    p.add_argument("--backend", choices=["hf", "onnx"], required=True)
    p.add_argument("--model", help="HF model ID or local path (required for --backend hf)")
    p.add_argument("--revision", default=None, help="Optional pinned HF revision SHA")
    p.add_argument("--onnx-path", default=None, help="Path to ONNX model (required for --backend onnx)")
    p.add_argument("--tokenizer-dir", default=None, help="Tokenizer dir or HF ID (required for --backend onnx)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--label-index", type=int, default=None)
    p.add_argument(
        "--normalize-unicode",
        nargs="?",
        const="builtin",
        default=None,
        help="Apply Unicode normalizer ('builtin' or 'module_or_path.py:func_name')",
    )
    p.add_argument("--held-out-dir", type=Path, default=DEFAULT_HELD_OUT_DIR)
    p.add_argument("--threshold", type=float, default=None, help="Optional validation-calibrated threshold")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--measure-cpu-latency", action="store_true")
    p.add_argument("--output-json", required=True, help="Output JSON path (or '-' for stdout)")
    p.add_argument("--trackio-run", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "hf" and not args.model:
        raise ValueError("--backend hf requires --model")
    result = evaluate_all(args)
    if args.trackio_run and not args.profile_only:
        log_to_trackio(args.trackio_run, result)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output_json == "-":
        sys.stdout.write(serialized)
    else:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
