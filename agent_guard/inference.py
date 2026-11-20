"""Unified Python inference wrapper for Track A (agent-guard-longctx).

Supports both PyTorch/Transformers (`backend="torch"`) and PyTorch-free
ONNX Runtime (`backend="onnx"`, using only `tokenizers` + `onnxruntime` + `numpy`).
"""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_guard.normalizer import normalize_text

_POS_KEYWORDS = ("INJECTION", "UNSAFE", "MALICIOUS", "JAILBREAK", "ATTACK", "POSITIVE")
_NEG_KEYWORDS = ("SAFE", "BENIGN", "LEGIT", "CLEAN", "NEGATIVE")


@dataclass(slots=True)
class GuardResult:
    """Structured inference result returned by AgentGuardClassifier."""

    is_injection: bool
    score: float
    label: str
    threshold: float
    normalized: bool
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _logits_to_pos_probs(logits: np.ndarray, pos_index: int) -> np.ndarray:
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


def _infer_positive_label_index(id2label: dict[int, str] | None, num_labels: int) -> int:
    if num_labels == 1:
        return 0
    if id2label:
        for idx, name in sorted(id2label.items()):
            if any(kw in name.upper() for kw in _POS_KEYWORDS):
                return idx
        for idx, name in sorted(id2label.items()):
            if any(kw in name.upper() for kw in _NEG_KEYWORDS) and num_labels == 2:
                return 1 - idx
    return 1 if num_labels >= 2 else 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_threshold(model_dir: Path | None, config_dict: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    meta = config_dict.get("agent_guard_metadata") or {}
    if meta.get("recommended_threshold") is not None:
        return float(meta["recommended_threshold"])
    if model_dir is not None:
        cal = _read_json(model_dir / "calibration.json")
        for key in ("recommended_threshold_fpr_1pct", "threshold", "recommended_threshold"):
            if cal.get(key) is not None:
                return float(cal[key])
    return 0.5


def _resolve_labels(config_dict: dict[str, Any]) -> tuple[dict[int, str], int, str, str]:
    raw_id2label = config_dict.get("id2label")
    num_labels = int(config_dict.get("num_labels", len(raw_id2label) if raw_id2label else 2))
    if raw_id2label:
        id2label = {int(k): str(v) for k, v in raw_id2label.items()}
    else:
        id2label = {0: "SAFE", 1: "INJECTION"}
    pos_idx = _infer_positive_label_index(id2label, num_labels)
    neg_idx = 1 - pos_idx if num_labels == 2 and (1 - pos_idx) in id2label else 0
    pos_label = id2label.get(pos_idx, "INJECTION")
    neg_label = id2label.get(neg_idx, "SAFE")
    return id2label, pos_idx, pos_label, neg_label


def _extract_token_str(val: Any, default: str) -> str:
    if isinstance(val, str) and val:
        return val
    if isinstance(val, dict) and isinstance(val.get("content"), str):
        return val["content"]
    return default


class AgentGuardClassifier:
    """Unified Prompt Injection / Indirect Agent-Input Guard Classifier."""

    def __init__(
        self,
        model_path: str | Path,
        backend: str = "auto",
        onnx_file: str = "onnx/model_quantized.onnx",
        threshold: float | None = None,
        max_length: int = 4096,
        normalize: bool = True,
        device: str = "cpu",
        threads: int = 4,
    ) -> None:
        if max_length < 4:
            raise ValueError(f"max_length must be >= 4, got {max_length}")
        if backend not in ("auto", "torch", "onnx"):
            raise ValueError(f"Unsupported backend={backend!r}; expected 'auto', 'torch', or 'onnx'")

        self.model_path = str(model_path)
        self.model_dir: Path | None = Path(model_path) if Path(model_path).exists() else None
        self.onnx_file = onnx_file
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.device_str = str(device)
        self.threads = int(threads)

        config_dict = _read_json(self.model_dir / "config.json") if self.model_dir is not None else {}

        resolved_backend, resolved_onnx_path = self._resolve_backend(backend, onnx_file)
        self.backend = resolved_backend

        if self.backend == "onnx":
            self._init_onnx(resolved_onnx_path, config_dict)
        else:
            self._init_torch(config_dict)

        self.id2label, self.pos_index, self.pos_label, self.neg_label = _resolve_labels(self._config_dict)
        self.threshold = _resolve_threshold(self.model_dir, self._config_dict, threshold)

    def _resolve_backend(self, backend: str, onnx_file: str) -> tuple[str, Path | None]:
        def _find_onnx() -> Path | None:
            cand = Path(onnx_file)
            if cand.is_absolute() and cand.is_file():
                return cand
            if self.model_dir is not None:
                rel = self.model_dir / onnx_file
                if rel.is_file():
                    return rel
                if onnx_file == "onnx/model_quantized.onnx":
                    fp32_fallback = self.model_dir / "onnx" / "model.onnx"
                    if fp32_fallback.is_file():
                        return fp32_fallback
            return None

        if backend == "onnx":
            if self.model_dir is None:
                raise FileNotFoundError(f"ONNX model directory does not exist: {self.model_path}")
            onnx_path = self.model_dir / onnx_file if not Path(onnx_file).is_absolute() else Path(onnx_file)
            if not onnx_path.is_file():
                raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
            return "onnx", onnx_path

        if backend == "torch":
            return "torch", None

        # backend == "auto": prefer PyTorch-free ONNX if model_dir has ONNX + tokenizer.json + ort
        onnx_path = _find_onnx()
        has_ort = importlib.util.find_spec("onnxruntime") is not None
        has_tok = self.model_dir is not None and (self.model_dir / "tokenizer.json").is_file()
        if onnx_path is not None and has_ort and has_tok:
            return "onnx", onnx_path
        return "torch", None

    def _init_onnx(self, onnx_path: Path | None, config_dict: dict[str, Any]) -> None:
        if onnx_path is None or self.model_dir is None:
            raise FileNotFoundError(f"Cannot initialize ONNX backend from {self.model_path}")
        tok_json = self.model_dir / "tokenizer.json"
        if not tok_json.is_file():
            raise FileNotFoundError(f"Missing tokenizer.json in {self.model_dir}")

        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._config_dict = config_dict
        self.onnx_path = onnx_path

        tok_cfg = _read_json(self.model_dir / "tokenizer_config.json")
        pad_token_str = _extract_token_str(tok_cfg.get("pad_token"), "[PAD]")

        tokenizer = Tokenizer.from_file(str(tok_json))
        pad_id = tokenizer.token_to_id(pad_token_str)
        if pad_id is None:
            pad_id = config_dict.get("pad_token_id")
        if pad_id is None:
            pad_id = 0
        self._pad_token_id = int(pad_id)

        # Keep an untruncated/unpadded clone for sliding-window chunking
        self._raw_tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self._raw_tokenizer.no_truncation()
        self._raw_tokenizer.no_padding()

        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(pad_id=self._pad_token_id, pad_token=pad_token_str)
        self._tokenizer = tokenizer

        empty_ids = self._raw_tokenizer.encode("", add_special_tokens=True).ids
        if len(empty_ids) >= 2:
            self._prefix_ids = [int(empty_ids[0])]
            self._suffix_ids = [int(x) for x in empty_ids[1:]]
        else:
            cls_id = config_dict.get("cls_token_id") or config_dict.get("bos_token_id")
            sep_id = config_dict.get("sep_token_id") or config_dict.get("eos_token_id")
            self._prefix_ids = [int(cls_id)] if cls_id is not None else []
            self._suffix_ids = [int(sep_id)] if sep_id is not None else []

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self.threads
        sess_opts.inter_op_num_threads = 1
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device_str.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
        self._input_names = [inp.name for inp in self._session.get_inputs()]

    def _init_torch(self, config_dict: dict[str, Any]) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        if self.device_str.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        self._device = torch.device(self.device_str)

        cfg = AutoConfig.from_pretrained(self.model_path)
        self._config_dict = config_dict or cfg.to_dict()
        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._hf_model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self._device)
        self._hf_model.eval()

        pad_id = getattr(self._hf_tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(cfg, "pad_token_id", 0) or 0
        self._pad_token_id = int(pad_id)

        empty_ids = self._hf_tokenizer.encode("", add_special_tokens=True)
        if len(empty_ids) >= 2:
            self._prefix_ids = [int(empty_ids[0])]
            self._suffix_ids = [int(x) for x in empty_ids[1:]]
        else:
            self._prefix_ids = []
            self._suffix_ids = []

    def _preprocess(self, texts: list[str]) -> list[str]:
        if not self.normalize:
            return texts
        return [normalize_text(t) for t in texts]

    def _forward_arrays(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            feed_all = {
                "input_ids": np.asarray(input_ids, dtype=np.int64),
                "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            }
            feed = {k: feed_all[k] for k in self._input_names if k in feed_all}
            logits = self._session.run(None, feed)[0]
            return _logits_to_pos_probs(logits, self.pos_index)

        torch = self._torch
        enc = {
            "input_ids": torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).to(self._device),
            "attention_mask": torch.from_numpy(np.asarray(attention_mask, dtype=np.int64)).to(self._device),
        }
        with torch.inference_mode():
            logits = self._hf_model(**enc).logits.detach().cpu().numpy()
        return _logits_to_pos_probs(logits, self.pos_index)

    def _score_texts_raw(self, proc_texts: list[str]) -> np.ndarray:
        if not proc_texts:
            return np.empty((0,), dtype=np.float64)
        if self.backend == "onnx":
            encs = self._tokenizer.encode_batch(proc_texts)
            input_ids = np.asarray([e.ids for e in encs], dtype=np.int64)
            attention_mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
            return self._forward_arrays(input_ids, attention_mask)

        enc = self._hf_tokenizer(
            proc_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return self._forward_arrays(enc["input_ids"], enc["attention_mask"])

    def _encode_untruncated_ids(self, proc_text: str) -> list[int]:
        if self.backend == "onnx":
            return [int(x) for x in self._raw_tokenizer.encode(proc_text, add_special_tokens=False).ids]
        return [int(x) for x in self._hf_tokenizer.encode(proc_text, add_special_tokens=False)]

    def _make_result(self, score: float, latency_ms: float) -> GuardResult:
        score_f = float(score)
        is_inj = bool(score_f >= self.threshold)
        return GuardResult(
            is_injection=is_inj,
            score=score_f,
            label=self.pos_label if is_inj else self.neg_label,
            threshold=self.threshold,
            normalized=self.normalize,
            latency_ms=float(latency_ms),
        )

    def predict(self, text: str) -> GuardResult:
        """Classify a single text input."""
        t0 = time.perf_counter()
        proc = self._preprocess([text])
        prob = float(self._score_texts_raw(proc)[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._make_result(prob, elapsed_ms)

    def __call__(self, text: str) -> GuardResult:
        return self.predict(text)

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[GuardResult]:
        """Classify a list of texts in batches."""
        if not texts:
            return []
        results: list[GuardResult] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            t0 = time.perf_counter()
            proc = self._preprocess(batch)
            probs = self._score_texts_raw(proc)
            per_item_ms = ((time.perf_counter() - t0) * 1000.0) / len(batch)
            for p in probs:
                results.append(self._make_result(float(p), per_item_ms))
        return results

    def predict_sliding_window(
        self,
        text: str,
        chunk_stride: int | None = None,
        batch_size: int = 8,
    ) -> GuardResult:
        """Score arbitrarily long documents (> max_length) via overlapping token windows."""
        t0 = time.perf_counter()
        proc_text = self._preprocess([text])[0]
        content_ids = self._encode_untruncated_ids(proc_text)
        special_budget = len(self._prefix_ids) + len(self._suffix_ids)
        window_size = max(1, self.max_length - special_budget)

        if len(content_ids) <= window_size:
            prob = float(self._score_texts_raw([proc_text])[0])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return self._make_result(prob, elapsed_ms)

        stride = int(chunk_stride) if chunk_stride is not None else max(1, window_size // 2)
        if stride < 1:
            raise ValueError(f"chunk_stride must be >= 1, got {stride}")

        windows: list[list[int]] = []
        pos = 0
        n = len(content_ids)
        while pos < n:
            end = min(n, pos + window_size)
            windows.append(self._prefix_ids + content_ids[pos:end] + self._suffix_ids)
            if end == n:
                break
            pos += stride

        max_prob = 0.0
        for start in range(0, len(windows), batch_size):
            chunk_batch = windows[start : start + batch_size]
            max_len = max(len(w) for w in chunk_batch)
            input_ids = np.full((len(chunk_batch), max_len), self._pad_token_id, dtype=np.int64)
            attention_mask = np.zeros((len(chunk_batch), max_len), dtype=np.int64)
            for i, seq in enumerate(chunk_batch):
                input_ids[i, : len(seq)] = seq
                attention_mask[i, : len(seq)] = 1
            probs = self._forward_arrays(input_ids, attention_mask)
            max_prob = max(max_prob, float(np.max(probs)))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._make_result(max_prob, elapsed_ms)
