"""Drop-in ProtectAI `llm-guard` input scanner adapter for Track A.

Implements the exact `llm_guard.input_scanners.base.Scanner` interface
(`scan(prompt: str) -> tuple[str, bool, float]`) without requiring `llm-guard`
to be installed at import time.
"""

from __future__ import annotations

from pathlib import Path

from agent_guard.inference import AgentGuardClassifier

try:
    from llm_guard.input_scanners.base import Scanner as _BaseScanner
except ImportError:

    class _BaseScanner:  # type: ignore[no-redef]
        """Duck-typed fallback when `llm-guard` is not installed."""

        def scan(self, prompt: str) -> tuple[str, bool, float]:
            raise NotImplementedError


_VALID_MATCH_TYPES = ("full", "chunked_sliding_window")


class AgentGuardPromptInjection(_BaseScanner):
    """ProtectAI `llm-guard` compatible Prompt Injection scanner backed by Track A.

    Args:
        model_path: Local exported ONNX/HF model directory or HF model ID.
        threshold: Risk score threshold in [0, 1]. `is_valid = score < threshold`.
            If None, reads calibrated threshold from `config.json` / `calibration.json`.
        max_length: Context window length per forward pass (up to 8192).
        normalize_unicode: Apply `agent_guard.normalizer.normalize_text` prior to
            tokenization (stripping invisible chars, decoding Unicode Tags & emoji
            variation-selector smuggling, and selective NFKC).
        match_type: `"full"` (single pass up to `max_length`) or
            `"chunked_sliding_window"` (overlapping token windows for >8192-token docs).
        backend: `"auto"`, `"onnx"` (PyTorch-free), or `"torch"` (alias `"hf"`).
        onnx_file: Relative path to ONNX graph inside `model_path`.
        device: `"cpu"` or `"cuda"`.
        chunk_stride: Token stride for `match_type="chunked_sliding_window"`.
        threads: Intra-op CPU threads for ONNX Runtime.
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float | None = None,
        max_length: int = 4096,
        normalize_unicode: bool = True,
        match_type: str = "full",
        backend: str = "auto",
        onnx_file: str = "onnx/model_quantized.onnx",
        device: str = "cpu",
        chunk_stride: int | None = None,
        threads: int = 4,
    ) -> None:
        if not (1 <= max_length <= 8192):
            raise ValueError(f"max_length must be in [1, 8192], got {max_length}")
        if match_type not in _VALID_MATCH_TYPES:
            raise ValueError(f"Unsupported match_type={match_type!r}; expected one of {_VALID_MATCH_TYPES}")

        normalized_backend = "torch" if backend == "hf" else backend
        self.match_type = match_type
        self.chunk_stride = chunk_stride
        self.normalize_unicode = bool(normalize_unicode)
        self._classifier = AgentGuardClassifier(
            model_path=model_path,
            backend=normalized_backend,
            onnx_file=onnx_file,
            threshold=threshold,
            max_length=max_length,
            normalize=self.normalize_unicode,
            device=device,
            threads=threads,
        )
        self.threshold = self._classifier.threshold
        self.max_length = self._classifier.max_length
        self.backend = self._classifier.backend

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """Scan `prompt` and return `(sanitized_prompt, is_valid, risk_score)`."""
        if not prompt or not prompt.strip():
            return prompt, True, 0.0

        if self.match_type == "chunked_sliding_window":
            result = self._classifier.predict_sliding_window(prompt, chunk_stride=self.chunk_stride)
        else:
            result = self._classifier.predict(prompt)

        risk_score = float(result.score)
        is_valid = bool(risk_score < result.threshold)
        return prompt, is_valid, risk_score
