"""Drop-in `vLLM Semantic Router` Jailbreak / Prompt-Guard middleware plugin.

Provides `AgentGuardJailbreakPlugin` with:
  - `classify(text: str) -> dict`
  - `check_messages(messages: list[dict], scan_tool_outputs: bool = True) -> GuardDecision`

Designed to detect both direct user prompt injections/jailbreaks and indirect
prompt injections hidden inside `role == "tool"` / `role == "function"` outputs,
with deterministic Unicode pre-normalization and PyTorch-free ONNX Runtime support.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_guard.inference import AgentGuardClassifier, GuardResult


@dataclass(slots=True)
class GuardDecision:
    """Decision returned by `AgentGuardJailbreakPlugin.check_messages`."""

    allowed: bool
    blocked: bool
    action: str
    max_score: float
    threshold: float
    flagged_indices: list[int]
    flagged_roles: list[str]
    message_results: list[dict[str, Any]]
    total_latency_ms: float
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_val = item.get("text") or item.get("content")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(parts)
    return ""


class AgentGuardJailbreakPlugin:
    """vLLM Semantic Router compatible jailbreak & indirect-injection guard plugin."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float | None = None,
        max_length: int = 4096,
        normalize: bool = True,
        backend: str = "auto",
        onnx_file: str = "onnx/model_quantized.onnx",
        device: str = "cpu",
        match_type: str = "full",
        chunk_stride: int | None = None,
        scan_roles: tuple[str, ...] = ("user",),
        threads: int = 4,
    ) -> None:
        if match_type not in ("full", "chunked_sliding_window"):
            raise ValueError(f"Unsupported match_type={match_type!r}")
        normalized_backend = "torch" if backend == "hf" else backend
        self.match_type = match_type
        self.chunk_stride = chunk_stride
        self.scan_roles = tuple(str(r).lower() for r in scan_roles)
        self._classifier = AgentGuardClassifier(
            model_path=model_path,
            backend=normalized_backend,
            onnx_file=onnx_file,
            threshold=threshold,
            max_length=max_length,
            normalize=normalize,
            device=device,
            threads=threads,
        )
        self.threshold = self._classifier.threshold
        self.backend = self._classifier.backend

    def _score_one(self, text: str) -> GuardResult:
        if self.match_type == "chunked_sliding_window":
            return self._classifier.predict_sliding_window(text, chunk_stride=self.chunk_stride)
        return self._classifier.predict(text)

    def classify(self, text: str) -> dict[str, Any]:
        """Classify a single string for prompt-injection / jailbreak risk."""
        res = self._score_one(text)
        return {
            "is_jailbreak": res.is_injection,
            "is_injection": res.is_injection,
            "label": res.label,
            "score": res.score,
            "confidence": float(max(res.score, 1.0 - res.score)),
            "threshold": res.threshold,
            "normalized": res.normalized,
            "latency_ms": res.latency_ms,
        }

    def check_messages(
        self,
        messages: list[dict[str, Any]],
        scan_tool_outputs: bool = True,
    ) -> GuardDecision:
        """Inspect a chat-completion message list for direct & indirect injections."""
        t0 = time.perf_counter()
        active_roles = set(self.scan_roles)
        if scan_tool_outputs:
            active_roles.update({"tool", "function"})

        target_indices: list[int] = []
        target_texts: list[str] = []
        for idx, msg in enumerate(messages):
            role = str(msg.get("role", "")).lower()
            if role not in active_roles:
                continue
            text = _extract_message_text(msg)
            if text.strip():
                target_indices.append(idx)
                target_texts.append(text)

        scored_by_idx: dict[int, GuardResult] = {}
        if target_texts:
            if self.match_type == "full":
                batch_res = self._classifier.predict_batch(target_texts)
                for idx, res in zip(target_indices, batch_res):
                    scored_by_idx[idx] = res
            else:
                for idx, text in zip(target_indices, target_texts):
                    scored_by_idx[idx] = self._score_one(text)

        message_results: list[dict[str, Any]] = []
        flagged_indices: list[int] = []
        flagged_roles: list[str] = []
        max_score = 0.0

        for idx, msg in enumerate(messages):
            role = str(msg.get("role", "")).lower()
            res = scored_by_idx.get(idx)
            if res is None:
                message_results.append(
                    {
                        "index": idx,
                        "role": role,
                        "scanned": False,
                        "is_injection": False,
                        "score": 0.0,
                        "label": self._classifier.neg_label,
                    }
                )
                continue

            max_score = max(max_score, res.score)
            if res.is_injection:
                flagged_indices.append(idx)
                flagged_roles.append(role)
            message_results.append(
                {
                    "index": idx,
                    "role": role,
                    "scanned": True,
                    "is_injection": res.is_injection,
                    "score": res.score,
                    "label": res.label,
                    "latency_ms": res.latency_ms,
                }
            )

        blocked = len(flagged_indices) > 0
        total_ms = (time.perf_counter() - t0) * 1000.0
        reason = (
            f"Blocked indirect/direct injection at message indices {flagged_indices} "
            f"(roles={flagged_roles}, max_score={max_score:.4f} >= threshold={self.threshold:.4f})"
            if blocked
            else None
        )
        return GuardDecision(
            allowed=not blocked,
            blocked=blocked,
            action="block" if blocked else "allow",
            max_score=float(max_score),
            threshold=float(self.threshold),
            flagged_indices=flagged_indices,
            flagged_roles=flagged_roles,
            message_results=message_results,
            total_latency_ms=float(total_ms),
            reason=reason,
        )
