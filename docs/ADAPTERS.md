# Adapters & API

## 1. `AgentGuardClassifier`

[`agent_guard/inference.py`](../agent_guard/inference.py):
- `backend="onnx"`: `onnxruntime`, `tokenizers`, `numpy` (no PyTorch)
- `backend="torch"`: `torch`, `transformers`
- `backend="auto"`: prefers ONNX when `onnx/model_quantized.onnx`, `tokenizer.json`, `onnxruntime` exist

Reads `agent_guard_metadata.recommended_threshold` (`0.953966`) and `id2label` from `config.json`; runs [`normalize_text`](../agent_guard/normalizer.py). `predict_sliding_window(text, chunk_stride=2048)` handles `>4,096` tokens (up to `8,192`).

## 2. `llm-guard` (`AgentGuardPromptInjection`)

[`agent_guard/adapters/llm_guard_adapter.py`](../agent_guard/adapters/llm_guard_adapter.py) implements `llm_guard.input_scanners.base.Scanner` (`scan(prompt: str) -> tuple[str, bool, float]`):

```python
from huggingface_hub import snapshot_download
from agent_guard.adapters import AgentGuardPromptInjection

model_dir = snapshot_download(
    repo_id="christianengel/agent-guard-longctx",
    allow_patterns=["config.json", "tokenizer*.json", "special_tokens_map.json", "onnx/model_quantized.onnx"],
)

scanner = AgentGuardPromptInjection(
    model_path=model_dir,
    match_type="chunked_sliding_window",  # or "full"
    backend="onnx",
    onnx_file="onnx/model_quantized.onnx",
    normalize_unicode=True,
    device="cpu",
)

sanitized_prompt, is_valid, risk_score = scanner.scan(untrusted_tool_output)
```

- `is_valid = risk_score < threshold`; empty inputs return `(prompt, True, 0.0)`.

## 3. `vLLM Semantic Router` (`AgentGuardJailbreakPlugin`)

[`agent_guard/adapters/vllm_semantic_router_adapter.py`](../agent_guard/adapters/vllm_semantic_router_adapter.py) scans user + `role in {"tool", "function"}`:

```python
from agent_guard.adapters import AgentGuardJailbreakPlugin

plugin = AgentGuardJailbreakPlugin(
    model_path=model_dir,
    backend="onnx",
    max_length=4096,
    scan_roles=("user",),
)

decision = plugin.check_messages(
    [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Check the repository README."},
        {"role": "tool", "content": fetched_readme_text},
    ],
    scan_tool_outputs=True,
)
print(decision.to_dict())
```

Returns `GuardDecision` (`allowed`, `blocked`, `action`, `max_score`, `threshold`, `flagged_indices`, `flagged_roles`, `message_results`, `total_latency_ms`, `reason`). `plugin.classify(text)` scores one string.
