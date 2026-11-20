#!/usr/bin/env python3
"""Zero-leakage, license-audited data curation pipeline (v2 / v3) for Track A (agent-guard-longctx).

Default (`--version v3`) builds:
  - agent_guard/datasets/train_v3.jsonl
  - agent_guard/datasets/dev_internal_v3.jsonl
  - agent_guard/datasets/train_provenance_manifest_v3.json
  - agent_guard/datasets/split_overlap_audit_v3.json

Key v3 improvements over v2 (targeting exp02 failure modes):
  1. Conversational/instructional FP cluster (`cyberec` 16/16 FPs):
     - Adds 5,000 diverse benign user prompts & multi-turn instructions from
       HuggingFaceH4/ultrachat_200k (MIT, train_sft) and +1,500 additional prompts
       from OpenAssistant/oasst1 (Apache-2.0, train -> 3,500 total).
     - Marks ultrachat_200k, oasst1, yanismiraoui_v2 benign, slabs benign, spml benign,
       jailbreak_hub benign, cve_pre2024, github_yaml_ci, and ericlu_sys with
       `is_hard_negative: true` so `--neg-weight-hard` protects both developer
       hard negatives AND benign conversational/roleplay/trigger-word prompts.
  2. Long-context indirect & Unicode-decoded inline recall (`longctx_indirect` & `unicode_obfuscated`):
     - Expands indirect_injection_composite from 4,000 -> 6,500 train pairs (500 dev pairs),
       paired with 6,500 (500 dev) length-matched benign documents:
       * 2,500 short/medium inline carriers (120-1,200 tokens): 50% NO synthetic wrapper tag
         (natural newline/sentence continuation) + 50% varied realistic wrappers.
       * 4,000 deep long-context carriers (1,200-4,040 tokens): pinned ModernBERT-base
         token-budgeted so 100% fit inside max_length=4096 (0 truncated!), explicitly
         covering the 2,500-3,900 token deep tail with naturalistic inline + varied wrappers.
  3. Full clean PASS_TRAIN direct injection & jailbreak coverage:
     - Includes all clean PASS_TRAIN direct injections (yanismiraoui_v2, spml Gandalf+benign,
       slabs, gandalf, shane_corpus, darkknight, aestera) and jailbreaks (jailbreak_hub,
       jackhhao, jailbreak_success).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_guard.normalizer import normalize_text  # noqa: E402

SEED = 42
MODERNBERT_MODEL = "answerdotai/ModernBERT-base"
MODERNBERT_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"

# v2 budgets
MAX_CARRIER_TOKENS = 3100
MIN_CARRIER_TOKENS = 650
MAX_ATTACK_TOKENS = 300
MIN_ATTACK_TOKENS = 10
MAX_INDIRECT_TOTAL_TOKENS = 3500

# v3 budgets (zero-truncation at max_length=4096)
V3_SHORT_MIN_TOKENS = 120
V3_SHORT_MAX_TOKENS = 1200
V3_SHORT_CARRIER_MIN_TOKENS = 100
V3_SHORT_CARRIER_MAX_TOKENS = 1000
V3_SHORT_MAX_ATTACK_TOKENS = 160

V3_DEEP_MIN_TOKENS = 1200
V3_DEEP_MAX_TOTAL_TOKENS = 4040
V3_DEEP_CARRIER_MIN_TOKENS = 1150
V3_DEEP_CARRIER_MAX_TOKENS = 3820
V3_DEEP_BENIGN_MAX_TOKENS = 3980
V3_DEEP_MAX_ATTACK_TOKENS = 180

AUDIT_PATH = HERE / "dataset_license_audit.json"
DATASETS_DIR = HERE / "datasets"
HELD_OUT_DIR = DATASETS_DIR / "held_out"
EXCLUSION_PATH = HELD_OUT_DIR / "held_out_exclusion_hashes.json"

ALLOWED_LICENSES = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0"})
FORBIDDEN_DATASET_IDS = frozenset(
    {
        "cyberec/Prompt-injection-dataset",
        "deepset/prompt-injections",
        "Slfgod/prompt-guard",
        "JailbreakBench/JBB-Behaviors",
        "h1alexbel/github-readmes",
        "prodnull/prompt-injection-repo-dataset",
        "ai4cloud/CloudEval-YAML",
    }
)


def normalize_for_dedup(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_clean(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", int(max_chars * 0.75), max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut].strip()


class ExclusionGuard:
    """Enforces zero exact, normalized, and attack-substring overlap with held_out/."""

    def __init__(self) -> None:
        if not EXCLUSION_PATH.is_file():
            raise FileNotFoundError(f"Missing exclusion registry: {EXCLUSION_PATH}")
        excl = json.loads(EXCLUSION_PATH.read_text(encoding="utf-8"))
        self.exact_hashes: set[str] = set(excl["exact_sha256"])
        self.norm_hashes: set[str] = set(excl["normalized_sha256"])

        self.core_attack_norms: list[str] = []
        self.all_heldout_attack_norms: list[str] = []
        for fname in ("external_core.jsonl", "longctx_indirect.jsonl", "unicode_obfuscated.jsonl"):
            fpath = HELD_OUT_DIR / fname
            for line in fpath.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                txt = str(row.get("text") or "")
                if not txt:
                    continue
                self.exact_hashes.add(sha256_text(txt))
                norm_raw = normalize_for_dedup(txt)
                norm_clean = normalize_for_dedup(normalize_text(txt))
                if norm_raw:
                    self.norm_hashes.add(sha256_text(norm_raw))
                if norm_clean:
                    self.norm_hashes.add(sha256_text(norm_clean))
                if int(row.get("label", 0)) == 1 and len(norm_clean) >= 40:
                    self.all_heldout_attack_norms.append(norm_clean)
                    if fname == "external_core.jsonl":
                        self.core_attack_norms.append(norm_clean)

        # Deduplicate longest-first for deterministic substring checks
        self.core_attack_norms = sorted(set(self.core_attack_norms), key=len, reverse=True)
        self.all_heldout_attack_norms = sorted(set(self.all_heldout_attack_norms), key=len, reverse=True)

    def is_excluded(self, text: str, check_substring: bool = True) -> tuple[bool, str | None]:
        if not text or not text.strip():
            return True, "empty"
        if sha256_text(text) in self.exact_hashes:
            return True, "exact_sha256"
        norm_raw = normalize_for_dedup(text)
        if not norm_raw or sha256_text(norm_raw) in self.norm_hashes:
            return True, "normalized_sha256"
        clean = normalize_text(text)
        if sha256_text(clean) in self.exact_hashes:
            return True, "exact_sha256_clean"
        norm_clean = normalize_for_dedup(clean)
        if not norm_clean or sha256_text(norm_clean) in self.norm_hashes:
            return True, "normalized_sha256_clean"

        if check_substring:
            for atk in self.core_attack_norms:
                if atk in norm_clean:
                    return True, "core_attack_substring"
            if len(norm_clean) >= 40:
                for h_atk in self.all_heldout_attack_norms:
                    if norm_clean in h_atk:
                        return True, "contained_in_heldout_attack"
        return False, None


def load_audit_registry() -> dict[str, dict[str, Any]]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    reg: dict[str, dict[str, Any]] = {}
    for entry in audit["datasets"]:
        reg[entry["id"]] = entry
    return reg


def assert_train_allowed(ds_id: str, reg: dict[str, dict[str, Any]]) -> str:
    if ds_id in FORBIDDEN_DATASET_IDS:
        raise RuntimeError(f"Forbidden held-out dataset requested for training: {ds_id}")
    if ds_id not in reg:
        raise RuntimeError(f"Dataset missing from license audit registry: {ds_id}")
    entry = reg[ds_id]
    if entry.get("verdict") != "PASS_TRAIN":
        raise RuntimeError(f"Dataset verdict is not PASS_TRAIN ({entry.get('verdict')}): {ds_id}")
    lic = str(entry.get("license") or "").lower()
    if lic not in ALLOWED_LICENSES:
        raise RuntimeError(f"Dataset license '{lic}' not in allowed list for {ds_id}")
    return str(entry["sha"])


def make_record(
    rec_id: str,
    text: str,
    label: int,
    source: str,
    dataset_id: str,
    license_name: str,
    category: str,
    is_hard_negative: bool = False,
    group_key: str | None = None,
) -> dict[str, Any]:
    return {
        "id": rec_id,
        "text": text,
        "label": int(label),
        "source": source,
        "dataset_id": dataset_id,
        "license": license_name,
        "category": category,
        "is_hard_negative": bool(is_hard_negative) if label == 0 else False,
        "char_length": len(text),
        "_group_key": group_key or sha256_text(normalize_for_dedup(normalize_text(text)))[:16],
    }


def truncate_to_token_budget(text: str, tokenizer: Any, max_tokens: int) -> tuple[str, list[tuple[int, int]]]:
    """Truncate normalized text cleanly to <= max_tokens using ModernBERT offset mapping."""
    norm = normalize_text(text).strip()
    enc = tokenizer(norm, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets: list[tuple[int, int]] = enc["offset_mapping"]
    if len(offsets) <= max_tokens:
        return norm, offsets
    cut_char = offsets[max_tokens - 1][1]
    nl = norm.rfind("\n", max(0, cut_char - 200), cut_char)
    if nl != -1 and nl > int(cut_char * 0.85):
        cut_char = nl
    truncated = norm[:cut_char].strip()
    enc2 = tokenizer(truncated, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets2: list[tuple[int, int]] = enc2["offset_mapping"]
    if len(offsets2) > max_tokens:
        cut2 = offsets2[max_tokens - 1][1]
        truncated = truncated[:cut2].strip()
        enc2 = tokenizer(truncated, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets2 = enc2["offset_mapping"]
    return truncated, offsets2


def embed_indirect_attack_token_budgeted(
    carrier_text: str,
    attack_text: str,
    tokenizer: Any,
    variant: int,
    deep: bool,
    rng: random.Random,
) -> tuple[str, int, int, int] | None:
    """v2 indirect embedder (kept for --version v2 reproducibility)."""
    carrier_trunc, offsets = truncate_to_token_budget(carrier_text, tokenizer, MAX_CARRIER_TOKENS)
    n_tok = len(offsets)
    if deep:
        if n_tok < MIN_CARRIER_TOKENS:
            return None
        lo_tok = 550
        hi_tok = min(3000, n_tok - 1)
        if hi_tok <= lo_tok:
            return None
        target_tok = rng.randint(lo_tok, hi_tok)
    else:
        if n_tok < 180:
            return None
        lo_tok = 100
        hi_tok = min(500, n_tok - 1)
        if hi_tok <= lo_tok:
            return None
        target_tok = rng.randint(lo_tok, hi_tok)

    raw_char = offsets[target_tok][0]
    offset = raw_char
    nl_fwd = carrier_trunc.find("\n", raw_char, min(len(carrier_trunc), raw_char + 120))
    if nl_fwd != -1:
        cand = nl_fwd + 1
        if deep and cand < offsets[min(3040, n_tok - 1)][1]:
            offset = cand
        elif not deep and cand <= offsets[min(520, n_tok - 1)][1]:
            offset = cand

    atk_clean = normalize_text(attack_text).strip()
    if variant == 0:
        wrapper = f"\n\n<!-- maintainer-note:\n{atk_clean}\n-->\n\n"
    elif variant == 1:
        wrapper = (
            "\n\ndef _verify_metadata():\n"
            '    """\n'
            f"    {atk_clean}\n"
            '    """\n'
            "    return True\n\n"
        )
    elif variant == 2:
        wrapper = f"\n\n[^footnote]: {atk_clean}\n\n"
    elif variant == 3:
        wrapper = f'\n\n{{"status": "ok", "tool_output": {json.dumps(atk_clean)}}}\n\n'
    else:
        wrapper = f"\n[2026-03-14T10:15:22Z] INFO agent_worker: note={json.dumps(atk_clean)}\n"

    injected = carrier_trunc[:offset] + wrapper + carrier_trunc[offset:]
    total_ids = tokenizer(normalize_text(injected), add_special_tokens=True, truncation=False)["input_ids"]
    if len(total_ids) > MAX_INDIRECT_TOTAL_TOKENS:
        prefix_and_needle = carrier_trunc[:offset] + wrapper
        suffix = carrier_trunc[offset:]
        pn_toks = len(tokenizer(normalize_text(prefix_and_needle), add_special_tokens=True, truncation=False)["input_ids"])
        rem_toks = MAX_INDIRECT_TOTAL_TOKENS - pn_toks - 16
        if rem_toks <= 0:
            return None
        suffix_trunc, _ = truncate_to_token_budget(suffix, tokenizer, rem_toks)
        injected = prefix_and_needle + suffix_trunc
        total_ids = tokenizer(normalize_text(injected), add_special_tokens=True, truncation=False)["input_ids"]
        if len(total_ids) > MAX_INDIRECT_TOTAL_TOKENS:
            return None

    return injected, offset, target_tok, len(total_ids)


def format_v3_indirect_payload(
    atk_clean: str,
    raw_inline: bool,
    wrapper_idx: int,
    rng: random.Random,
) -> tuple[str, str]:
    """Format an indirect payload either as raw inline continuation or via diverse realistic wrappers."""
    if raw_inline:
        style = rng.randrange(3)
        if style == 0:
            return f"\n\n{atk_clean}\n\n", "raw_para"
        if style == 1:
            return f"\n{atk_clean}\n", "raw_line"
        return f" {atk_clean} ", "raw_sentence"

    w = wrapper_idx % 6
    if w == 0:
        sub = rng.randrange(3)
        if sub == 0:
            return f'\n{{"tool": "read_file", "content": {json.dumps(atk_clean)}}}\n', "json_content"
        if sub == 1:
            return f'\n{{"status": "ok", "stdout": {json.dumps(atk_clean)}}}\n', "json_stdout"
        return f'\n{{"match": true, "snippet": {json.dumps(atk_clean)}}}\n', "json_snippet"
    if w == 1:
        prefix = "> Note: " if rng.random() < 0.5 else "> "
        return f"\n\n{prefix}{atk_clean}\n\n", "md_blockquote"
    if w == 2:
        if rng.random() < 0.5:
            return f'\n\ndef _context_check():\n    """{atk_clean}"""\n    return True\n\n', "py_docstring"
        return f"\n/* {atk_clean} */\n", "js_comment"
    if w == 3:
        return f"\n[2026-04-18T14:22:09Z] INFO worker: detail={json.dumps(atk_clean)}\n", "log_line"
    if w == 4:
        return f"\n\n<!-- {atk_clean} -->\n\n", "html_comment"
    return f"\n# context-note: {atk_clean}\n", "yaml_comment"


def embed_indirect_attack_v3(
    carrier_trunc: str,
    offsets: list[tuple[int, int]],
    attack_text: str,
    tokenizer: Any,
    raw_inline: bool,
    wrapper_idx: int,
    deep: bool,
    rng: random.Random,
) -> tuple[str, int, int, int, str] | None:
    """Embed attack inside a v3 short/med (120-1200 tok) or deep (1200-4040 tok) carrier.

    Returns (injected_text, char_offset, token_offset, total_tokens, style_tag) or None.
    """
    n_tok = len(offsets)
    atk_clean = normalize_text(attack_text).strip()
    if not atk_clean:
        return None

    if deep:
        if n_tok < V3_DEEP_CARRIER_MIN_TOKENS:
            return None
        if n_tok >= 2600 and rng.random() < 0.72:
            lo_tok = 2500
            hi_tok = min(3800, n_tok - 1)
        elif n_tok >= 1500:
            lo_tok = 1100
            hi_tok = min(2500, n_tok - 1)
        else:
            lo_tok = 450
            hi_tok = n_tok - 1
        max_total = V3_DEEP_MAX_TOTAL_TOKENS
        min_total = V3_DEEP_MIN_TOKENS
    else:
        if n_tok < V3_SHORT_CARRIER_MIN_TOKENS:
            return None
        lo_tok = max(16, int(n_tok * 0.12))
        hi_tok = max(lo_tok + 1, int(n_tok * 0.88))
        hi_tok = min(hi_tok, n_tok - 1)
        max_total = V3_SHORT_MAX_TOKENS
        min_total = V3_SHORT_MIN_TOKENS

    if hi_tok <= lo_tok:
        return None
    target_tok = rng.randint(lo_tok, hi_tok)
    raw_char = offsets[target_tok][0]
    offset = raw_char

    # Snap to a nearby newline or sentence boundary if available within narrow token bounds
    if raw_inline and rng.random() < 0.35:
        sp_fwd = carrier_trunc.find(". ", raw_char, min(len(carrier_trunc), raw_char + 120))
        if sp_fwd != -1:
            offset = sp_fwd + 2
    else:
        nl_fwd = carrier_trunc.find("\n", raw_char, min(len(carrier_trunc), raw_char + 120))
        if nl_fwd != -1:
            cand = nl_fwd + 1
            max_snap_tok = min(n_tok - 1, target_tok + 35)
            if cand <= offsets[max_snap_tok][1]:
                offset = cand

    payload, style_tag = format_v3_indirect_payload(atk_clean, raw_inline=raw_inline, wrapper_idx=wrapper_idx, rng=rng)
    injected = carrier_trunc[:offset] + payload + carrier_trunc[offset:]
    total_ids = tokenizer(normalize_text(injected), add_special_tokens=True, truncation=False)["input_ids"]

    if len(total_ids) > max_total:
        # Trim ONLY trailing suffix after the inserted needle so the needle is NEVER touched
        prefix_and_needle = carrier_trunc[:offset] + payload
        suffix = carrier_trunc[offset:]
        pn_toks = len(tokenizer(normalize_text(prefix_and_needle), add_special_tokens=True, truncation=False)["input_ids"])
        rem_toks = max_total - pn_toks - 12
        if rem_toks <= 0:
            return None
        suffix_trunc, _ = truncate_to_token_budget(suffix, tokenizer, rem_toks)
        injected = prefix_and_needle + suffix_trunc
        total_ids = tokenizer(normalize_text(injected), add_special_tokens=True, truncation=False)["input_ids"]
        if len(total_ids) > max_total:
            return None

    if len(total_ids) < min_total:
        return None

    # Compute exact token offset of prefix before payload in final normalized text
    prefix_toks = len(tokenizer(normalize_text(carrier_trunc[:offset]), add_special_tokens=False, truncation=False)["input_ids"])
    return injected, offset, prefix_toks, len(total_ids), style_tag


def classify_ultrachat_topic(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("password", "security", "encrypt", "vulnerability", "cyber", "phish", "malware", "hack", "privacy", "ethic", "exploit", "firewall")):
        return "security_ethics"
    if any(k in low for k in ("roleplay", "act as", "pretend", "character", "story", "poem", "fiction", "novel", "dialogue", "screenplay", "write a creative")):
        return "creative_roleplay"
    if any(k in low for k in ("python", "javascript", "typescript", "sql", "function", "script", "code", "algorithm", "bug", "regex", "api", "html", "css", "bash")):
        return "coding_tech"
    if any(k in low for k in ("summarize", "summary", "brainstorm", "ideas", "list ", "compare", "explain", "analyze", "outline", "bullet")):
        return "summarize_brainstorm"
    return "general"


def collect_records(
    reg: dict[str, dict[str, Any]],
    guard: ExclusionGuard,
    version: str = "v3",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Collect license-verified records.

    Returns (base_records, short_med_carrier_pool, deep_carrier_pool, stats).
    For v2 compatibility, short_med_carrier_pool is empty and deep_carrier_pool holds long_doc_pool.
    """
    rng = random.Random(SEED)
    records: list[dict[str, Any]] = []
    short_med_pool: list[dict[str, Any]] = []
    deep_pool: list[dict[str, Any]] = []
    stats: dict[str, Any] = defaultdict(lambda: {"raw_loaded": 0, "kept_pre_dedup": 0, "excluded_heldout": 0})
    is_v3 = version == "v3"

    def try_add(rec: dict[str, Any], pool: list[dict[str, Any]] = records, count_stats: bool = True) -> bool:
        src = rec["source"]
        if count_stats:
            stats[src]["raw_loaded"] += 1
        excl, _reason = guard.is_excluded(rec["text"], check_substring=True)
        if excl:
            if count_stats:
                stats[src]["excluded_heldout"] += 1
            return False
        pool.append(rec)
        if count_stats:
            stats[src]["kept_pre_dedup"] += 1
        return True

    # ------------------------------------------------------------------
    # 1. yanismiraoui/prompt_injections_v2 (Apache-2.0)
    # ------------------------------------------------------------------
    ds_id = "yanismiraoui/prompt_injections_v2"
    sha = assert_train_allowed(ds_id, reg)
    en_rows, multi_atk, benign_rows = [], [], []
    for sp in ("train", "test"):
        ds = load_dataset(ds_id, split=sp, revision=sha)
        for i, r in enumerate(ds):
            txt = str(r.get("text") or "").strip()
            lbl = int(r["label"])
            lang = str(r.get("language") or "")
            tech = str(r.get("technique") or "")
            if not txt:
                continue
            item = (f"{sp}:{i}", txt, lbl, lang, tech)
            if lbl == 0:
                benign_rows.append(item)
            elif lang == "en":
                en_rows.append(item)
            else:
                multi_atk.append(item)
    rng.shuffle(multi_atk)
    selected_yanis = (en_rows + multi_atk + benign_rows) if is_v3 else (en_rows + multi_atk[:1600] + benign_rows)
    for rid, txt, lbl, lang, tech in selected_yanis:
        try_add(
            make_record(
                f"yanis_v2:{rid}",
                txt,
                lbl,
                "yanismiraoui_v2",
                ds_id,
                "apache-2.0",
                f"{tech}_{lang}",
                is_hard_negative=(lbl == 0),
            )
        )

    # ------------------------------------------------------------------
    # 2. reshabhs/SPML_Chatbot_Prompt_Injection (MIT)
    #    Symmetric 30% [SYSTEM]+[USER] / 70% bare User Prompt for BOTH classes
    # ------------------------------------------------------------------
    ds_id = "reshabhs/SPML_Chatbot_Prompt_Injection"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    spml_pos = [
        i
        for i, r in enumerate(ds)
        if int(r["Prompt injection"]) == 1 and str(r.get("Source") or "") == "Gandalf"
    ]
    spml_neg = [i for i, r in enumerate(ds) if int(r["Prompt injection"]) == 0]
    rng.shuffle(spml_pos)
    rng.shuffle(spml_neg)
    pos_cap = len(spml_pos) if is_v3 else 3000
    neg_cap = len(spml_neg) if is_v3 else 3000
    for lbl_target, idxs in ((1, spml_pos[:pos_cap]), (0, spml_neg[:neg_cap])):
        for k, idx in enumerate(idxs):
            r = ds[idx]
            sys_p = str(r.get("System Prompt") or "").strip()
            usr_p = str(r.get("User Prompt") or "").strip()
            if not usr_p:
                continue
            if sys_p and (k % 10) < 3:
                txt = f"[SYSTEM] {sys_p}\n[USER] {usr_p}"
            else:
                txt = usr_p
            try_add(
                make_record(
                    f"spml:{idx}",
                    txt,
                    lbl_target,
                    "spml",
                    ds_id,
                    "mit",
                    "spml_prompt_injection" if lbl_target == 1 else "spml_benign",
                    is_hard_negative=(lbl_target == 0) if is_v3 else (lbl_target == 0 and sys_p != ""),
                )
            )

    # ------------------------------------------------------------------
    # 3. S-Labs/prompt-injection-dataset (MIT, train split ONLY)
    # ------------------------------------------------------------------
    ds_id = "S-Labs/prompt-injection-dataset"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    sl_pos = [i for i, r in enumerate(ds) if int(r["label"]) == 1]
    sl_neg = [i for i, r in enumerate(ds) if int(r["label"]) == 0]
    rng.shuffle(sl_pos)
    rng.shuffle(sl_neg)
    sl_pos_cap = len(sl_pos) if is_v3 else 3500
    sl_neg_cap = len(sl_neg) if is_v3 else 3500
    for idx in sl_pos[:sl_pos_cap] + sl_neg[:sl_neg_cap]:
        r = ds[idx]
        lbl = int(r["label"])
        txt = str(r.get("text") or "").strip()
        if txt:
            try_add(
                make_record(
                    f"slabs:{idx}",
                    txt,
                    lbl,
                    "slabs",
                    ds_id,
                    "mit",
                    "prompt_injection" if lbl == 1 else "hard_negative_trigger",
                    is_hard_negative=(lbl == 0),
                )
            )

    # ------------------------------------------------------------------
    # 4. Lakera/gandalf_ignore_instructions (MIT, all splits -> train)
    # ------------------------------------------------------------------
    ds_id = "Lakera/gandalf_ignore_instructions"
    sha = assert_train_allowed(ds_id, reg)
    for sp in ("train", "validation", "test"):
        ds = load_dataset(ds_id, split=sp, revision=sha)
        for i, r in enumerate(ds):
            txt = str(r.get("text") or "").strip()
            if txt:
                try_add(make_record(f"gandalf:{sp}:{i}", txt, 1, "gandalf", ds_id, "mit", "human_injection"))

    # ------------------------------------------------------------------
    # 5. walledai/JailbreakHub (MIT)
    # ------------------------------------------------------------------
    ds_id = "walledai/JailbreakHub"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    jh_pos = [i for i, r in enumerate(ds) if bool(r["jailbreak"])]
    jh_neg = [i for i, r in enumerate(ds) if not bool(r["jailbreak"])]
    rng.shuffle(jh_neg)
    jh_neg_cap = 2200 if is_v3 else 1800
    for idx in jh_pos + jh_neg[:jh_neg_cap]:
        r = ds[idx]
        lbl = 1 if bool(r["jailbreak"]) else 0
        txt = str(r.get("prompt") or "").strip()
        if txt:
            try_add(
                make_record(
                    f"jhub:{idx}",
                    txt,
                    lbl,
                    "jailbreak_hub",
                    ds_id,
                    "mit",
                    "jailbreak" if lbl == 1 else "benign_community",
                    is_hard_negative=(lbl == 0),
                )
            )

    # ------------------------------------------------------------------
    # 6. jackhhao/jailbreak-classification (Apache-2.0)
    # ------------------------------------------------------------------
    ds_id = "jackhhao/jailbreak-classification"
    sha = assert_train_allowed(ds_id, reg)
    for sp in ("train", "test"):
        ds = load_dataset(ds_id, split=sp, revision=sha)
        for i, r in enumerate(ds):
            txt = str(r.get("prompt") or "").strip()
            lbl = 1 if str(r.get("type") or "").strip().lower() == "jailbreak" else 0
            if txt:
                try_add(
                    make_record(
                        f"jackhhao:{sp}:{i}",
                        txt,
                        lbl,
                        "jackhhao",
                        ds_id,
                        "apache-2.0",
                        "jailbreak" if lbl == 1 else "benign",
                        is_hard_negative=(lbl == 0) if is_v3 else False,
                    )
                )

    # ------------------------------------------------------------------
    # 7. sevdeawesome/jailbreak_success (MIT, diverse wrappers)
    # ------------------------------------------------------------------
    ds_id = "sevdeawesome/jailbreak_success"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    by_wrapper: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(ds):
        wname = str(r.get("jailbreak_prompt_name") or "")
        if wname in ("original_prompt", "none"):
            continue
        by_wrapper[wname].append(i)
    sev_selected: list[int] = []
    per_wrap = 60 if is_v3 else 40
    sev_cap = 1800 if is_v3 else 1200
    for wname in sorted(by_wrapper):
        idxs = by_wrapper[wname]
        rng.shuffle(idxs)
        sev_selected.extend(idxs[:per_wrap])
    rng.shuffle(sev_selected)
    for idx in sev_selected[:sev_cap]:
        r = ds[idx]
        txt = str(r.get("jailbreak_prompt_text") or "").strip()
        if txt:
            try_add(
                make_record(
                    f"sev:{idx}",
                    truncate_clean(txt, 12000),
                    1,
                    "jailbreak_success",
                    ds_id,
                    "mit",
                    f"wrapper_{r.get('jailbreak_prompt_name')}",
                )
            )

    # ------------------------------------------------------------------
    # 9-13. Small high-signal attack/benign corpora
    # ------------------------------------------------------------------
    ds_id = "shanemhamilton/llm-prompt-guard-tuning-corpus"
    sha = assert_train_allowed(ds_id, reg)
    for fname, lbl in (("attacks.jsonl", 1), ("benign.jsonl", 0)):
        fpath = hf_hub_download(ds_id, fname, repo_type="dataset", revision=sha)
        for i, line in enumerate(Path(fpath).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            txt = str(r.get("text") or "").strip()
            if txt:
                try_add(
                    make_record(
                        f"shane:{fname}:{i}",
                        txt,
                        lbl,
                        "shane_corpus",
                        ds_id,
                        "mit",
                        str(r.get("category") or r.get("domain") or "general"),
                        is_hard_negative=(lbl == 0),
                    )
                )

    ds_id = "darkknight25/Prompt_Injection_Benign_Prompt_Dataset"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    for i, r in enumerate(ds):
        txt = str(r.get("prompt") or "").strip()
        lbl = 1 if str(r.get("label") or "").strip().lower() == "malicious" else 0
        if txt:
            try_add(
                make_record(
                    f"darkknight:{i}",
                    txt,
                    lbl,
                    "darkknight",
                    ds_id,
                    "mit",
                    str(r.get("attack_type") or "benign"),
                    is_hard_negative=(lbl == 0),
                )
            )

    ds_id = "Naomibas/llm-system-prompts-benchmark"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    for i, r in enumerate(ds):
        sys_p = str(r.get("prompt") or "").strip()
        probe = str(r.get("probe") or "").strip()
        txt = f"[SYSTEM] {sys_p}\n[USER] {probe}" if (sys_p and probe) else (sys_p or probe)
        if txt:
            try_add(make_record(f"naomibas:{i}", txt, 0, "naomibas", ds_id, "apache-2.0", "system_prompt", is_hard_negative=True))

    ds_id = "hse-llm/prompt-injections"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    for i, r in enumerate(ds):
        mal = str(r.get("Malicious prompts") or "").strip()
        if mal:
            try_add(make_record(f"hse:mal:{i}", mal, 1, "hse_llm", ds_id, "mit", "multilingual_injection"))
        for col in ("Safe prompt DeepSeek", "Safe prompt Llama"):
            safe = str(r.get(col) or "").strip()
            if safe:
                try_add(make_record(f"hse:{col}:{i}", safe, 0, "hse_llm", ds_id, "mit", "safe_pair", is_hard_negative=True))

    ds_id = "aestera/prompt_injection_payloads"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    for i, r in enumerate(ds):
        atk = str(r.get("input_payload") or "").strip()
        ben = str(r.get("next_payload") or "").strip()
        if atk:
            try_add(make_record(f"aestera:atk:{i}", atk, 1, "aestera", ds_id, "mit", "payload"))
        if ben:
            try_add(make_record(f"aestera:ben:{i}", ben, 0, "aestera", ds_id, "mit", "benign_followup", is_hard_negative=True))

    # ------------------------------------------------------------------
    # 14-15. Benign Developer Hard Negatives: SQL
    # ------------------------------------------------------------------
    ds_id = "gretelai/synthetic_text_to_sql"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    g_idx = list(range(len(ds)))
    rng.shuffle(g_idx)
    for k, idx in enumerate(g_idx[:2200]):
        r = ds[idx]
        if k % 3 == 0:
            txt = f"Question: {r['sql_prompt'].strip()}\nSchema:\n{r['sql_context'].strip()}\nSQL:\n{r['sql'].strip()}"
        elif k % 3 == 1:
            txt = f"{r['sql_context'].strip()}\n\n-- {r['sql_prompt'].strip()}\n{r['sql'].strip()}"
        else:
            txt = f"Task: {r['sql_prompt'].strip()}\nQuery:\n{r['sql'].strip()}\nExplanation: {r['sql_explanation'].strip()}"
        try_add(make_record(f"gretel_sql:{idx}", truncate_clean(txt, 8000), 0, "gretel_sql", ds_id, "apache-2.0", "developer_sql", is_hard_negative=True))

    ds_id = "kaxap/pg-wikiSQL-sql-instructions-80k"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    w_idx = list(range(len(ds)))
    rng.shuffle(w_idx)
    for idx in w_idx[:1500]:
        r = ds[idx]
        txt = f"Question: {r['question'].strip()}\nSchema:\n{r['create_table_statement'].strip()}\nSQL:\n{r['sql_query'].strip()}"
        try_add(make_record(f"wikisql:{idx}", truncate_clean(txt, 8000), 0, "wikisql_train", ds_id, "bsd-3-clause", "developer_sql", is_hard_negative=True))

    # ------------------------------------------------------------------
    # 16-18. Benign Developer Hard Negatives: MCP / Tool Calls / JSON
    # ------------------------------------------------------------------
    ds_id = "quotientai/limbic-eval-tool-use-mcp"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="test", revision=sha)
    l_idx = list(range(len(ds)))
    rng.shuffle(l_idx)
    if is_v3:
        limbic_sm, limbic_dp, limbic_rec = 0, 0, 0
        for idx in l_idx:
            r = ds[idx]
            txt = f"Available Tools:\n{str(r.get('available_tools') or '').strip()}\n\nMessage History:\n{str(r.get('message_history') or '').strip()}"
            txt = truncate_clean(txt, 16000)
            rec = make_record(f"limbic_mcp:{idx}", txt, 0, "limbic_mcp", ds_id, "mit", "mcp_tools", is_hard_negative=True)
            if 500 <= len(txt) < 4200 and limbic_sm < 1000:
                if try_add(rec, pool=short_med_pool, count_stats=False):
                    limbic_sm += 1
            elif len(txt) >= 4800 and limbic_dp < 900:
                if try_add(rec, pool=deep_pool, count_stats=False):
                    limbic_dp += 1
            elif limbic_rec < 1200:
                if try_add(rec):
                    limbic_rec += 1
            if limbic_sm >= 1000 and limbic_dp >= 900 and limbic_rec >= 1200:
                break
    else:
        for idx in l_idx[:2400]:
            r = ds[idx]
            txt = f"Available Tools:\n{str(r.get('available_tools') or '').strip()}\n\nMessage History:\n{str(r.get('message_history') or '').strip()}"
            txt = truncate_clean(txt, 14000)
            rec = make_record(f"limbic_mcp:{idx}", txt, 0, "limbic_mcp", ds_id, "mit", "mcp_tools", is_hard_negative=True)
            if len(txt) >= 1800 and len(deep_pool) < 1200:
                excl, _ = guard.is_excluded(txt)
                if not excl:
                    deep_pool.append(rec)
            else:
                try_add(rec)

    ds_id = "jdaddyalbs/playwright-mcp-toolcalling"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    pw_idx = list(range(len(ds)))
    rng.shuffle(pw_idx)
    for idx in pw_idx[:1200]:
        r = ds[idx]
        txt = str(r.get("text") or "").strip()
        if not txt:
            txt = f"{r.get('prompt', '')}\n{r.get('query', '')}\n{r.get('true_answer', '')}".strip()
        if txt:
            try_add(make_record(f"playwright_mcp:{idx}", truncate_clean(txt, 12000), 0, "playwright_mcp", ds_id, "mit", "mcp_browser", is_hard_negative=True))

    ds_id = "NousResearch/hermes-function-calling-v1"
    sha = assert_train_allowed(ds_id, reg)
    hermes_rows: list[tuple[str, str]] = []
    for cfg in ("func_calling", "json_mode_agentic", "json_mode_singleturn"):
        ds = load_dataset(ds_id, cfg, split="train", revision=sha)
        for i, r in enumerate(ds):
            conv = "\n".join(f"{m.get('from', '')}: {m.get('value', '')}" for m in (r.get("conversations") or []))
            tools = str(r.get("tools") or "").strip()
            txt = f"Tools:\n{tools}\n\n{conv}".strip() if tools else conv.strip()
            if txt:
                hermes_rows.append((f"{cfg}:{i}", truncate_clean(txt, 16000)))
    rng.shuffle(hermes_rows)
    if is_v3:
        h_sm, h_dp, h_rec = 0, 0, 0
        for rid, txt in hermes_rows:
            rec = make_record(f"hermes:{rid}", txt, 0, "hermes_fn", ds_id, "apache-2.0", "function_calling", is_hard_negative=True)
            if 500 <= len(txt) < 4200 and h_sm < 1000:
                if try_add(rec, pool=short_med_pool, count_stats=False):
                    h_sm += 1
            elif len(txt) >= 4800 and h_dp < 900:
                if try_add(rec, pool=deep_pool, count_stats=False):
                    h_dp += 1
            elif h_rec < 1500:
                if try_add(rec):
                    h_rec += 1
            if h_sm >= 1000 and h_dp >= 900 and h_rec >= 1500:
                break
    else:
        for rid, txt in hermes_rows[:2200]:
            rec = make_record(f"hermes:{rid}", txt, 0, "hermes_fn", ds_id, "apache-2.0", "function_calling", is_hard_negative=True)
            if len(txt) >= 1800 and len(deep_pool) < 1800:
                excl, _ = guard.is_excluded(txt)
                if not excl:
                    deep_pool.append(rec)
            else:
                try_add(rec)

    # ------------------------------------------------------------------
    # 19-21b. Real System Prompts, Assistant Chats, & UltraChat Prompts
    # ------------------------------------------------------------------
    ds_id = "EricLu/System-Prompt-Instruction-Real-world-Implementation-Training-set"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    seen_sys: set[str] = set()
    unique_sys_norms: list[str] = []
    for r in ds:
        for msg in r["messages"]:
            if msg.get("role") == "system" and msg.get("content") and msg["content"].strip():
                nsp = normalize_for_dedup(msg["content"].strip())
                if nsp not in seen_sys:
                    seen_sys.add(nsp)
                    unique_sys_norms.append(nsp)
                break
    excluded_last200 = set(unique_sys_norms[-200:])
    eric_candidates: list[tuple[int, str, str]] = []
    for i, r in enumerate(ds):
        sys_txt = ""
        usr_txt = ""
        for msg in r["messages"]:
            role = msg.get("role")
            content = str(msg.get("content") or "").strip()
            if role == "system" and content and not sys_txt:
                sys_txt = content
            elif role == "user" and content and not usr_txt:
                usr_txt = content
        if not sys_txt or normalize_for_dedup(sys_txt) in excluded_last200:
            continue
        eric_candidates.append((i, sys_txt, usr_txt))
    rng.shuffle(eric_candidates)
    seen_eric_sys: set[str] = set()
    eric_added = 0
    for idx, sys_txt, usr_txt in eric_candidates:
        if eric_added >= 2000:
            break
        nsp = normalize_for_dedup(sys_txt)
        if nsp not in seen_eric_sys and eric_added < 1500:
            seen_eric_sys.add(nsp)
            txt = sys_txt
        elif usr_txt:
            txt = f"[SYSTEM] {sys_txt}\n[USER] {usr_txt}"
        else:
            continue
        if try_add(
            make_record(
                f"ericlu:{idx}",
                truncate_clean(txt, 10000),
                0,
                "ericlu_sys",
                ds_id,
                "mit",
                "system_prompt",
                is_hard_negative=True,
                group_key=sha256_text(nsp)[:16],
            )
        ):
            eric_added += 1

    ds_id = "fka/awesome-chatgpt-prompts"
    sha = assert_train_allowed(ds_id, reg)
    fka_csv = hf_hub_download(ds_id, "prompts.csv", repo_type="dataset", revision=sha)
    fka_ds = load_dataset("csv", data_files=fka_csv, split="train")
    fka_idx = list(range(len(fka_ds)))
    rng.shuffle(fka_idx)
    for idx in fka_idx[:250]:
        r = fka_ds[idx]
        txt = str(r.get("prompt") or "").strip()
        if txt:
            try_add(make_record(f"fka:{idx}", truncate_clean(txt, 6000), 0, "awesome_prompts", ds_id, "cc0-1.0", "persona_prompt", is_hard_negative=True))

    ds_id = "OpenAssistant/oasst1"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    if is_v3:
        oasst_prompter = [
            i for i, r in enumerate(ds)
            if not r.get("deleted") and str(r.get("role") or "") == "prompter" and len(str(r.get("text") or "").strip()) >= 20
        ]
        oasst_other = [
            i for i, r in enumerate(ds)
            if not r.get("deleted") and str(r.get("role") or "") != "prompter" and len(str(r.get("text") or "").strip()) >= 20
        ]
        rng.shuffle(oasst_prompter)
        rng.shuffle(oasst_other)
        # 3,500 total (+1,500 over v2), prioritizing real user prompters and marked is_hard_negative=True
        for idx in oasst_prompter[:2500] + oasst_other[:1000]:
            r = ds[idx]
            txt = truncate_clean(str(r["text"]), 8000)
            cat = "human_user_prompt" if str(r.get("role") or "") == "prompter" else "human_chat"
            try_add(make_record(f"oasst1:{idx}", txt, 0, "oasst1", ds_id, "apache-2.0", cat, is_hard_negative=True))
    else:
        oasst_idx = [i for i, r in enumerate(ds) if r.get("text") and len(str(r["text"]).strip()) >= 20 and not r.get("deleted")]
        rng.shuffle(oasst_idx)
        for idx in oasst_idx[:2000]:
            r = ds[idx]
            txt = truncate_clean(str(r["text"]), 8000)
            try_add(make_record(f"oasst1:{idx}", txt, 0, "oasst1", ds_id, "apache-2.0", "human_chat", is_hard_negative=False))

    if is_v3:
        ds_id = "HuggingFaceH4/ultrachat_200k"
        sha = assert_train_allowed(ds_id, reg)
        uc_ds = load_dataset(ds_id, split="train_sft", revision=sha)
        uc_indices = list(range(len(uc_ds)))
        rng.shuffle(uc_indices)

        # Collect 5,000 diverse benign conversational/instructional hard negatives
        topic_caps = {
            "creative_roleplay": 1100,
            "coding_tech": 1100,
            "security_ethics": 800,
            "summarize_brainstorm": 1000,
            "multi_turn": 500,
            "general": 500,
        }
        topic_counts: Counter[str] = Counter()
        uc_added = 0
        uc_short_med_carriers = 0

        for idx in uc_indices:
            if uc_added >= 5000 and uc_short_med_carriers >= 600:
                break
            r = uc_ds[idx]
            msgs = r.get("messages") or []
            user_turns = [str(m.get("content") or "").strip() for m in msgs if m.get("role") == "user" and str(m.get("content") or "").strip()]
            if not user_turns:
                continue

            # Route longer benign multi-turn/user docs to short_med_pool (disjoint from base!)
            if uc_added >= 5000 or (idx % 11 == 0 and uc_short_med_carriers < 600):
                conv_doc = "\n\n".join(f"{m.get('role', 'user').title()}: {str(m.get('content') or '').strip()}" for m in msgs[:4])
                conv_doc = truncate_clean(conv_doc, 4200)
                if 600 <= len(conv_doc) <= 4200 and uc_short_med_carriers < 600:
                    rec_c = make_record(
                        f"ultrachat_doc:{idx}",
                        conv_doc,
                        0,
                        "ultrachat_200k",
                        ds_id,
                        "mit",
                        "benign_user_doc",
                        is_hard_negative=True,
                        group_key=f"uc_{idx}",
                    )
                    if try_add(rec_c, pool=short_med_pool, count_stats=False):
                        uc_short_med_carriers += 1
                    continue

            if uc_added >= 5000:
                continue

            # ~10% multi-turn instruction exchanges, ~90% diverse single-turn user prompts
            if len(msgs) >= 3 and topic_counts["multi_turn"] < topic_caps["multi_turn"] and (idx % 7 == 0):
                u0 = str(msgs[0].get("content") or "").strip()
                a0 = str(msgs[1].get("content") or "").strip()
                u1 = str(msgs[2].get("content") or "").strip()
                if u0 and a0 and u1:
                    txt = truncate_clean(f"User: {u0}\n\nAssistant: {truncate_clean(a0, 1200)}\n\nUser: {u1}", 5000)
                    cat = "multi_turn"
                else:
                    continue
            else:
                # Pick first user turn or occasionally a follow-up user turn
                turn_idx = 1 if (len(user_turns) > 1 and idx % 5 == 0) else 0
                txt = truncate_clean(user_turns[turn_idx], 4500)
                if len(txt) < 25:
                    continue
                cat = classify_ultrachat_topic(txt)
                if topic_counts[cat] >= topic_caps[cat]:
                    # Spill over to general only if general still has room
                    if topic_counts["general"] < topic_caps["general"]:
                        cat = "general"
                    else:
                        continue

            rec = make_record(
                f"ultrachat:{idx}",
                txt,
                0,
                "ultrachat_200k",
                ds_id,
                "mit",
                f"uc_{cat}",
                is_hard_negative=True,
                group_key=f"uc_{idx}",
            )
            if try_add(rec):
                topic_counts[cat] += 1
                uc_added += 1

    # ------------------------------------------------------------------
    # 22-24. Security Docs / CVEs (anti-trigger-word hard negatives)
    # ------------------------------------------------------------------
    ds_id = "CyberNative/Code_Vulnerability_Security_DPO"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    cn_idx = list(range(len(ds)))
    rng.shuffle(cn_idx)
    for k, idx in enumerate(cn_idx[:1800]):
        r = ds[idx]
        q = str(r.get("question") or "").strip()
        c = str(r.get("chosen") or "").strip()
        txt = f"{q}\n\n{c}".strip() if k % 2 == 0 else f"Vulnerability: {r.get('vulnerability', '')}\n{q}"
        txt = truncate_clean(txt, 10000)
        if txt:
            try_add(make_record(f"cybernative:{idx}", txt, 0, "cybernative_sec", ds_id, "apache-2.0", "security_code", is_hard_negative=True))

    ds_id = "stasvinokur/cve-and-cwe-dataset-1999-2025"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    cve_pre2024 = [
        i
        for i, r in enumerate(ds)
        if r.get("CVE-ID")
        and not str(r["CVE-ID"]).startswith(("CVE-2024-", "CVE-2025-"))
        and r.get("DESCRIPTION")
        and str(r["DESCRIPTION"]).strip()
    ]
    rng.shuffle(cve_pre2024)
    for idx in cve_pre2024[:2200]:
        r = ds[idx]
        cve_code = str(r["CVE-ID"]).strip()
        cwe_code = str(r.get("CWE-ID") or "N/A").strip()
        sev = str(r.get("SEVERITY") or "UNKNOWN").strip()
        desc = str(r["DESCRIPTION"]).strip()
        txt = f"{cve_code} ({cwe_code}, {sev}): {desc}"
        try_add(make_record(f"cve_pre2024:{cve_code}", truncate_clean(txt, 6000), 0, "cve_pre2024", ds_id, "cc0-1.0", "security_cve", is_hard_negative=True))

    ds_id = "zeroshot/cybersecurity-corpus"
    sha = assert_train_allowed(ds_id, reg)
    for sp in ("train", "validation"):
        ds = load_dataset(ds_id, split=sp, revision=sha)
        for i, r in enumerate(ds):
            txt = str(r.get("text") or "").strip()
            if txt:
                try_add(make_record(f"zeroshot_sec:{sp}:{i}", txt, 0, "zeroshot_sec", ds_id, "cc0-1.0", "security_news", is_hard_negative=True))

    # ------------------------------------------------------------------
    # 25-27. Long Real Developer Docs / Code / READMEs
    # ------------------------------------------------------------------
    ds_id = "patched-codes/generate-readme-eval"
    sha = assert_train_allowed(ds_id, reg)
    for sp in ("train", "test"):
        ds = load_dataset(ds_id, split=sp, revision=sha)
        for i, r in enumerate(ds):
            for field, cat in (("repo_readme", "dev_readme"), ("repo_content", "dev_code")):
                txt = truncate_clean(str(r.get(field) or ""), 16000)
                if len(txt) < 200:
                    continue
                rec = make_record(f"patched:{sp}:{i}:{field}", txt, 0, "patched_readme", ds_id, "apache-2.0", cat, is_hard_negative=True)
                if is_v3:
                    if 500 <= len(txt) < 4200:
                        try_add(rec, pool=short_med_pool, count_stats=False)
                    elif len(txt) >= 4800:
                        try_add(rec, pool=deep_pool, count_stats=False)
                    else:
                        try_add(rec)
                else:
                    if len(txt) >= 1800:
                        excl, _ = guard.is_excluded(txt)
                        if not excl:
                            deep_pool.append(rec)
                    else:
                        try_add(rec)

    ds_id = "CatOwl/terraform3"
    sha = assert_train_allowed(ds_id, reg)
    ds = load_dataset(ds_id, split="train", revision=sha)
    for i, r in enumerate(ds):
        txt = truncate_clean(str(r.get("response_code_only") or r.get("inputs_code_only") or ""), 12000)
        if txt:
            try_add(make_record(f"terraform:{i}", txt, 0, "terraform_iac", ds_id, "mit", "iac_hcl", is_hard_negative=True))

    ds_id = "codeparrot/github-code-clean"
    sha = assert_train_allowed(ds_id, reg)
    target_langs = {"Markdown", "HTML", "Python", "JavaScript", "TypeScript", "SQL", "Dockerfile", "Shell", "Makefile", "CMake"}
    gh_buckets: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    yaml_ci_blocks: list[tuple[str, str, str, str]] = []
    yaml_fence_re = re.compile(r"```(?:yaml|yml)\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)
    for shard in (
        "data/train-00000-of-00880.parquet",
        "data/train-00001-of-00880.parquet",
        "data/train-00002-of-00880.parquet",
        "data/train-00003-of-00880.parquet",
    ):
        pq_path = hf_hub_download(ds_id, shard, repo_type="dataset", revision=sha)
        gh_ds = load_dataset("parquet", data_files=pq_path, split="train")
        shard_tag = shard[-16:-8]
        for i, r in enumerate(gh_ds):
            lic = str(r.get("license") or "").lower()
            if lic not in ("mit", "apache-2.0"):
                continue
            lang = str(r.get("language") or "")
            if lang not in target_langs:
                continue
            code = str(r.get("code") or "").strip()
            if not code:
                continue
            if lang == "Markdown":
                for bi, m in enumerate(yaml_fence_re.finditer(code)):
                    blk = m.group(1).strip()
                    if 80 <= len(blk) <= 10000 and ":" in blk:
                        yaml_ci_blocks.append((f"{shard_tag}:{i}:yml{bi}", truncate_clean(blk, 8000), lic, "yaml_block"))
            if len(code) < 450:
                continue
            gh_buckets[lang].append((f"{shard_tag}:{i}", truncate_clean(code, 22000), lic, lang))

    # Benign YAML/CI/Dockerfile/Makefile hard negatives (~1,600, zero overlap with CloudEval-YAML)
    rng.shuffle(yaml_ci_blocks)
    ci_candidates: list[tuple[str, str, str, str]] = list(yaml_ci_blocks[:1000])
    for ci_lang, ci_cap in (("Dockerfile", 450), ("Makefile", 200), ("CMake", 150)):
        items = list(gh_buckets.get(ci_lang, []))
        rng.shuffle(items)
        ci_candidates.extend(items[:ci_cap])
    rng.shuffle(ci_candidates)
    ci_added = 0
    for rid, txt, lic, kind in ci_candidates:
        if ci_added >= 1600:
            break
        if try_add(
            make_record(
                f"github_yaml_ci:{rid}",
                truncate_clean(txt, 10000),
                0,
                "github_yaml_ci",
                ds_id,
                lic,
                f"ci_{kind.lower()}",
                is_hard_negative=True,
            )
        ):
            ci_added += 1

    if is_v3:
        # Partition github-code-clean cleanly by character length (disjoint across pools & base!)
        used_gh_ids: set[str] = set()
        sm_caps = {
            "Markdown": 1200,
            "HTML": 700,
            "Python": 900,
            "JavaScript": 600,
            "TypeScript": 500,
            "SQL": 300,
            "Shell": 300,
        }
        deep_mid_caps = {
            "Markdown": 1300,
            "HTML": 700,
            "Python": 800,
            "JavaScript": 500,
            "TypeScript": 450,
            "SQL": 150,
            "Shell": 200,
        }
        deep_tail_caps = {
            "Markdown": 2300,
            "HTML": 1200,
            "Python": 1200,
            "JavaScript": 800,
            "TypeScript": 700,
            "SQL": 250,
            "Shell": 350,
        }
        base_short_caps = {
            "Markdown": 450,
            "HTML": 250,
            "Python": 350,
            "JavaScript": 250,
            "TypeScript": 200,
            "SQL": 150,
            "Shell": 150,
        }

        gh_sm_sel: list[tuple[str, str, str, str]] = []
        gh_deep_mid_sel: list[tuple[str, str, str, str]] = []
        gh_deep_tail_sel: list[tuple[str, str, str, str]] = []
        gh_base_sel: list[tuple[str, str, str, str]] = []

        for lang in sorted(sm_caps):
            items = list(gh_buckets.get(lang, []))
            rng.shuffle(items)
            sm_items = [x for x in items if 450 <= len(x[1]) < 4300]
            mid_items = [x for x in items if 4800 <= len(x[1]) < 10500]
            tail_items = [x for x in items if len(x[1]) >= 10500]

            sc = sm_caps[lang]
            bc = base_short_caps[lang]
            gh_sm_sel.extend(sm_items[:sc])
            gh_base_sel.extend(sm_items[sc : sc + bc])
            gh_deep_mid_sel.extend(mid_items[: deep_mid_caps[lang]])
            gh_deep_tail_sel.extend(tail_items[: deep_tail_caps[lang]])

        # Also add ~600 synthetic git-diff / JSON-tool-wrapped code snippets into short_med_pool
        diff_pool_src: list[tuple[str, str, str, str]] = []
        for lang in ("Python", "JavaScript", "TypeScript", "Markdown"):
            items = [x for x in gh_buckets.get(lang, []) if 600 <= len(x[1]) <= 3200]
            diff_pool_src.extend(items[-300:])
        rng.shuffle(diff_pool_src)
        diff_added = 0
        for rid, txt, lic, lang in diff_pool_src:
            if diff_added >= 600:
                break
            lines = txt.splitlines()[:28]
            if len(lines) < 6:
                continue
            if diff_added % 2 == 0:
                body = "\n".join(f"+{ln}" if j % 4 == 0 else f" {ln}" for j, ln in enumerate(lines))
                formatted = f"diff --git a/src/module.{lang.lower()} b/src/module.{lang.lower()}\n@@ -1,{len(lines)} +1,{len(lines)} @@\n{body}"
                cat = "carrier_git_diff"
            else:
                formatted = json.dumps({"tool": "workspace_read", "language": lang, "content": "\n".join(lines)})
                cat = "carrier_json_tool"
            rec = make_record(f"github_fmt:{rid}", formatted, 0, "github_code_clean", ds_id, lic, cat, is_hard_negative=True)
            if try_add(rec, pool=short_med_pool, count_stats=False):
                used_gh_ids.add(rid)
                diff_added += 1

        rng.shuffle(gh_sm_sel)
        for rid, txt, lic, lang in gh_sm_sel:
            if rid in used_gh_ids:
                continue
            rec = make_record(f"github_clean:{rid}", txt, 0, "github_code_clean", ds_id, lic, f"code_{lang.lower()}", is_hard_negative=True)
            if try_add(rec, pool=short_med_pool, count_stats=False):
                used_gh_ids.add(rid)

        # Interleave deep_mid (~40%) and deep_tail (~60%) so deep_pool preserves both strata cleanly
        gh_deep_combined = gh_deep_mid_sel + gh_deep_tail_sel
        rng.shuffle(gh_deep_combined)
        for rid, txt, lic, lang in gh_deep_combined:
            if rid in used_gh_ids:
                continue
            rec = make_record(f"github_clean:{rid}", txt, 0, "github_code_clean", ds_id, lic, f"code_{lang.lower()}", is_hard_negative=True)
            if try_add(rec, pool=deep_pool, count_stats=False):
                used_gh_ids.add(rid)

        rng.shuffle(gh_base_sel)
        for rid, txt, lic, lang in gh_base_sel:
            if stats["github_code_clean"]["kept_pre_dedup"] >= 1600:
                break
            if rid in used_gh_ids:
                continue
            try_add(make_record(f"github_clean:{rid}", truncate_clean(txt, 10000), 0, "github_code_clean", ds_id, lic, f"code_{lang.lower()}", is_hard_negative=True))
    else:
        gh_long_selected: list[tuple[str, str, str, str]] = []
        gh_short_selected: list[tuple[str, str, str, str]] = []
        long_caps = {"Markdown": 3600, "HTML": 2000, "Python": 2000, "JavaScript": 1400, "TypeScript": 1200, "SQL": 400, "Shell": 500}
        short_caps = {"Markdown": 500, "HTML": 300, "Python": 400, "JavaScript": 300, "TypeScript": 250, "SQL": 250, "Shell": 250}
        for lang, lcap in long_caps.items():
            items = gh_buckets.get(lang, [])
            long_items = [x for x in items if len(x[1]) >= 2600]
            short_items = [x for x in items if 500 <= len(x[1]) < 2600]
            rng.shuffle(long_items)
            rng.shuffle(short_items)
            gh_long_selected.extend(long_items[:lcap])
            gh_short_selected.extend(short_items[: short_caps.get(lang, 200)])

        rng.shuffle(gh_long_selected)
        rng.shuffle(gh_short_selected)
        for rid, txt, lic, lang in gh_long_selected:
            if len(deep_pool) >= 10800:
                break
            rec = make_record(f"github_clean:{rid}", txt, 0, "github_code_clean", ds_id, lic, f"code_{lang.lower()}", is_hard_negative=True)
            excl, _ = guard.is_excluded(txt)
            if excl:
                stats["github_code_clean"]["excluded_heldout"] += 1
                continue
            deep_pool.append(rec)

        for rid, txt, lic, lang in gh_short_selected:
            if stats["github_code_clean"]["kept_pre_dedup"] >= 1600:
                break
            try_add(make_record(f"github_clean:{rid}", truncate_clean(txt, 10000), 0, "github_code_clean", ds_id, lic, f"code_{lang.lower()}", is_hard_negative=True))

    return records, short_med_pool, deep_pool, dict(stats)


def deduplicate_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen_exact: set[str] = set()
    seen_norm: set[str] = set()
    out: list[dict[str, Any]] = []
    removed = 0
    for r in rows:
        txt = r["text"]
        ex = sha256_text(txt)
        nm = sha256_text(normalize_for_dedup(normalize_text(txt)))
        if ex in seen_exact or nm in seen_norm:
            removed += 1
            continue
        seen_exact.add(ex)
        seen_norm.add(nm)
        out.append(r)
    return out, removed


def stratified_group_split(rows: list[dict[str, Any]], dev_frac: float = 0.08) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group-aware stratified 92% train / 8% dev split."""
    rng = random.Random(SEED)
    by_stratum: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_stratum[(int(r["label"]), str(r["source"]))][r["_group_key"]].append(r)

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for _key, groups in sorted(by_stratum.items()):
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        total = sum(len(groups[gk]) for gk in group_keys)
        target_dev = max(1, round(total * dev_frac)) if total >= 12 else 0
        dev_count = 0
        for gk in group_keys:
            items = groups[gk]
            if dev_count < target_dev and (dev_count + len(items) <= target_dev + 2 or dev_count == 0):
                dev_rows.extend(items)
                dev_count += len(items)
            else:
                train_rows.extend(items)
    return train_rows, dev_rows


def build_indirect_pairs(
    carriers: list[dict[str, Any]],
    attacks: list[dict[str, Any]],
    n_pairs: int,
    split_tag: str,
    tokenizer: Any,
    guard: ExclusionGuard,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """v2 indirect pair builder (kept for --version v2)."""
    atk_pool: list[dict[str, Any]] = []
    for a in attacks:
        norm_a = normalize_text(a["text"]).strip()
        if not norm_a:
            continue
        n_at = len(tokenizer(norm_a, add_special_tokens=False, truncation=False)["input_ids"])
        if MIN_ATTACK_TOKENS <= n_at <= MAX_ATTACK_TOKENS:
            atk_pool.append(a)
    if not atk_pool:
        raise RuntimeError(f"Empty token-budgeted attack pool for split={split_tag}")

    eligible_carriers: list[tuple[dict[str, Any], str, int]] = []
    for c in carriers:
        if len(c["text"]) < 1800:
            continue
        trunc_txt, offsets = truncate_to_token_budget(c["text"], tokenizer, MAX_CARRIER_TOKENS)
        if len(offsets) >= MIN_CARRIER_TOKENS:
            eligible_carriers.append((c, trunc_txt, len(offsets)))

    rng.shuffle(eligible_carriers)
    rng.shuffle(atk_pool)

    actual_pairs = min(n_pairs, len(eligible_carriers) // 2)
    benign_long: list[dict[str, Any]] = []
    indirect_rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for c, trunc_txt, n_tok in eligible_carriers:
        if len(benign_long) >= actual_pairs:
            break
        excl, _ = guard.is_excluded(trunc_txt, check_substring=True)
        if excl:
            continue
        used_ids.add(c["id"])
        rec = dict(c)
        rec["text"] = trunc_txt
        rec["char_length"] = len(trunc_txt)
        rec["token_length"] = n_tok
        benign_long.append(rec)

    deep_count = 0
    shallow_count = 0
    max_total_tokens = 0
    atk_idx = 0

    for c, trunc_txt, _n_tok in eligible_carriers:
        if len(indirect_rows) >= actual_pairs:
            break
        if c["id"] in used_ids:
            continue
        k = len(indirect_rows)
        deep = (k % 5) != 0
        atk = atk_pool[atk_idx % len(atk_pool)]
        atk_idx += 1

        res = embed_indirect_attack_token_budgeted(trunc_txt, atk["text"], tokenizer=tokenizer, variant=k % 5, deep=deep, rng=rng)
        if res is None:
            continue
        injected_text, char_offset, tok_offset, total_toks = res
        assert total_toks <= MAX_INDIRECT_TOTAL_TOKENS
        excl, _ = guard.is_excluded(injected_text, check_substring=True)
        if excl:
            continue

        used_ids.add(c["id"])
        rec = make_record(
            f"indirect_{split_tag}:{k}:{c['id']}",
            injected_text,
            1,
            "indirect_injection_composite",
            f"{c['dataset_id']}+{atk['dataset_id']}",
            c["license"],
            f"indirect_{'deep' if deep else 'shallow'}_v{k % 5}",
            is_hard_negative=False,
            group_key=f"ind_{split_tag}_{k}",
        )
        rec["injection_offset_chars"] = char_offset
        rec["injection_offset_tokens"] = tok_offset
        rec["token_length"] = total_toks
        indirect_rows.append(rec)
        if deep:
            deep_count += 1
        else:
            shallow_count += 1
        if total_toks > max_total_tokens:
            max_total_tokens = total_toks

    benign_long = benign_long[: len(indirect_rows)]
    pair_stats = {
        "eligible_carriers": len(eligible_carriers),
        "eligible_attacks": len(atk_pool),
        "pairs_created": len(indirect_rows),
        "deep_count": deep_count,
        "shallow_count": shallow_count,
        "pct_deep": round(deep_count / max(1, len(indirect_rows)), 4),
        "max_total_tokens": max_total_tokens,
        "truncated_at_3584": sum(1 for r in indirect_rows if r["token_length"] > 3584),
    }
    assert pair_stats["truncated_at_3584"] == 0
    return benign_long, indirect_rows, pair_stats


def _quantile(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return s[idx]


def build_indirect_pairs_v3(
    short_carriers: list[dict[str, Any]],
    deep_carriers: list[dict[str, Any]],
    attacks: list[dict[str, Any]],
    n_short_pairs: int,
    n_deep_pairs: int,
    split_tag: str,
    tokenizer: Any,
    guard: ExclusionGuard,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build v3 length-matched benign + indirect pairs across short/med and deep strata."""
    short_atk_pool: list[dict[str, Any]] = []
    deep_atk_pool: list[dict[str, Any]] = []
    for a in attacks:
        norm_a = normalize_text(a["text"]).strip()
        if not norm_a:
            continue
        n_at = len(tokenizer(norm_a, add_special_tokens=False, truncation=False)["input_ids"])
        if MIN_ATTACK_TOKENS <= n_at <= V3_SHORT_MAX_ATTACK_TOKENS:
            short_atk_pool.append(a)
        if MIN_ATTACK_TOKENS <= n_at <= V3_DEEP_MAX_ATTACK_TOKENS:
            deep_atk_pool.append(a)
    if not short_atk_pool or not deep_atk_pool:
        raise RuntimeError(f"Empty v3 attack pool for split={split_tag}")

    rng.shuffle(short_atk_pool)
    rng.shuffle(deep_atk_pool)

    # 1. Prepare short/medium carriers (total 120-1,200 tokens)
    elig_short: list[tuple[dict[str, Any], str, list[tuple[int, int]]]] = []
    for c in short_carriers:
        trunc_txt, offsets = truncate_to_token_budget(c["text"], tokenizer, V3_SHORT_CARRIER_MAX_TOKENS)
        if V3_SHORT_CARRIER_MIN_TOKENS <= len(offsets) <= V3_SHORT_CARRIER_MAX_TOKENS:
            elig_short.append((c, trunc_txt, offsets))
    rng.shuffle(elig_short)

    # 2. Prepare deep carriers (total 1,200-4,040 tokens)
    elig_deep_inj: list[tuple[dict[str, Any], str, list[tuple[int, int]]]] = []
    elig_deep_ben: list[tuple[dict[str, Any], str, int]] = []
    for idx_c, c in enumerate(deep_carriers):
        # Alternate reserving for benign (budget up to 3,980 tok) vs injected (budget up to 3,820 tok)
        # so both classes span [1200, 4040] tokens evenly
        if idx_c % 2 == 0:
            trunc_txt, offsets = truncate_to_token_budget(c["text"], tokenizer, V3_DEEP_BENIGN_MAX_TOKENS)
            if len(offsets) >= V3_DEEP_MIN_TOKENS:
                elig_deep_ben.append((c, trunc_txt, len(offsets)))
        else:
            trunc_txt, offsets = truncate_to_token_budget(c["text"], tokenizer, V3_DEEP_CARRIER_MAX_TOKENS)
            if len(offsets) >= V3_DEEP_CARRIER_MIN_TOKENS:
                elig_deep_inj.append((c, trunc_txt, offsets))

    rng.shuffle(elig_deep_ben)
    rng.shuffle(elig_deep_inj)

    benign_pairs: list[dict[str, Any]] = []
    indirect_rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    # Build short/medium indirects + matched benign
    short_ben: list[dict[str, Any]] = []
    for c, trunc_txt, offsets in elig_short:
        if len(short_ben) >= n_short_pairs:
            break
        if len(offsets) < V3_SHORT_MIN_TOKENS:
            continue
        excl, _ = guard.is_excluded(trunc_txt, check_substring=True)
        if excl:
            continue
        used_ids.add(c["id"])
        rec = dict(c)
        rec["text"] = trunc_txt
        rec["char_length"] = len(trunc_txt)
        rec["token_length"] = len(offsets)
        rec["is_hard_negative"] = True
        short_ben.append(rec)

    short_ind: list[dict[str, Any]] = []
    atk_idx = 0
    for c, trunc_txt, offsets in elig_short:
        if len(short_ind) >= n_short_pairs:
            break
        if c["id"] in used_ids:
            continue
        k = len(short_ind)
        raw_inline = (k % 2) == 0  # Exact 50% NO synthetic wrapper tag, 50% varied wrappers
        atk = short_atk_pool[atk_idx % len(short_atk_pool)]
        atk_idx += 1
        res = embed_indirect_attack_v3(
            trunc_txt,
            offsets,
            atk["text"],
            tokenizer=tokenizer,
            raw_inline=raw_inline,
            wrapper_idx=(k // 2) % 6,
            deep=False,
            rng=rng,
        )
        if res is None:
            continue
        injected_text, char_offset, tok_offset, total_toks, style_tag = res
        assert V3_SHORT_MIN_TOKENS <= total_toks <= V3_SHORT_MAX_TOKENS
        excl, _ = guard.is_excluded(injected_text, check_substring=True)
        if excl:
            continue
        used_ids.add(c["id"])
        rec = make_record(
            f"indirect_{split_tag}_sm:{k}:{c['id']}",
            injected_text,
            1,
            "indirect_injection_composite",
            f"{c['dataset_id']}+{atk['dataset_id']}",
            c["license"],
            f"indirect_short_{style_tag}",
            is_hard_negative=False,
            group_key=f"ind_{split_tag}_sm_{k}",
        )
        rec["injection_offset_chars"] = char_offset
        rec["injection_offset_tokens"] = tok_offset
        rec["token_length"] = total_toks
        short_ind.append(rec)

    short_ben = short_ben[: len(short_ind)]
    if len(short_ind) < n_short_pairs:
        raise RuntimeError(f"Short/medium indirect shortage for {split_tag}: {len(short_ind)} < {n_short_pairs}")

    # Build deep indirects + matched benign
    deep_ben: list[dict[str, Any]] = []
    for c, trunc_txt, n_tok in elig_deep_ben:
        if len(deep_ben) >= n_deep_pairs:
            break
        excl, _ = guard.is_excluded(trunc_txt, check_substring=True)
        if excl:
            continue
        used_ids.add(c["id"])
        rec = dict(c)
        rec["text"] = trunc_txt
        rec["char_length"] = len(trunc_txt)
        rec["token_length"] = n_tok
        rec["is_hard_negative"] = True
        deep_ben.append(rec)

    deep_ind: list[dict[str, Any]] = []
    for c, trunc_txt, offsets in elig_deep_inj:
        if len(deep_ind) >= n_deep_pairs:
            break
        if c["id"] in used_ids:
            continue
        k = len(deep_ind)
        raw_inline = (k % 20) < 7  # 35% NO synthetic wrapper tag, 65% varied realistic wrappers
        atk = deep_atk_pool[atk_idx % len(deep_atk_pool)]
        atk_idx += 1
        res = embed_indirect_attack_v3(
            trunc_txt,
            offsets,
            atk["text"],
            tokenizer=tokenizer,
            raw_inline=raw_inline,
            wrapper_idx=k % 6,
            deep=True,
            rng=rng,
        )
        if res is None:
            continue
        injected_text, char_offset, tok_offset, total_toks, style_tag = res
        assert V3_DEEP_MIN_TOKENS <= total_toks <= V3_DEEP_MAX_TOTAL_TOKENS
        excl, _ = guard.is_excluded(injected_text, check_substring=True)
        if excl:
            continue
        used_ids.add(c["id"])
        rec = make_record(
            f"indirect_{split_tag}_dp:{k}:{c['id']}",
            injected_text,
            1,
            "indirect_injection_composite",
            f"{c['dataset_id']}+{atk['dataset_id']}",
            c["license"],
            f"indirect_deep_{style_tag}",
            is_hard_negative=False,
            group_key=f"ind_{split_tag}_dp_{k}",
        )
        rec["injection_offset_chars"] = char_offset
        rec["injection_offset_tokens"] = tok_offset
        rec["token_length"] = total_toks
        deep_ind.append(rec)

    deep_ben = deep_ben[: len(deep_ind)]
    if len(deep_ind) < n_deep_pairs:
        raise RuntimeError(f"Deep indirect shortage for {split_tag}: {len(deep_ind)} < {n_deep_pairs}")

    benign_pairs = short_ben + deep_ben
    indirect_rows = short_ind + deep_ind

    sm_inj_toks = [r["token_length"] for r in short_ind]
    sm_ben_toks = [r["token_length"] for r in short_ben]
    dp_inj_toks = [r["token_length"] for r in deep_ind]
    dp_ben_toks = [r["token_length"] for r in deep_ben]
    dp_offsets = [r["injection_offset_tokens"] for r in deep_ind]
    truncated_4096 = sum(1 for r in indirect_rows if r["token_length"] > 4096)
    assert truncated_4096 == 0, f"Truncated indirect needles at 4096: {truncated_4096}"

    style_counts = Counter(str(r["category"]) for r in indirect_rows)
    pair_stats = {
        "eligible_short_carriers": len(elig_short),
        "eligible_deep_inj_carriers": len(elig_deep_inj),
        "eligible_deep_ben_carriers": len(elig_deep_ben),
        "short_pairs_created": len(short_ind),
        "deep_pairs_created": len(deep_ind),
        "total_pairs_created": len(indirect_rows),
        "truncated_at_4096": truncated_4096,
        "max_total_tokens": max(r["token_length"] for r in indirect_rows),
        "short_injected_tokens": {
            "min": min(sm_inj_toks),
            "p50": _quantile(sm_inj_toks, 0.5),
            "p90": _quantile(sm_inj_toks, 0.9),
            "max": max(sm_inj_toks),
        },
        "short_benign_tokens": {
            "min": min(sm_ben_toks),
            "p50": _quantile(sm_ben_toks, 0.5),
            "p90": _quantile(sm_ben_toks, 0.9),
            "max": max(sm_ben_toks),
        },
        "deep_injected_tokens": {
            "min": min(dp_inj_toks),
            "p50": _quantile(dp_inj_toks, 0.5),
            "p90": _quantile(dp_inj_toks, 0.9),
            "max": max(dp_inj_toks),
        },
        "deep_benign_tokens": {
            "min": min(dp_ben_toks),
            "p50": _quantile(dp_ben_toks, 0.5),
            "p90": _quantile(dp_ben_toks, 0.9),
            "max": max(dp_ben_toks),
        },
        "deep_needle_offset_tokens": {
            "min": min(dp_offsets),
            "p50": _quantile(dp_offsets, 0.5),
            "p90": _quantile(dp_offsets, 0.9),
            "max": max(dp_offsets),
            "in_2500_3900_tail": sum(1 for o in dp_offsets if 2500 <= o <= 3900),
            "pct_in_2500_3900_tail": round(sum(1 for o in dp_offsets if 2500 <= o <= 3900) / max(1, len(dp_offsets)), 4),
        },
        "short_raw_inline_count": sum(1 for r in short_ind if "_raw_" in r["category"]),
        "deep_raw_inline_count": sum(1 for r in deep_ind if "_raw_" in r["category"]),
        "by_style": dict(sorted(style_counts.items())),
    }
    return benign_pairs, indirect_rows, pair_stats


def verify_zero_overlap(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    guard: ExclusionGuard,
) -> dict[str, Any]:
    """Assert zero exact, normalized, or attack-substring overlap with held_out/ and between train/dev."""
    held_out_exact = 0
    held_out_norm = 0
    held_out_sub = 0

    for split_rows in (train_rows, dev_rows):
        for r in split_rows:
            excl, reason = guard.is_excluded(r["text"], check_substring=True)
            if excl:
                if "exact" in str(reason):
                    held_out_exact += 1
                elif "normalized" in str(reason):
                    held_out_norm += 1
                else:
                    held_out_sub += 1

    train_exact = {sha256_text(r["text"]) for r in train_rows}
    train_norm = {sha256_text(normalize_for_dedup(normalize_text(r["text"]))) for r in train_rows}
    dev_exact_overlap = sum(1 for r in dev_rows if sha256_text(r["text"]) in train_exact)
    dev_norm_overlap = sum(1 for r in dev_rows if sha256_text(normalize_for_dedup(normalize_text(r["text"]))) in train_norm)

    audit = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "held_out_exact_overlaps": held_out_exact,
        "held_out_normalized_overlaps": held_out_norm,
        "held_out_attack_substring_overlaps": held_out_sub,
        "train_dev_exact_overlaps": dev_exact_overlap,
        "train_dev_normalized_overlaps": dev_norm_overlap,
        "passed": (
            held_out_exact == 0
            and held_out_norm == 0
            and held_out_sub == 0
            and dev_exact_overlap == 0
            and dev_norm_overlap == 0
        ),
    }
    if not audit["passed"]:
        raise AssertionError(f"Zero-leakage audit FAILED: {json.dumps(audit, indent=2)}")
    return audit


def strip_internal_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lbls = Counter(int(r["label"]) for r in rows)
    srcs = Counter(str(r["source"]) for r in rows)
    lics = Counter(str(r["license"]) for r in rows)
    hard_negs = sum(1 for r in rows if int(r["label"]) == 0 and bool(r.get("is_hard_negative", False)))
    lengths = [int(r["char_length"]) for r in rows]
    return {
        "total": len(rows),
        "benign_0": lbls.get(0, 0),
        "attack_1": lbls.get(1, 0),
        "hard_negative_0": hard_negs,
        "benign_ratio": round(lbls.get(0, 0) / max(1, len(rows)), 4),
        "mean_char_length": round(sum(lengths) / max(1, len(lengths)), 1),
        "max_char_length": max(lengths) if lengths else 0,
        "by_license": dict(sorted(lics.items())),
        "by_source": dict(sorted(srcs.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zero-leakage Track A dataset recipe")
    parser.add_argument("--version", choices=["v2", "v3"], default="v3")
    args = parser.parse_args()
    version = args.version

    rng = random.Random(SEED)
    reg = load_audit_registry()
    guard = ExclusionGuard()

    print(f"Loading pinned ModernBERT tokenizer ({MODERNBERT_MODEL}@{MODERNBERT_REVISION[:8]})...")
    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL, revision=MODERNBERT_REVISION)

    print(f"Collecting license-audited datasets ({version} recipe)...")
    base_records, short_med_pool, deep_pool, collect_stats = collect_records(reg, guard, version=version)

    print(f"Pre-dedup base={len(base_records)}, short_med_pool={len(short_med_pool)}, deep_pool={len(deep_pool)}")
    base_deduped, base_dup_removed = deduplicate_records(base_records)
    short_deduped, short_dup_removed = deduplicate_records(short_med_pool)
    deep_deduped, deep_dup_removed = deduplicate_records(deep_pool)

    # Enforce strict disjointness across base, short_med_pool, and deep_pool
    seen_norms = {sha256_text(normalize_for_dedup(normalize_text(r["text"]))) for r in base_deduped}
    short_filtered: list[dict[str, Any]] = []
    for r in short_deduped:
        nm = sha256_text(normalize_for_dedup(normalize_text(r["text"])))
        if nm not in seen_norms:
            seen_norms.add(nm)
            short_filtered.append(r)
    deep_filtered: list[dict[str, Any]] = []
    for r in deep_deduped:
        nm = sha256_text(normalize_for_dedup(normalize_text(r["text"])))
        if nm not in seen_norms:
            seen_norms.add(nm)
            deep_filtered.append(r)

    print(
        f"Post-dedup base={len(base_deduped)} (-{base_dup_removed}), "
        f"short_pool={len(short_filtered)} (-{short_dup_removed}), "
        f"deep_pool={len(deep_filtered)} (-{deep_dup_removed})"
    )

    train_base, dev_base = stratified_group_split(base_deduped, dev_frac=0.08)
    train_attacks = [r for r in train_base if int(r["label"]) == 1]
    dev_attacks = [r for r in dev_base if int(r["label"]) == 1]

    if version == "v3":
        train_short_pool, dev_short_pool = stratified_group_split(short_filtered, dev_frac=0.08)
        train_deep_pool, dev_deep_pool = stratified_group_split(deep_filtered, dev_frac=0.08)
        print(
            f"Building v3 token-budgeted indirect pairs: "
            f"train(short={len(train_short_pool)}, deep={len(train_deep_pool)}), "
            f"dev(short={len(dev_short_pool)}, deep={len(dev_deep_pool)})..."
        )
        train_ind_ben, train_ind_pos, train_ind_stats = build_indirect_pairs_v3(
            train_short_pool, train_deep_pool, train_attacks, 2500, 4000, "train", tokenizer, guard, rng
        )
        dev_ind_ben, dev_ind_pos, dev_ind_stats = build_indirect_pairs_v3(
            dev_short_pool, dev_deep_pool, dev_attacks, 192, 308, "dev", tokenizer, guard, rng
        )
    else:
        train_long_pool, dev_long_pool = stratified_group_split(deep_filtered, dev_frac=0.08)
        train_ind_ben, train_ind_pos, train_ind_stats = build_indirect_pairs(
            train_long_pool, train_attacks, 4000, "train", tokenizer, guard, rng
        )
        dev_ind_ben, dev_ind_pos, dev_ind_stats = build_indirect_pairs(
            dev_long_pool, dev_attacks, 350, "dev", tokenizer, guard, rng
        )

    print("train indirect stats:", json.dumps(train_ind_stats))
    print("dev indirect stats:", json.dumps(dev_ind_stats))

    train_all = strip_internal_keys(train_base + train_ind_ben + train_ind_pos)
    dev_all = strip_internal_keys(dev_base + dev_ind_ben + dev_ind_pos)

    train_all, tr_extra_dup = deduplicate_records(train_all)
    train_norm_set = {sha256_text(normalize_for_dedup(normalize_text(r["text"]))) for r in train_all}
    dev_all, dv_extra_dup = deduplicate_records(dev_all)
    dev_all = [r for r in dev_all if sha256_text(normalize_for_dedup(normalize_text(r["text"]))) not in train_norm_set]

    rng.shuffle(train_all)
    rng.shuffle(dev_all)

    audit = verify_zero_overlap(train_all, dev_all, guard)

    # License sanity check: 100% permissive SPDX licenses
    all_lics = {str(r["license"]).lower() for r in train_all + dev_all}
    assert all_lics.issubset(ALLOWED_LICENSES), f"Unexpected license found: {all_lics - ALLOWED_LICENSES}"
    assert not any(r["source"] == "jailbreakv_28k" for r in train_all + dev_all), "jailbreakv_28k must be dropped!"

    spml_pos_sys = sum(1 for r in train_all if r["source"] == "spml" and r["label"] == 1 and r["text"].startswith("[SYSTEM] "))
    spml_pos_tot = sum(1 for r in train_all if r["source"] == "spml" and r["label"] == 1)
    spml_neg_sys = sum(1 for r in train_all if r["source"] == "spml" and r["label"] == 0 and r["text"].startswith("[SYSTEM] "))
    spml_neg_tot = sum(1 for r in train_all if r["source"] == "spml" and r["label"] == 0)
    print(
        f"SPML symmetry check (train): pos [SYSTEM]={spml_pos_sys}/{spml_pos_tot} "
        f"({spml_pos_sys/max(1,spml_pos_tot):.3f}) | neg [SYSTEM]={spml_neg_sys}/{spml_neg_tot} "
        f"({spml_neg_sys/max(1,spml_neg_tot):.3f})"
    )

    if version == "v3":
        train_path = DATASETS_DIR / "train_v3.jsonl"
        dev_path = DATASETS_DIR / "dev_internal_v3.jsonl"
        manifest_path = DATASETS_DIR / "train_provenance_manifest_v3.json"
        audit_path = DATASETS_DIR / "split_overlap_audit_v3.json"
        tr_key, dv_key = "train_v3", "dev_internal_v3"
    else:
        train_path = DATASETS_DIR / "train_v2.jsonl"
        dev_path = DATASETS_DIR / "dev_internal_v2.jsonl"
        manifest_path = DATASETS_DIR / "train_provenance_manifest_v2.json"
        audit_path = DATASETS_DIR / "split_overlap_audit_v2.json"
        tr_key, dv_key = "train_v2", "dev_internal_v2"

    write_jsonl(train_path, train_all)
    write_jsonl(dev_path, dev_all)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recipe_version": version,
        "seed": SEED,
        "tokenizer": f"{MODERNBERT_MODEL}@{MODERNBERT_REVISION}",
        "allowed_licenses": sorted(ALLOWED_LICENSES),
        "forbidden_datasets": sorted(FORBIDDEN_DATASET_IDS),
        "dedup_removed": {
            "base": base_dup_removed,
            "short_med_pool": short_dup_removed,
            "deep_pool": deep_dup_removed,
            "post_composite_train": tr_extra_dup,
            "post_composite_dev": dv_extra_dup,
        },
        "collect_stats": collect_stats,
        tr_key: summarize_split(train_all),
        dv_key: summarize_split(dev_all),
        "indirect_pairs_stats": {"train": train_ind_stats, "dev": dev_ind_stats},
        "spml_symmetry_train": {
            "pos_with_system": spml_pos_sys,
            "pos_total": spml_pos_tot,
            "neg_with_system": spml_neg_sys,
            "neg_total": spml_neg_tot,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print("SUCCESS!")
    print(f"{tr_key}:", json.dumps(manifest[tr_key], indent=2))
    print(f"{dv_key}:", json.dumps(manifest[dv_key], indent=2))
    print(f"split_overlap_audit_{version}:", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
