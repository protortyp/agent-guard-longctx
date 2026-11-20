"""Deterministic pure-Python Unicode obfuscation normalizer for Track A.

Handles:
1. Unicode Tag character decoding (U+E0020-U+E007E -> ASCII) with clean word-boundary spacing.
2. Zero-width, soft-hyphen, invisible operator, and bidi control stripping.
3. Variation selector stripping + emoji-smuggling byte-encoding recovery (>= 8 contiguous selectors).
4. Selective NFKC normalization for fullwidth ASCII and mathematical alphanumeric lookalikes
   without altering legitimate developer symbols (arrows, box-drawing, CJK, accented Latin, math ops).
"""

from __future__ import annotations

import unicodedata

INVISIBLE_CODEPOINTS: frozenset[int] = frozenset(
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
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,  # bidi embeddings & overrides
        0x2060,  # word joiner
        0x2061,
        0x2062,
        0x2063,
        0x2064,  # invisible math operators
        0x2066,
        0x2067,
        0x2068,
        0x2069,  # bidi isolates
        0xFEFF,  # BOM / zero width no-break space
    ]
)

_MIN_VS_SMUGGLE_RUN = 8


def _is_tag_char(cp: int) -> bool:
    return 0xE0000 <= cp <= 0xE007F


def _is_variation_selector(cp: int) -> bool:
    return (0xFE00 <= cp <= 0xFE0F) or (0xE0100 <= cp <= 0xE01EF)


def _should_nfkc_char(cp: int) -> bool:
    """Apply NFKC only to fullwidth ASCII, math alphanumeric homoglyphs, and Latin ligatures."""
    return (
        (0xFF01 <= cp <= 0xFF5E)
        or (0x1D400 <= cp <= 0x1D7FF)
        or (0xFB00 <= cp <= 0xFB06)
    )


def _decode_variation_selector_bytes(cps: list[int]) -> str | None:
    """Decode Butler-style emoji smuggling variation selector byte sequences to ASCII.

    Mapping:
      U+FE00..U+FE0F   -> byte 0x00..0x0F
      U+E0100..U+E01EF -> byte 0x10..0xFF
    Returns decoded string only if len >= 8 and all bytes are printable ASCII (plus tab/LF/CR)
    with at least one non-whitespace printable ASCII character.
    """
    if len(cps) < _MIN_VS_SMUGGLE_RUN:
        return None
    decoded_chars: list[str] = []
    has_visible = False
    for cp in cps:
        if 0xFE00 <= cp <= 0xFE0F:
            b = cp - 0xFE00
        elif 0xE0100 <= cp <= 0xE01EF:
            b = (cp - 0xE0100) + 16
        else:
            return None
        if b in (0x09, 0x0A, 0x0D) or (0x20 <= b <= 0x7E):
            if 0x21 <= b <= 0x7E:
                has_visible = True
            decoded_chars.append(chr(b))
        else:
            return None
    if not has_visible:
        return None
    return "".join(decoded_chars)


def normalize_text(text: str) -> str:
    """Deterministically normalize Unicode obfuscation while preserving developer text."""
    if not text:
        return ""

    out: list[str] = []
    need_trailing_space = False
    n = len(text)
    i = 0

    def _emit_decoded_run(decoded: str) -> None:
        nonlocal need_trailing_space
        if not decoded:
            return
        has_visible = any(not c.isspace() for c in decoded)
        if has_visible and out and not out[-1][-1].isspace() and not decoded[0].isspace():
            out.append(" ")
        out.append(decoded)
        if has_visible and not decoded[-1].isspace():
            need_trailing_space = True
        elif decoded[-1].isspace():
            need_trailing_space = False

    while i < n:
        ch = text[i]
        cp = ord(ch)

        if cp in INVISIBLE_CODEPOINTS:
            i += 1
            continue

        if _is_tag_char(cp):
            decoded_tags: list[str] = []
            while i < n and _is_tag_char(ord(text[i])):
                tcp = ord(text[i])
                if 0xE0020 <= tcp <= 0xE007E:
                    decoded_tags.append(chr(tcp - 0xE0000))
                i += 1
            _emit_decoded_run("".join(decoded_tags))
            continue

        if _is_variation_selector(cp):
            vs_cps: list[int] = []
            while i < n and _is_variation_selector(ord(text[i])):
                vs_cps.append(ord(text[i]))
                i += 1
            smuggled = _decode_variation_selector_bytes(vs_cps)
            if smuggled is not None:
                _emit_decoded_run(smuggled)
            continue

        norm_ch = unicodedata.normalize("NFKC", ch) if _should_nfkc_char(cp) else ch
        if need_trailing_space:
            if norm_ch and not norm_ch[0].isspace():
                out.append(" ")
            need_trailing_space = False
        out.append(norm_ch)
        i += 1

    return "".join(out)


# Alias matching eval_locked.py convention
normalize_unicode = normalize_text
