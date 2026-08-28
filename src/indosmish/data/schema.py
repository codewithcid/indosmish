"""Unified message schema and text normalization used across the pipeline."""
import re
import unicodedata

# Unified record columns produced by build_corpus and consumed everywhere downstream.
COLUMNS = [
    "id",          # stable hash id
    "text",        # message text (as stored, in its `script`)
    "label",       # ham | spam | smishing
    "script",      # roman | native
    "lang",        # ml | hi | en  (primary language; per-language eval axis)
    "source",      # provenance tag: dravidian_codemix | sms_phishing | augmented_* | ...
    "synthetic",   # bool: True for augmented messages (train-only per protocol)
]

LABELS = ["ham", "spam", "smishing"]

# Unicode blocks used for script/language detection.
_MALAYALAM = range(0x0D00, 0x0D80)   # U+0D00–U+0D7F
_DEVANAGARI = range(0x0900, 0x0980)  # U+0900–U+097F (Hindi)


def detect_script(text: str) -> str:
    """native if any Indic-block char (Malayalam or Devanagari) is present, else roman."""
    return "native" if any(ord(c) in _MALAYALAM or ord(c) in _DEVANAGARI
                           for c in str(text)) else "roman"


def detect_lang(text: str, source: str = "") -> str:
    """Primary language tag for the per-language evaluation axis.

    Script is authoritative when a native Indic block is present; otherwise the
    provenance tag disambiguates romanized code-mix (which no detector reads reliably):
    romanized augmented Malayalam vs romanized augmented Hindi look alike in Latin.
    """
    t = str(text)
    if any(ord(c) in _DEVANAGARI for c in t):
        return "hi"
    if any(ord(c) in _MALAYALAM for c in t):
        return "ml"
    # Romanized code-mix: rely on explicit provenance tokens, NOT loose substrings
    # ('phishing' contains 'hi'; 'sms' contains 'ml'-adjacent noise) — match on tags.
    s = source.lower()
    if s.startswith("augmented_ml") or "dravidian" in s or "malayalam" in s:
        return "ml"
    if s.startswith("augmented_hi") or "hindi" in s or "gold_hi" in s:
        return "hi"
    return "en"

_WS = re.compile(r"\s+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def normalize_text(text: str) -> str:
    """Light normalization for storage: NFC, strip zero-width, collapse whitespace.

    Intentionally does NOT lowercase or strip punctuation — casing and symbols
    (URLs, currency, urgency markers) are signal for smishing.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_ZERO_WIDTH)
    text = _WS.sub(" ", text).strip()
    return text


def dedup_key(text: str) -> str:
    """Aggressive key for near-duplicate detection (NOT for storage/model input).

    Lowercase, drop all non-alphanumeric (keeps Latin + Malayalam letters), so that
    'Win FREE prize!!!' and 'win free prize' collapse to the same signature.
    """
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = "".join(ch for ch in text if ch.isalnum())
    return text
