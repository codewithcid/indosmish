"""Unified message schema and text normalization used across the pipeline."""
import re
import unicodedata

# Unified record columns produced by build_corpus and consumed everywhere downstream.
COLUMNS = [
    "id",          # stable hash id
    "text",        # message text (as stored, in its `script`)
    "label",       # ham | spam | smishing
    "script",      # roman | native
    "source",      # provenance tag: dravidian_spam | sms_phishing | augmented
    "synthetic",   # bool: True for augmented messages (train-only per protocol)
]

LABELS = ["ham", "spam", "smishing"]

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
