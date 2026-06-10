"""Language-independent structural features used by the i18n risk layer."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


_NUMBER = re.compile(
    r"(?<![\w])[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?\s?%?"
)
_URL = re.compile(r'https?://[^\s<>")]+', re.IGNORECASE)
_DATE_NUMERIC = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)
_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")
_IDENTIFIER = re.compile(
    r"\b(?:doi:\s?10\.\d{4,}/\S+|arxiv:\s?\d{4}\.\d{4,}|"
    r"isbn[:\s]?[\d-]{10,}|pmid:\s?\d+)\b",
    re.IGNORECASE,
)


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。！？؟।])\s*|\n+", text.strip())
    result = [part.strip() for part in parts if part.strip()]
    if len(result) == 1 and len(result[0]) > 400:
        return [result[0][i:i + 400] for i in range(0, len(result[0]), 400)]
    return result or [text.strip()]


def _proper_noun_pairs(text: str) -> int:
    tokens = re.findall(r"\S+", text)
    capitalized = []
    for token in tokens:
        core = token.lstrip("\"'([{")
        capitalized.append(bool(core and unicodedata.category(core[0]) == "Lu"))
    return sum(
        1 for index in range(len(capitalized) - 1)
        if capitalized[index] and capitalized[index + 1]
    )


@dataclass
class StructuralFeatures:
    n_sentences: int
    n_numbers: int
    n_urls: int
    n_dates: int
    n_identifiers: int
    n_proper_pairs: int
    anchor_density: float
    has_strong_custody: bool


def extract_structural(text: str) -> StructuralFeatures:
    text_sentences = sentences(text)
    numbers = len(_NUMBER.findall(text))
    urls = len(_URL.findall(text))
    dates = len(_DATE_NUMERIC.findall(text)) + len(_YEAR.findall(text))
    identifiers = len(_IDENTIFIER.findall(text))
    proper_pairs = _proper_noun_pairs(text)
    anchors = identifiers * 2 + urls * 2 + numbers + dates + proper_pairs
    return StructuralFeatures(
        n_sentences=len(text_sentences),
        n_numbers=numbers,
        n_urls=urls,
        n_dates=dates,
        n_identifiers=identifiers,
        n_proper_pairs=proper_pairs,
        anchor_density=round(anchors / max(1, len(text_sentences)), 3),
        has_strong_custody=identifiers > 0 or urls > 0,
    )


def anchor_check(
    text: str,
    marker_spans: List[Tuple[int, int]],
    window: int = 60,
    extra_hooks: Optional[List[int]] = None,
    exclude_spans: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[int, int]:
    if not marker_spans:
        return 0, 0
    hooks = [
        match.start()
        for pattern in (_NUMBER, _URL, _DATE_NUMERIC, _YEAR, _IDENTIFIER)
        for match in pattern.finditer(text)
    ]
    hooks.extend(extra_hooks or [])
    if exclude_spans:
        hooks = [
            hook for hook in hooks
            if not any(start <= hook < end for start, end in exclude_spans)
        ]
    anchored = sum(
        1 for start, end in marker_spans
        if any(start - window <= hook <= end + window for hook in hooks)
    )
    return anchored, len(marker_spans) - anchored


def detect_script(text: str) -> str:
    counts: Dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            bucket = "latin"
        elif "CYRILLIC" in name:
            bucket = "cyrillic"
        elif any(label in name for label in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            bucket = "cjk"
        elif "ARABIC" in name:
            bucket = "arabic"
        elif "DEVANAGARI" in name:
            bucket = "devanagari"
        else:
            bucket = "other"
        counts[bucket] = counts.get(bucket, 0) + 1
    if not counts:
        return "unknown"
    top = max(counts, key=counts.get)
    return "mixed" if counts[top] / sum(counts.values()) < 0.70 else top
