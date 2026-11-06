"""
signal_compare.py — cross-document numeric claim extraction and comparison.

Philosophy: extract the minimum invariant representation needed for cross-
reference. Each claim is normalized to a canonical entity key so claims across
documents can be matched even when they use different surface forms
(e.g. "10-Year Treasury Yield: 4.61%" and "10yr Treasury: 4.61%").

Engineering principle (Arch review 2026-11-05, v1.8.1):
Extract -> Identify -> Contextualize -> Compare -> Recommend. Each stage
preserves the evidence supporting its decisions. Recognizing a pattern is
not establishing what the pattern represents: a number means nothing
without its relationship to a measurement, a word does not identify an
entity without context, and a recommendation is not justified merely
because an entity extractor produced a name.

Comparator identity (Sieve gap 2026-11-06, v1.8.5): a benchmark claim is
benchmark + variant + version + shot + metric + score + unit. A bare
entity name is not an identity — "SWE-bench Pro" and "SWE-bench
Verified" are different benchmarks that happen to share a family name,
and pricing tiers (input / output / cached / per-task) are different
claims. Differing versions, shots, or variants block comparison with a
visible note instead of manufacturing a conflict; the same claim
(same benchmark + variant + version + shot + unit + value) present in
two or more docs aligns.

Zero external dependencies.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ── Entity normalization map ──────────────────────────────────────────────────
# (pattern, canonical_key, display_label)
# Order matters: more specific patterns must precede more general ones.
_ENTITY_KEY_MAP: List[Tuple[re.Pattern, str, str]] = [
    # Treasury yields — ordered long→short to avoid partial matches
    (re.compile(r"\b30[-\s]?y(?:ear|r)(?:\s+(?:treasury|T(?:reasury)?))?", re.I),
     "30yr-treasury-yield",  "30-Year Treasury Yield"),
    (re.compile(r"\b10[-\s]?y(?:ear|r)(?:\s+(?:treasury|T(?:reasury)?))?", re.I),
     "10yr-treasury-yield",  "10-Year Treasury Yield"),
    (re.compile(r"\b5[-\s]?y(?:ear|r)(?:\s+(?:treasury|T(?:reasury)?))?", re.I),
     "5yr-treasury-yield",   "5-Year Treasury Yield"),
    (re.compile(r"\b2[-\s]?y(?:ear|r)(?:\s+(?:treasury|T(?:reasury)?))?", re.I),
     "2yr-treasury-yield",   "2-Year Treasury Yield"),
    # Equity indices
    (re.compile(r"\bS&P\s*500\b",                              re.I), "sp500",       "S&P 500"),
    (re.compile(r"\bNasdaq(?:\s+Composite)?\b",                re.I), "nasdaq",      "Nasdaq Composite"),
    (re.compile(r"\bDow(?:\s+Jones(?:\s+Industrial(?:\s+Avg(?:erage)?)?)?)?\b|\bDJIA\b", re.I),
     "dow", "Dow Jones"),
    (re.compile(r"\bRussell\s+2000\b",                         re.I), "russell2000", "Russell 2000"),
    # Commodities (Brent before generic "crude" to avoid mis-keying)
    (re.compile(r"\bBrent(?:\s+crude)?\b",                     re.I), "brent-crude", "Brent Crude"),
    (re.compile(r"\bWTI(?:\s+crude)?\b",                       re.I), "wti-crude",   "WTI Crude"),
    (re.compile(r"\bGold\b",                                   re.I), "gold",        "Gold"),
    # Volatility / currency
    (re.compile(r"\bVIX\b",                                    re.I), "vix",         "VIX"),
    (re.compile(r"\bDXY\b|\bDollar\s+Index\b|U\.S\.\s+Dollar\s+Index", re.I), "dxy", "US Dollar Index"),
    (re.compile(r"\bUSD[/]EUR\b|\bEUR[/]USD\b",               re.I), "eur-usd",     "EUR/USD"),
    (re.compile(r"\bUSD[/]JPY\b|\bJPY[/]USD\b",               re.I), "usd-jpy",     "USD/JPY"),
    # Macro indicators
    (re.compile(r"\bCPI\b",                                    re.I), "cpi",         "CPI"),
    (re.compile(r"\bPPI\b",                                    re.I), "ppi",         "PPI"),
    (re.compile(r"\bPCE\b",                                    re.I), "pce",         "PCE"),
    (re.compile(r"\bPMI\b",                                    re.I), "pmi",         "PMI"),
    (re.compile(r"\bGDP\b",                                    re.I), "gdp",         "GDP"),
    # Benchmark scores (Sieve gap 2026-10-31: benchmark-style numerics were
    # invisible to the market-only map). MMLU-Pro precedes MMLU so the
    # longer name wins; same order rule as the Treasury yields above.
    # GDPx precedes GDP for the same reason (Arch review 2026-11-04):
    # "GDPx" must key as its own benchmark, never collapse to "GDP".
    (re.compile(r"\bGPQA(?:\s+Diamond)?\b",                 re.I), "gpqa",        "GPQA"),
    (re.compile(r"\bSWE[-\s]?bench(?:\s+Verified)?\b",       re.I), "swe-bench",   "SWE-bench"),
    (re.compile(r"\bHumanEval(?:\+)?\b",                     re.I), "humaneval",   "HumanEval"),
    (re.compile(r"\bMMLU[-\s]?Pro\b",                        re.I), "mmlu-pro",    "MMLU-Pro"),
    (re.compile(r"\bMMLU\b",                                 re.I), "mmlu",        "MMLU"),
    (re.compile(r"\bMMMU\b",                                 re.I), "mmmu",        "MMMU"),
    (re.compile(r"\bAIME\b",                                 re.I), "aime",        "AIME"),
    (re.compile(r"\bGSM8K\b",                                re.I), "gsm8k",       "GSM8K"),
    (re.compile(r"\bMATH(?:[-\s]?500)?\b",                   re.I), "math",        "MATH"),
    (re.compile(r"\bARC[-\s]?AGI\b",                         re.I), "arc-agi",     "ARC-AGI"),
    (re.compile(r"\bHumanity'?s\s+Last\s+Exam\b",             re.I), "hle",         "Humanity's Last Exam"),
    # v1.8.5 (Nov 6): the abbreviation HLE is the same benchmark — it
    # also makes "HLE-Diamond 28.7%" visible, where the full name never
    # appears. Terminal-Bench, PostTrainBench and the other table rows
    # stay unmapped and invisible (documented limit).
    (re.compile(r"\bHLE\b",                                  re.I), "hle",         "Humanity's Last Exam"),
    (re.compile(r"\bGDPx\b",                                 re.I), "gdpx",        "GDPx"),
    # Initial/jobless claims (keep last to not over-match)
    (re.compile(r"\b(?:initial\s+)?jobless\s+claims?\b|\binitial\s+claims?\b", re.I),
     "jobless-claims", "Jobless Claims"),
    (re.compile(r"\bretail\s+sales?\b",                        re.I), "retail-sales", "Retail Sales"),
]

# Crude benchmarks are related but different contracts — flag, don't call it a conflict
_CRUDE_KEYS = frozenset({"wti-crude", "brent-crude"})

# ── Numeric claim extraction patterns ────────────────────────────────────────

# High-confidence colon-separated data lines:
#   "10-Year Treasury Yield: 4.61%"  or  "WTI Crude: $68.43/bbl"
# The trailing lookahead keeps a shot count out of the value slot:
# "MMLU: 5-shot 93.3" must not mint a claim of value 5 — the 5 is a
# measurement condition, not a result (Arch review 2026-11-05, v1.8.1).
_COLON_RE = re.compile(
    r"([A-Za-z&][^:\n]{2,55}):"          # label (2–55 chars)
    r"\s*"
    r"([+-]?\$?\d[\d,\.]*"               # leading digit (with optional sign/$)
    r"(?:%|bps|(?:\s*/bbl)|(?:\s*/oz)|(?:\s+per\s+(?:barrel|oz(?:ounce)?)))?"
    r"(?:\s*[-–]\s*\d[\d,\.]*(?:%|bps)?)?)"  # optional range "4.25-4.75"
    r"(?!\s*-\s*shot\b)",
    re.I,
)

# Lower-confidence inline patterns: "S&P 500 +0.02%" or "S&P 500 reached 5,894"
# Whole-word guard (v1.8.1, gap 3): the entity alternation is followed by
# (?!\w), so "Downloads"/"downloadable" can never match "Dow", "Goldman"
# can never match "Gold", and "MATHEMATICS" can never match "MATH".
# Number guards (v1.8.1, gap 2): the value must not be embedded in another
# token — (?<![A-Za-z0-9@]) blocks the "3" in "Hy3", the "8" in "FP8",
# the "1" in "Pass@1" — and must not be a shot count — the trailing
# lookahead blocks the "5" in "5-shot" so the regex finds the real score
# ("93.3") instead.
_INLINE_ENTITY_RE = re.compile(
    r"\b(S&P\s*500|Nasdaq(?:\s+Composite)?|Dow(?:\s+Jones)?|DJIA|Russell\s+2000|"
    r"\d{1,2}[-\s]?y(?:ear|r)(?:\s+Treasury)?|"
    r"WTI(?:\s+crude)?|Brent(?:\s+crude)?|Gold|VIX|DXY|"
    r"CPI|PPI|PCE|PMI|GDPx|GDP|"
    r"GPQA(?:\s+Diamond)?|SWE[-\s]?bench(?:\s+Verified)?|HumanEval(?:\+)?|"
    r"MMLU[-\s]?Pro|MMLU|MMMU|AIME|GSM8K|MATH(?:[-\s]?500)?|ARC[-\s]?AGI|Humanity'?s\s+Last\s+Exam|HLE|"
    r"(?:initial\s+)?jobless\s+claims?|retail\s+sales?)(?!\w)"
    r"[^.!?\n]{0,70}?"
    r"(?<![A-Za-z0-9@])([+-]?\$?\d[\d,\.]*"
    r"(?:%|\s*bps|\s*/bbl|\s*/oz|\s+per\s+(?:barrel|oz)|\s*,\d{3})*)"
    r"(?!\s*-?\s*shot\b)",
    re.I,
)

# Benchmark-reverse pass: "61.3% on GPQA" — the score precedes the
# benchmark name, so the forward inline pattern cannot see it. Restricted
# to benchmark names + a % value so prose percentages never match.
_BENCHMARK_NAMES_RE = (
    r"GPQA(?:\s+Diamond)?|SWE[-\s]?bench(?:\s+Verified)?|HumanEval(?:\+)?|"
    r"MMLU[-\s]?Pro|MMLU|MMMU|AIME|GSM8K|MATH(?:[-\s]?500)?|ARC[-\s]?AGI|Humanity'?s\s+Last\s+Exam|HLE|"
    r"GDPx"
)
_BENCHMARK_REVERSE_RE = re.compile(
    r"(\d+\.?\d*\s*%)[^.!?\n]{0,40}?\b(" + _BENCHMARK_NAMES_RE + r")\b",
    re.I,
)

# Pricing-row pass (Sieve gap 2026-11-01: a $15/$75 pricing row vs no row
# was invisible because pricing was not an entity). A $ amount counts as
# a pricing claim only when pricing words sit nearby, so market prices
# ("WTI Crude: $68.43/bbl") never match. Input/output words split the
# key so an input price and an output price are never conflated.
_PRICING_WORDS_RE = re.compile(
    r"\b(pric(?:e|ing)|per\s+(?:million|1M)\s+tokens?|tokens?|per[-\s]?task)\b", re.I)
_INPUT_WORD_RE = re.compile(r"\binput\b", re.I)
_OUTPUT_WORD_RE = re.compile(r"\boutput\b", re.I)
# v1.8.5 (Nov 6): "cached" and "per task" are their own pricing tiers —
# $0.03 cached input is not an input price and $1.18/task is not an
# output price, and "per task" must also count as a pricing word so a
# per-task price mints at all. The per-task regex requires "per" so
# "per index task" (a token count, not a price) never matches.
_CACHED_WORD_RE = re.compile(r"\bcached\b", re.I)
_PER_TASK_RE = re.compile(r"\bper[-\s]?task\b|\b/task\b", re.I)
# v1.8.5 (Nov 6): a tier word immediately before a $ amount — "input
# $15", "Input: $0.30" — is that amount's own label. Only whitespace
# or a colon may sit between the word and the $, so "output and $0.03"
# (the previous price's tier) never matches.
_BACKWARD_TIER_RE = re.compile(r"\b(input|output|cached)\b\s*:?\s*$", re.I)
_DOLLAR_RE = re.compile(r"\$\s?(\d[\d,\.]*)")

# ── Date claims (Sieve gap 2026-11-02: rollout stories live on dates) ──────
# Full dates normalize to ISO; bare years stay years — a year is not a
# date, and a full date must not also mint a year claim (else every dated
# doc trivially "shares" the year with every other dated doc).
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = (
    r"January|February|March|April|May|June|July|August|September|Sept|"
    r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|"
    r"Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?"
)
_DATE_FULL_RE = re.compile(
    r"\b(" + _MONTH_ALT + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
    r"|\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_ALT + r")\s+(\d{4})\b"
    r"|\b(\d{4})-(\d{2})-(\d{2})\b",
    re.I,
)
_YEAR_RE = re.compile(
    r"\b(?:in|from|by|during|starting|effective|early|late|mid)[\s\-]+(20\d{2})\b",
    re.I,
)


# Benchmark variant identity (Sieve gap 2026-11-06, v1.8.5): a benchmark
# claim is benchmark + variant + version + shot + metric + score + unit.
# "SWE-bench Pro" and "SWE-bench Verified" are different benchmarks that
# happen to share a family name — comparing their scores manufactures a
# conflict. The variant is the first qualifier token after the base
# benchmark name ("Pro", "Verified", "Diamond", "Standard", "Hard",
# "text", ...); "HumanEval+" carries its "+" as the variant. A claim
# that names no variant is variant "" and is compared provisionally,
# mirroring the shot-count gate. MATH vs MATH-500 stay one series (no
# "500" variant token); AIME year-variants (2024/2025) are not yet
# distinguished — both are documented limits.
_VARIANT_TOKENS = (
    "pro", "verified", "diamond", "standard", "hard", "text", "plus",
    "lite", "flash", "mini", "base", "instruct", "chat", "reasoning",
)
_VARIANT_RE = re.compile(r"\b(" + "|".join(_VARIANT_TOKENS) + r")\b", re.I)


def _benchmark_variant(key: str, match_text: str, between_text: str = "", after_text: str = "") -> str:
    """Variant qualifier for a benchmark claim, or "" when unstated.

    match_text is the entity pattern's matched text ("GPQA Diamond",
    "SWE-bench Verified", "MMLU-Pro", "HumanEval+"); between_text is
    the fragment between the entity match and the value (inline/colon
    passes); after_text is up to ~20 chars after the entity match
    (reverse pass, where "(text)" follows the name). The FIRST qualifier
    token wins — it is the one attached to this benchmark's name, not a
    neighbour's. "+" in the match text means the "+" variant
    (HumanEval+ is a different test set from HumanEval). Non-benchmark
    keys have no variant.
    """
    if key not in _BENCHMARK_KEYS:
        return ""
    if "+" in match_text:
        return "+"
    m = _VARIANT_RE.search(match_text + between_text + after_text[:20])
    return m.group(1).lower() if m else ""


def _normalize_entity_key(fragment: str) -> Optional[Tuple[str, str, "re.Match[str]"]]:
    """Return (canonical_key, display_label, match) for the first matching entity pattern."""
    for pattern, key, label in _ENTITY_KEY_MAP:
        m = pattern.search(fragment)
        if m:
            return key, label, m
    return None


def _parse_value(raw: str) -> Optional[float]:
    """Parse a raw value string to float.

    Handles: "4.61%", "$68.43", "68.43/bbl", "4.25-4.75" (→ midpoint), "5,894".
    Returns None if unparseable.
    """
    clean = (raw
             .replace(",", "")
             .replace("$", "")
             .replace("/bbl", "")
             .replace("/oz", "")
             .strip())
    clean = re.sub(r"\s+per\s+(?:barrel|oz(?:ounce)?)", "", clean, flags=re.I).strip()
    # A sentence-final period is punctuation, not part of the number:
    # the extraction patterns' [\d,\.]* class eats it ("93.3."), and
    # float() rejects a two-dot literal. Strip it here (v1.8.1).
    clean = clean.rstrip(".")
    # Range: "4.25-4.75" or "4.25–4.75" → midpoint
    range_m = re.match(r"([+-]?\d+\.?\d*)[-–](\d+\.?\d*)", clean)
    if range_m:
        try:
            lo, hi = float(range_m.group(1)), float(range_m.group(2))
            return round((lo + hi) / 2.0, 5)
        except ValueError:
            pass
    try:
        return float(clean.rstrip("% "))
    except ValueError:
        return None


def _unit_of(raw: str) -> str:
    if "%" in raw:
        return "%"
    if "bps" in raw:
        return "bps"
    if "$" in raw or "/bbl" in raw or "/oz" in raw:
        return "$"
    return ""


# Benchmark entity keys — the only keys that carry a version. Arch review
# 2026-11-04: two claims must concern the same metric, subject, unit, AND
# benchmark version before they compare. A version token adjacent to the
# benchmark name is recorded; when both sides name versions and they
# differ, the pair is version-mismatched, never a conflict or an
# alignment. When neither side (or only one side) names a version,
# sameness cannot be disproven, so the pair still compares — the version
# field on each value lets the caller see what was (not) established.
_BENCHMARK_KEYS = frozenset({
    "gpqa", "swe-bench", "humaneval", "mmlu-pro", "mmlu", "mmmu", "aime",
    "gsm8k", "math", "arc-agi", "hle", "gdpx",
})
_VERSION_RE = re.compile(r"\bv(\d+(?:\.\d+)?)\b", re.I)
# A bare number followed by a count word is a count, not an unspecified
# unit. This is what the Nov 3 false conflict missed: the counts
# (206 tasks, 13 models) came through the inline pass, which had no
# count detection, so they read as unit "" and "compared" with 12%.
_COUNT_WORDS_RE = re.compile(
    r"\b(problems|questions|items|examples|tasks|samples|models|agents|systems|entries|runs)\b",
    re.I,
)


def _benchmark_version(key: str, window: str) -> str:
    if key not in _BENCHMARK_KEYS:
        return ""
    m = _VERSION_RE.search(window)
    return f"v{m.group(1)}" if m else ""


def _count_after(text: str, pos: int, window: int = 25) -> bool:
    return bool(_COUNT_WORDS_RE.search(text[pos:pos + window]))


# Shot counts (Arch review 2026-11-05, v1.8.1): a shot count is a
# measurement-condition attribute, not a performance result. A benchmark
# claim is approximately benchmark + model + benchmark version + shot
# count + metric + score + unit, with the source excerpt preserved
# (the claim's "context" field). "MMLU 5-shot: 93.3" yields score 93.3
# with shot_count=5 — never a claim of value 5. Word forms
# ("five-shot", "one-shot", "zero-shot") count the same as digits.
_SHOT_RE = re.compile(r"(\d+)\s*-\s*shot\b", re.I)
_SHOT_WORD_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\s*-\s*shot\b", re.I)
_SHOT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _shot_count_in_window(window: str) -> Optional[int]:
    """First shot count mentioned in a window, or None when unstated."""
    m = _SHOT_RE.search(window)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    m = _SHOT_WORD_RE.search(window)
    if m:
        return _SHOT_WORDS[m.group(1).lower()]
    return None


def _shot_count_for_match(text: str, m: "re.Match[str]", num_group: int) -> Optional[int]:
    """Shot count for one extracted claim.

    Prefer a shot count inside the claim's own matched fragment (the
    "5-shot" in "MMLU (EM) 5-shot: ERNIE 93.3"); only when the fragment
    carries none ("MMLU: 93.3 (5-shot)") fall back to the nearest shot
    mention in the surrounding ±80 chars. Taking the FIRST shot in a
    wide window is wrong: a table line's neighbours carry other
    benchmarks' shot counts (the Nov 5 card's HumanEval line sits
    under the GPQA 5-shot line, and must read 1-shot, not 5-shot).
    """
    frag = text[m.start():m.end()]
    shot = _shot_count_in_window(frag)
    if shot is not None:
        return shot
    win_start = max(0, m.start() - 80)
    win_end = m.end() + 80
    window = text[win_start:win_end]
    num_pos = m.start(num_group) - win_start
    best: Optional[int] = None
    best_dist: Optional[int] = None
    for regex, conv in ((_SHOT_RE, int), (_SHOT_WORD_RE, lambda w: _SHOT_WORDS[w.lower()])):
        for hit in regex.finditer(window):
            try:
                value = conv(hit.group(1))
            except (ValueError, KeyError):
                continue
            dist = abs(hit.start() - num_pos)
            if best_dist is None or dist < best_dist:
                best, best_dist = value, dist
    return best


# A bare number with an optional unit, for the setting-number rescan
# (v1.8.5): the same shape as the inline pass's value group, minus the
# entity prefix. The trailing lookahead blocks shot counts ("5-shot",
# "5 shot") so a rescan never mints a shot count as a score.
_RESCAN_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9@])([+-]?\$?\d[\d,\.]*(?:%|\s*bps|\s*/bbl|\s*/oz|\s+per\s+(?:barrel|oz)|\s*,\d{3})*)(?!\s*-?\s*shot\b)"
)


def _rescan_inline_score(text: str, m: "re.Match[str]", key: str, label: str, variant: str, _add) -> None:
    """After a setting number, find the real score later in the same line.

    "MMMU Pro (Standard 10) reaches 81.6 percent" — the inline pass
    meets the 10 first; the 81.6% is the score. Scan the rest of the
    line (up to 80 chars) for the next number that is not itself a
    setting or version number and not a following-benchmark score, and
    mint it under this entity and variant. At most 3 candidates; if
    none qualifies, the score stays invisible rather than manufacturing
    a claim.
    """
    line_end = text.find("\n", m.end())
    limit = m.end() + 80 if line_end == -1 else min(line_end, m.end() + 80)
    window = text[m.end():limit]
    tried = 0
    for nm in _RESCAN_NUMBER_RE.finditer(window):
        if tried >= 3:
            return
        tried += 1
        num_start = m.end() + nm.start(1)
        num_end = m.end() + nm.end(1)
        raw_val = nm.group(1).strip()
        if not raw_val or raw_val == ".":
            continue
        if _is_version_number(text, num_start):
            continue
        if _is_setting_number(text, num_start):
            continue
        if key in _BENCHMARK_KEYS and _is_following_benchmark_score(text, num_end):
            continue
        bare = _unit_of(raw_val) == ""
        count_after = bare and _count_after(text, num_end)
        _add(key, label, raw_val, text[m.start():num_end],
             unit_override="count" if count_after else "",
             version=_benchmark_version(key, text[m.start():num_end]),
             shot_count=_shot_count_in_window(text[m.start():num_end]),
             variant=variant)
        return


# Market entities (Arch review 2026-11-05, v1.8.1): a market entity must
# have positive market context before it fires. Whole-word matching
# alone is not sufficient evidence of entity identity — a document
# about downloading language models must never acquire stock-market
# interpretations because a short financial abbreviation appears inside
# another word. The gate is document-level: a genuine market snapshot
# carries market vocabulary somewhere; a download page does not.
# Treasury yields and macro indicators (CPI/PPI/GDP) are NOT gated:
# their names cannot be confused with ordinary words.
_MARKET_ENTITY_KEYS = frozenset({
    "sp500", "nasdaq", "dow", "russell2000", "vix", "dxy",
    "eur-usd", "usd-jpy", "gold", "brent-crude", "wti-crude",
})
_MARKET_CONTEXT_RE = re.compile(
    r"\b(stock|stocks|market|markets|index|indices|trading|traded|"
    r"shares?|investors?|exchange|nyse|nasdaq|dow\s+jones|djia|"
    r"points|futures|equities|bonds?|yields?|"
    r"treasury|treasuries|crude|oil|wall\s+street|s&p)\b",
    re.I,
)


_VERSION_PREFIX_RE = re.compile(r"(?:\bv|version)\s*$", re.I)


def _is_version_number(text: str, num_start: int) -> bool:
    """True when a matched number is a version token, not a value.

    "GDPx v1.0" must not mint a claim of value 1.0: the digits belong
    to the version, and the version is recorded on the real claim (if
    any) instead. Without this, every versioned benchmark mention
    injects a spurious 1.0/2.0 claim into the comparison.
    """
    return bool(_VERSION_PREFIX_RE.search(text[max(0, num_start - 10):num_start]))


# v1.8.5 (Nov 6): a number that is the score OF a following benchmark
# ("82.2% on SWE-bench Verified", "44.1% on Humanity's Last Exam",
# "13.1% on Terminal-Bench 4.0") belongs to that following name, not
# to the entity the forward scan started from. The reverse pass mints
# it correctly (or it stays invisible when the following benchmark is
# unmapped) — the forward pass must not claim it for the wrong
# benchmark. Benchmark keys only, and scoped to the demonstrated
# failure: day/month names after "on" are excluded (a date is not a
# benchmark), "on the…" never triggers, and a sentence break between
# the number and the "on" blocks the guard (only spaces/tabs may
# intervene, no period or newline).
_FOLLOWING_BENCH_RE = re.compile(
    r"[ \t]*(?:percent[ \t]+)?on[ \t]+"
    r"(?!Monday\b|Tuesday\b|Wednesday\b|Thursday\b|Friday\b|Saturday\b|Sunday\b"
    r"|January\b|February\b|March\b|April\b|May\b|June\b|July\b|August\b"
    r"|September\b|October\b|November\b|December\b)"
    r"[A-Z]"
)


def _is_following_benchmark_score(text: str, num_end: int) -> bool:
    """True when a matched number is the score of a following benchmark."""
    return bool(_FOLLOWING_BENCH_RE.match(text, num_end))


# v1.8.5 (Nov 6): a bare number that is a measurement setting, not a
# result — "MMMU Pro (Standard 10)" — the 10 is the setting, and the
# score (81.6%) comes later in the same line. Minting the 10 as a
# claim manufactures a conflict with the real score. A number
# immediately preceded by a setting word is a setting, not a value.
_SETTING_WORD_BEFORE_RE = re.compile(r"(standard|example|sample)\s*$", re.I)


def _is_setting_number(text: str, num_start: int) -> bool:
    """True when a number is a measurement setting, not a score."""
    return bool(_SETTING_WORD_BEFORE_RE.search(text[max(0, num_start - 24):num_start]))


def _label_regex(label: str) -> str:
    """Regex for an entity label with flexible whitespace ("S&P 500" etc.)."""
    tokens = [re.escape(t) for t in re.split(r"\s+", label.strip()) if t]
    return r"\s+".join(tokens)


def _has_explicit_absence(text: str, label: str) -> bool:
    """True when the text EXPLICITLY says this entity is absent.

    Only explicit absence language naming the entity counts: "no <entity>
    row", "<entity> not listed/shown/included/present/available", "without
    a <entity> row". A document that merely lacks the entity is NOT
    evidence of absence — that is NOT_OBSERVED, never EXPLICITLY_ABSENT
    (Arch design note 2026-11-01).
    """
    lr = _label_regex(label)
    patterns = [
        rf"\bno\s+(?:separate\s+)?{lr}\s+row\b",
        rf"\b{lr}\s+not\s+(?:listed|shown|included|present|available)\b",
        rf"\bwithout\s+(?:a\s+)?{lr}\s+row\b",
    ]
    return any(re.search(p, text, re.I) for p in patterns)


def extract_date_claims(text: str) -> List[Dict]:
    """Extract normalized date claims from text.

    Full dates → ISO "YYYY-MM-DD" (kind "date"); bare years → "YYYY"
    (kind "year"). A full date never also mints a year claim: the year
    inside "February 1, 2027" is part of that date, not a separate
    year-only claim, or every dated document would trivially share every
    year with every other dated document. Deduplicates on date_key.
    Returns [{date_key, date_label, kind, raw, context}].
    """
    claims: List[Dict] = []
    seen: set = set()
    # Mask full-date spans so the year pass cannot re-read their years.
    masked = list(text)
    for m in _DATE_FULL_RE.finditer(text):
        if m.group(7):
            year, month, day = int(m.group(7)), int(m.group(8)), int(m.group(9))
        elif m.group(1):
            month = _MONTHS[m.group(1).lower().rstrip(".")]
            day, year = int(m.group(2)), int(m.group(3))
        else:
            day = int(m.group(4))
            month = _MONTHS[m.group(5).lower().rstrip(".")]
            year = int(m.group(6))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        key = f"{year:04d}-{month:02d}-{day:02d}"
        if key in seen:
            continue
        seen.add(key)
        for i in range(m.start(), m.end()):
            masked[i] = " "
        claims.append({
            "date_key":   key,
            "date_label": key,
            "kind":       "date",
            "raw":        m.group(0).strip(),
            "context":    text[max(0, m.start() - 40):m.end() + 40].strip()[:120],
        })
    masked_text = "".join(masked)
    for m in _YEAR_RE.finditer(masked_text):
        key = m.group(1)
        if key in seen:
            continue
        seen.add(key)
        claims.append({
            "date_key":   key,
            "date_label": key,
            "kind":       "year",
            "raw":        m.group(0).strip(),
            "context":    text[max(0, m.start() - 40):m.end() + 40].strip()[:120],
        })
    return claims


# ── Timeline reconciliation (Arch review 2026-11-04) ───────────────────────
# A SEPARATE DIAGNOSTIC, not a conflict channel. Rollout stories live on
# dates, and "February 1, 2027" vs "early 2027" vs "in the coming months"
# are not necessarily contradictory — they may describe different stages,
# or the same stage at different precisions. This channel extracts the
# event-adjacent date expressions per doc and flags shared-year frames
# where expressions differ, so a caller can ask for clarification instead
# of inheriting a false conflict (or a false agreement from a shared year).
_VAGUE_TIMING_RE = re.compile(
    r"\b(in the coming (?:weeks|months)|coming (?:weeks|months)|"
    r"later this year|later next year)\b",
    re.I,
)


def _timeline_expressions(text: str) -> List[Dict]:
    """Timeline expressions in one doc: exact dates, year-only claims, vague phrases."""
    exprs: List[Dict] = []
    for c in extract_date_claims(text):
        exprs.append({
            "expression": c["date_label"],
            "precision": "exact" if c["kind"] == "date" else "year_only",
            "raw": c["raw"],
        })
    for m in _VAGUE_TIMING_RE.finditer(text):
        exprs.append({"expression": m.group(0), "precision": "vague", "raw": m.group(0)})
    return exprs


def reconcile_timelines(analyzed_docs: List[Dict]) -> List[Dict]:
    """Flag shared-year frames where timeline expressions differ.

    A doc participates in a year frame through its exact dates in that
    year or its year-only claims; its vague phrases ("coming months")
    ride along on every frame it participates in. A frame fires only
    when 2+ docs participate AND their expressions are not all
    identical — same date at same precision is agreement, not a note.
    Returns [{year, docs: {name: [expressions]}, note}].
    """
    per_doc = [_timeline_expressions(doc.get("text", "")) for doc in analyzed_docs]
    doc_years: List[set] = []
    for exprs in per_doc:
        years = set()
        for e in exprs:
            if e["precision"] == "exact":
                years.add(e["expression"][:4])
            elif e["precision"] == "year_only":
                years.add(e["expression"])
        doc_years.append(years)
    shared_years: set = set()
    for i in range(len(analyzed_docs)):
        for j in range(i + 1, len(analyzed_docs)):
            shared_years |= (doc_years[i] & doc_years[j])
    notes: List[Dict] = []
    for year in sorted(shared_years):
        docs_with: Dict[str, List[Dict]] = {}
        for idx, doc in enumerate(analyzed_docs):
            if year not in doc_years[idx]:
                continue
            exprs = [e for e in per_doc[idx]
                     if (e["precision"] == "exact" and e["expression"].startswith(year))
                     or (e["precision"] == "year_only" and e["expression"] == year)
                     or e["precision"] == "vague"]
            if exprs:
                docs_with[doc.get("source_name", f"doc_{idx + 1}")] = exprs
        if len(docs_with) < 2:
            continue
        distinct = {(e["expression"], e["precision"]) for es in docs_with.values() for e in es}
        if len(distinct) <= 1:
            continue
        notes.append({
            "year": year,
            "docs": docs_with,
            "note": (
                "Timeline expressions for this year differ in date or precision "
                "across sources; they may describe different stages, not a "
                "disagreement. Needs clarification, not a conflict."
            ),
        })
    return notes


# ── Public API ────────────────────────────────────────────────────────────────

def extract_numeric_claims(
    text: str,
    dropped_out: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Extract normalized numeric claims from text.

    Returns a list of::

        {
          "entity_key":    "10yr-treasury-yield",
          "entity_label":  "10-Year Treasury Yield",
          "raw_value":     "4.61%",
          "numeric_value": 4.61,
          "unit":          "%",
          "shot_count":    5,        # benchmark claims only; None when unstated
          "variant":       "",       # benchmark variant ("pro", "verified", ...); "" when unstated (v1.8.5)
          "context":       "10-Year Treasury Yield: 4.61%",
        }

    Deduplicates on (entity_key, numeric_value, variant) pairs (v1.8.5:
    the same key and value in different variants are different claims).
    When the same entity + value is mentioned twice and only one
    mention states a shot count, the surviving claim keeps the stated
    count — the condition is evidence, not a second claim.

    Market claims dropped by the context gate (v1.8.1) are appended to
    *dropped_out* when the caller passes a list, so the caller can
    audit the drop instead of silently losing the evidence.

    The claim's "context" field preserves the source excerpt (Arch
    review 2026-11-05: each stage preserves the evidence supporting
    its decisions).
    """
    claims: List[Dict] = []
    seen: Dict[Tuple[str, float, str], int] = {}  # (entity_key, numeric_value, variant) -> index into claims
    dropped: List[Dict] = []

    def _add(key: str, label: str, raw_val: str, context: str, unit_override: str = "",
             version: str = "", shot_count: Optional[int] = None, variant: str = "") -> None:
        numeric = _parse_value(raw_val)
        if numeric is None:
            return
        # Market-context gate (v1.8.1, gap 3): a market entity with no
        # positive market context anywhere in the document is not a
        # claim at all. Record the drop for the caller to audit.
        if key in _MARKET_ENTITY_KEYS and not _MARKET_CONTEXT_RE.search(text):
            dropped.append({
                "entity_key":   key,
                "entity_label": label,
                "raw_value":    raw_val.strip(),
                "reason":       "no positive market context in document",
            })
            return
        unit    = unit_override or _unit_of(raw_val)
        # v1.8.5: a claim is key + value + variant. Two claims for the
        # same key and value in different variants are DIFFERENT claims
        # (Pro vs Verified) and must not dedup into one.
        dedup   = (key, numeric, variant)
        if dedup in seen:
            idx = seen[dedup]
            if claims[idx].get("shot_count") is None and shot_count is not None:
                claims[idx]["shot_count"] = shot_count
            return
        seen[dedup] = len(claims)
        claims.append({
            "entity_key":    key,
            "entity_label":  label,
            "raw_value":     raw_val.strip(),
            "numeric_value": numeric,
            "unit":          unit,
            "version":       version,
            "shot_count":    shot_count,
            "variant":       variant,
            "context":       context[:120],
        })

    # Pass 1 — colon-delimited data lines (higher precision)
    for m in _COLON_RE.finditer(text):
        label_fragment = m.group(1).strip()
        raw_val        = m.group(2).strip()
        entity = _normalize_entity_key(label_fragment)
        if entity is None:
            continue
        key, label, ematch = entity
        # A bare number followed by a count word ("GPQA: 94 problems") is
        # a count, not an unspecified unit — without this, a count and a
        # percentage on the same entity read as one comparable series.
        # Count detection applies to bare numbers only: a percentage
        # followed by a count word ("12% on 206 tasks") is still a %.
        bare = _unit_of(raw_val) == ""
        count_after = bare and _count_after(text, m.end())
        _add(key, label, raw_val, m.group(0),
             unit_override="count" if count_after else "",
             version=_benchmark_version(key, m.group(0)),
             shot_count=_shot_count_for_match(text, m, 2),
             variant=_benchmark_variant(key, ematch.group(0), label_fragment[ematch.end():]))

    # Pass 2 — inline patterns (catches prose and snapshot lines)
    for m in _INLINE_ENTITY_RE.finditer(text):
        entity_frag = m.group(1).strip()
        raw_val     = m.group(2).strip()
        if not raw_val or raw_val == ".":
            continue
        if _is_version_number(text, m.start(2)):
            continue
        entity = _normalize_entity_key(entity_frag)
        if entity is None:
            continue
        key, label, ematch = entity
        # Variant identity: first qualifier after the base name, in the
        # entity match text + the span up to the value. ("SWE-bench Pro
        # v2" matches only "SWE-bench" in the alternation, so the "Pro"
        # lives in the span — the search covers both.)
        frag = text[m.start():m.end()]
        between = frag[ematch.end():m.start(2) - m.start()]
        variant = _benchmark_variant(key, ematch.group(0), between)
        if key in _BENCHMARK_KEYS and _is_setting_number(text, m.start(2)):
            # The number is a measurement setting, not a score — look
            # for the real score later in the same line instead of
            # minting the setting as a claim (v1.8.5, Nov 6: "MMMU Pro
            # (Standard 10)" must not mint a claim of 10).
            _rescan_inline_score(text, m, key, label, variant, _add)
            continue
        if key in _BENCHMARK_KEYS and _is_following_benchmark_score(text, m.end()):
            # The number is the score of a following benchmark — the
            # reverse pass mints it under its own name (v1.8.5, Nov 6:
            # "82.2% on SWE-bench Verified" is not GPQA's score).
            continue
        # Same bare-number count rule as the colon pass (Sieve gap
        # 2026-11-03: inline counts had no unit and "compared" with %).
        bare = _unit_of(raw_val) == ""
        count_after = bare and _count_after(text, m.end())
        _add(key, label, raw_val, m.group(0),
             unit_override="count" if count_after else "",
             version=_benchmark_version(key, m.group(0)),
             shot_count=_shot_count_for_match(text, m, 2),
             variant=variant)

    # Pass 3 — benchmark-reverse: "61.3% on GPQA" (score precedes the name)
    for m in _BENCHMARK_REVERSE_RE.finditer(text):
        raw_val = m.group(1).strip()
        if _is_version_number(text, m.start(1)):
            continue
        entity = _normalize_entity_key(m.group(2).strip())
        if entity is None:
            continue
        key, label, ematch = entity
        _add(key, label, raw_val, m.group(0),
             version=_benchmark_version(key, m.group(0)),
             shot_count=_shot_count_for_match(text, m, 1),
             variant=_benchmark_variant(key, ematch.group(0), "", text[m.end():m.end() + 20]))

    # Pass 4 — pricing rows: a $ amount is a pricing claim only when
    # pricing words sit within ~60 chars, so market prices never match.
    # The tier is decided by ownership, not mere nearest-word distance:
    # in a pricing row each $ amount owns the tier word that FOLLOWS it
    # ("$0.30 per million input tokens") or IMMEDIATELY PRECEDES it
    # ("Input: $0.30") — never a tier word that belongs to a
    # neighbouring price. The old nearest-word rule keyed "$1.00 per
    # million output tokens" as input, because the previous price's
    # "input" was 14 chars before it and its own "output" 18 chars
    # after it (v1.8.5, Nov 6). So: a tier word immediately before the
    # amount (nothing but space/colon between) wins; otherwise the
    # nearest tier word after the amount within the same sentence (up
    # to the next $). The minting check above stays whole-window: a
    # pricing word anywhere nearby means it is a price; only the tier
    # is direction-scoped.
    for m in _DOLLAR_RE.finditer(text):
        window_start = max(0, m.start() - 60)
        window = text[window_start:m.end() + 60]
        if not _PRICING_WORDS_RE.search(window):
            continue
        dollar_pos = m.start() - window_start
        # The amount itself ends after its digits (and any decimal
        # point): sentence/forward scans start there, not at the $
        # sign — the "." in "$0.30" is a decimal point, not a
        # sentence break.
        amount_end = dollar_pos + len(m.group(0))
        seg_end = len(window)
        next_dollar = window.find("$", amount_end)
        if next_dollar != -1:
            seg_end = next_dollar
        for ch in (".", "!", "?", "\n"):
            p = window.find(ch, amount_end, seg_end)
            if p != -1:
                seg_end = min(seg_end, p)
        # Tier ownership (v1.8.5): a tier word belongs to an amount
        # only when it is that amount's own label — immediately before
        # it ("input $15", "Input: $0.30", nothing but space/colon
        # between) or following it in the same sentence ("$0.30 per
        # million input tokens", "$0.03 cached", "$1.18 per task").
        # Backward-adjacent wins: in "input $15 per million tokens,
        # output $75", the "output" after $15 is the NEXT price's
        # tier, not $15's. And a tier word across a neighbouring $
        # never counts — "$1.00 per million output tokens" must not
        # key as input because the previous price's "input" sits
        # 14 chars before it.
        nearest = ""
        back = _BACKWARD_TIER_RE.search(window[max(0, dollar_pos - 16):dollar_pos])
        if back:
            nearest = back.group(1).lower()
        else:
            nearest_dist = None
            for word_re, word_key in ((_INPUT_WORD_RE, "input"), (_OUTPUT_WORD_RE, "output"),
                                      (_CACHED_WORD_RE, "cached"), (_PER_TASK_RE, "per-task")):
                for wm in word_re.finditer(window, amount_end, seg_end):
                    dist = wm.start() - dollar_pos
                    if dist <= 30 and (nearest_dist is None or dist < nearest_dist):
                        nearest, nearest_dist = word_key, dist
        if nearest == "input":
            key, label = "pricing-input", "Pricing (input)"
        elif nearest == "output":
            key, label = "pricing-output", "Pricing (output)"
        elif nearest == "cached":
            key, label = "pricing-cached", "Pricing (cached)"
        elif nearest == "per-task":
            key, label = "pricing-per-task", "Pricing (per task)"
        else:
            key, label = "pricing", "Pricing"
        _add(key, label, m.group(1), window.strip()[:120], unit_override="$")

    if dropped_out is not None:
        dropped_out.extend(dropped)

    return claims


def compare_documents(analyzed_docs: List[Dict]) -> Dict:
    """Cross-reference numeric claims across multiple documents.

    Each item in *analyzed_docs* must have at minimum::

        { "text": str, "source_name": str }   # source_name optional

    Returns::

        {
          "doc_count":         int,
          "shared_claims":     [...],   # same entity in 2+ docs
          "unique_claims":     [...],   # entity found in only one doc
          "conflicts":         [...],   # same entity, values outside tolerance
          "alignment":         [...],   # same entity, values within tolerance
          "unit_mismatches":   [...],   # same entity, different units
          "version_mismatches": [...],  # same benchmark, different versions
          "evaluation_setting_mismatches": [...],  # same benchmark, different shot counts
          "variant_mismatches": [...],  # same benchmark, different variants (v1.8.5)
          "timeline_reconciliation": [...],  # shared-year date/precision notes
          "agreement_status":  str,     # aligned | conflicts_present | not_established
          "comparable_claims": int,     # claims that actually went through comparison
          "presence_absence_mismatches": [...],  # PRESENT vs EXPLICITLY_ABSENT
          "presence_status":   {entity_key: {doc_name: status}},
          "crude_note":        str,     # WTI/Brent note if both found
          "follow_up_sources": [...],   # gated on compared entities (v1.8.1, gap 5)
          "follow_up_trace":   [...],   # which entity + claim supports each suggestion
          "dropped_market_claims": [...],  # market entities dropped for lack of context
          "triage_summary":    str,
        }

    Presence statuses are PRESENT / EXPLICITLY_ABSENT / NOT_OBSERVED /
    UNKNOWN (empty doc text). Numeric alignment and conflicts only run
    where both sides are PRESENT; a merely-missing entity is NOT_OBSERVED
    and never becomes a conflict (Arch design note 2026-11-01).
    """
    if len(analyzed_docs) < 2:
        return {
            "doc_count":         len(analyzed_docs),
            "error":             "compare requires at least 2 documents",
            "shared_claims":     [],
            "unique_claims":     [],
            "conflicts":         [],
            "alignment":         [],
            "unit_mismatches":   [],
            "version_mismatches": [],
            "evaluation_setting_mismatches": [],
            "variant_mismatches": [],
            "shared_dates":      [],
            "shared_years":      [],
            "unique_dates":      [],
            "timeline_reconciliation": [],
            "agreement_status":  "not_established",
            "comparable_claims": 0,
            "presence_absence_mismatches": [],
            "presence_status":   {},
            "crude_note":        "",
            "follow_up_sources": [],
            "follow_up_trace":   [],
            "dropped_market_claims": [],
            "triage_summary":    "Insufficient documents for comparison.",
        }

    def _doc_name(idx: int) -> str:
        return analyzed_docs[idx].get("source_name", f"doc_{idx + 1}")

    # Extract claims per doc. Dropped market claims are collected per
    # doc so the caller can audit the context gate (v1.8.1, gap 3)
    # instead of silently losing the evidence.
    doc_claims: List[List[Dict]] = []
    dropped_market_claims: List[Dict] = []
    for doc_idx, doc in enumerate(analyzed_docs):
        dropped: List[Dict] = []
        doc_claims.append(extract_numeric_claims(doc.get("text", ""), dropped_out=dropped))
        for entry in dropped:
            dropped_market_claims.append({**entry, "source": _doc_name(doc_idx)})
    doc_dates: List[List[Dict]] = [
        extract_date_claims(doc.get("text", ""))
        for doc in analyzed_docs
    ]

    # Group all claims by canonical entity key
    entity_map: Dict[str, List[Dict]] = {}   # key → [{doc_idx, claim}]
    for doc_idx, claims in enumerate(doc_claims):
        for claim in claims:
            key = claim["entity_key"]
            entity_map.setdefault(key, []).append({"doc_idx": doc_idx, "claim": claim})

    shared_claims: List[Dict] = []
    unique_claims: List[Dict] = []
    conflicts:     List[Dict] = []
    alignment:     List[Dict] = []
    unit_mismatches: List[Dict] = []
    version_mismatches: List[Dict] = []
    evaluation_setting_mismatches: List[Dict] = []
    presence_absence_mismatches: List[Dict] = []
    variant_mismatches: List[Dict] = []
    # Provisional variant comparisons (v1.8.5): one stated variant
    # compared against unstated sides — counted for the triage summary.
    provisional_variant_comparisons: List[str] = []

    # WTI vs Brent: note their presence but don't call it a conflict
    has_wti   = "wti-crude"   in entity_map
    has_brent = "brent-crude" in entity_map
    crude_note = (
        "WTI and Brent crude benchmarks both present — these are different contracts "
        "(WTI = US delivery; Brent = North Sea/international) and cannot be directly compared."
    ) if (has_wti and has_brent) else ""

    # ── Presence/absence channel (Arch design note 2026-11-01) ──────────
    # Presence and absence are different kinds of evidence, not two values
    # on one axis. Per entity key and document:
    #   PRESENT          — a numeric claim for the entity was extracted
    #   EXPLICITLY_ABSENT — the doc explicitly says the entity is absent
    #                      ("no <entity> row", "<entity> not listed", ...)
    #   NOT_OBSERVED     — the doc merely lacks the entity (never evidence
    #                      of absence, never a conflict)
    #   UNKNOWN          — the doc text was empty / extraction impossible
    # Scope basis, recorded on each mismatch entry: docs in one compare
    # call are the caller-compared set about one topic. Version/page-level
    # scope matching is NOT independently verified by this channel — a
    # caller that needs it must check scope before forwarding a mismatch.
    presence_status: Dict[str, Dict[str, str]] = {}
    for key, entries in entity_map.items():
        label = entries[0]["claim"]["entity_label"]
        present_docs = {e["doc_idx"] for e in entries}
        status: Dict[str, str] = {}
        for doc_idx, doc in enumerate(analyzed_docs):
            name = _doc_name(doc_idx)
            doc_text = doc.get("text", "")
            if doc_idx in present_docs:
                status[name] = "PRESENT"
            elif not doc_text.strip():
                status[name] = "UNKNOWN"
            elif _has_explicit_absence(doc_text, label):
                status[name] = "EXPLICITLY_ABSENT"
            else:
                status[name] = "NOT_OBSERVED"
        presence_status[key] = status

    def _mismatch_entry(key: str, label: str, present_in: List[str]) -> Dict:
        status = presence_status[key]
        absent_in = [n for n, s in status.items() if s == "EXPLICITLY_ABSENT"]
        return {
            "entity_key":   key,
            "entity_label": label,
            "present_in":   present_in,
            "explicitly_absent_in": absent_in,
            "presence_status": status,
            "note": (
                "Presence/absence mismatch: present in one source, explicitly "
                "absent in another. Numeric alignment runs only where both "
                "sides are PRESENT, so the absent side is not numerically "
                "compared. Scope basis: caller-compared set about one topic; "
                "version/page-level scope not independently verified by "
                "this channel."
            ),
        }

    # Variant partitions (v1.8.5, Nov 6): benchmark claims compare
    # within their variant, never across it. Two or more STATED variants
    # for one key → the variants are not comparable at all: record a
    # variant mismatch and compare within each stated partition
    # separately (each partition becomes its own work item below).
    # Zero or one stated variant → compare the whole group as before,
    # with a provisional note when some sides state a variant and
    # others don't. The work list carries the group's variant so the
    # loop body can tag every entry it emits.
    grouped: List[Tuple[str, List[Dict], str]] = []
    for key, entries in entity_map.items():
        partitions: Dict[str, List[Dict]] = {}
        for e in entries:
            partitions.setdefault(e["claim"].get("variant", ""), []).append(e)
        stated_variants = {v for v in partitions if v}
        if len(stated_variants) > 1:
            variant_mismatches.append({
                "entity_key":   key,
                "entity_label": entries[0]["claim"]["entity_label"],
                "variants":     sorted(stated_variants),
                "values":       [
                    {"source": _doc_name(e["doc_idx"]), "value": e["claim"]["numeric_value"],
                     "unit": e["claim"]["unit"], "raw": e["claim"]["raw_value"],
                     "version": e["claim"].get("version", ""),
                     "shot_count": e["claim"].get("shot_count"),
                     "variant": e["claim"].get("variant", "")}
                    for e in entries
                ],
                "note": (
                    "Same benchmark reported in different variants across sources "
                    "(e.g. Pro vs Verified) — scores from different variants are not "
                    "directly comparable. Each variant was compared within itself; "
                    "do not read as a conflict or an alignment."
                ),
            })
            for variant, part in partitions.items():
                grouped.append((key, part, variant))
        else:
            grouped.append((key, entries, next(iter(stated_variants), "")))

    # A presence/absence mismatch is recorded once per key, even when
    # the key was split into several variant partitions.
    _presence_mismatch_done: set = set()

    for key, entries, variant in grouped:
        label      = entries[0]["claim"]["entity_label"]
        doc_names  = [_doc_name(e["doc_idx"]) for e in entries]
        values     = [e["claim"]["numeric_value"] for e in entries]
        units      = [e["claim"]["unit"]          for e in entries]
        raws       = [e["claim"]["raw_value"]      for e in entries]

        val_objs = [
            {"source": n, "value": v, "unit": u, "raw": r, "version": ver, "shot_count": sh, "variant": va}
            for n, v, u, r, ver, sh, va in zip(
                doc_names, values, units, raws,
                [e["claim"].get("version", "") for e in entries],
                [e["claim"].get("shot_count") for e in entries],
                [e["claim"].get("variant", "") for e in entries])
        ]

        if len(entries) == 1:
            # A lone claim where another doc EXPLICITLY lacks the entity
            # is a presence/absence mismatch, not a merely unique claim.
            if any(s == "EXPLICITLY_ABSENT" for s in presence_status[key].values()):
                if key not in _presence_mismatch_done:
                    presence_absence_mismatches.append(
                        _mismatch_entry(key, label, doc_names)
                    )
                    _presence_mismatch_done.add(key)
                continue
            unique_claims.append({
                "entity_key":   key,
                "entity_label": label,
                "found_in":     doc_names[0],
                "value":        values[0],
                "unit":         units[0],
                "raw":          raws[0],
                "variant":      variant,
            })
            continue

        shared_claims.append({
            "entity_key":    key,
            "entity_label":  label,
            "found_in_docs": list(dict.fromkeys(doc_names)),
            "values":        val_objs,
            "variant":       variant,
        })

        # Shared key, but some compared doc explicitly lacks the entity:
        # record the mismatch alongside (not instead of) numeric comparison
        # of the PRESENT sides.
        if any(s == "EXPLICITLY_ABSENT" for s in presence_status[key].values()):
            if key not in _presence_mismatch_done:
                presence_absence_mismatches.append(
                    _mismatch_entry(key, label, doc_names)
                )
                _presence_mismatch_done.add(key)

        # Skip crude benchmark conflict check — already noted as different contracts
        if key in _CRUDE_KEYS and crude_note:
            continue

        # Unit guard (Sieve gap 2026-10-31: a count vs a percentage on the
        # same entity was read as one comparable series). Different
        # non-empty units are not comparable at all — neither a conflict
        # nor an alignment. An empty unit is "unspecified", not a unit,
        # so "$2,650" vs "2650" still compares; but a bare number
        # followed by a count word ("94 problems") carries unit "count"
        # (see the colon pass), so counts never merge with percentages.
        specified_units = {u for u in units if u}
        if len(specified_units) > 1:
            unit_mismatches.append({
                "entity_key":   key,
                "entity_label": label,
                "units":        sorted(specified_units),
                "values":       val_objs,
                "variant":      variant,
                "note": (
                    "Same entity reported in different units across sources "
                    "(e.g. a count vs a percentage) — values are not directly "
                    "comparable. Do not read as a conflict or an alignment."
                ),
            })
            continue

        # Version gate (Arch review 2026-11-04): the same benchmark in
        # different declared versions is not comparable — neither a
        # conflict nor an alignment. Only positive version disagreement
        # fires: unversioned claims still compare (see _BENCHMARK_KEYS).
        named_versions = {e["claim"].get("version", "") for e in entries} - {""}
        if key in _BENCHMARK_KEYS and len(named_versions) > 1:
            version_mismatches.append({
                "entity_key":   key,
                "entity_label": label,
                "versions":     sorted(named_versions),
                "values":       val_objs,
                "variant":      variant,
                "note": (
                    "Same benchmark reported in different versions across "
                    "sources — scores from different versions are not directly "
                    "comparable. Do not read as a conflict or an alignment."
                ),
            })
            continue

        # Shot-count gate (Arch review 2026-11-05, v1.8.1): a shot count
        # is a measurement-condition attribute, not a performance result.
        # Different STATED shot counts block direct score comparison —
        # the scores were measured under different evaluation settings —
        # and land in evaluation_setting_mismatches with a note naming
        # the difference. When no side states a shot count (or only one
        # side does), the comparison proceeds PROVISIONALLY, but the
        # missing conditions stay visible on the entry: a score must
        # never be called aligned or conflicting while its measurement
        # conditions are unstated and invisible.
        named_shots: set = set()
        evaluation_note = ""
        if key in _BENCHMARK_KEYS:
            shots = [e["claim"].get("shot_count") for e in entries]
            named_shots = {s for s in shots if s is not None}
            if len(named_shots) > 1:
                evaluation_setting_mismatches.append({
                    "entity_key":   key,
                    "entity_label": label,
                    "shot_counts":  sorted(named_shots),
                    "values":       val_objs,
                    "variant":      variant,
                    "note": (
                        "Same benchmark evaluated under different shot counts "
                        f"({', '.join(str(s) + '-shot' for s in sorted(named_shots))}) — "
                        "scores measured under different evaluation settings are not "
                        "directly comparable. Do not read as a conflict or an alignment."
                    ),
                })
                continue
            if named_shots:
                stated = next(iter(named_shots))
                if any(s is None for s in shots):
                    evaluation_note = (
                        f"Shot count stated in some sources ({stated}-shot) and unstated "
                        "elsewhere — comparison is provisional; missing evaluation "
                        "conditions must remain visible."
                    )
                else:
                    evaluation_note = f"All sources state {stated}-shot evaluation."
            else:
                evaluation_note = (
                    "Shot count not stated in any source — comparison is provisional; "
                    "missing evaluation conditions must remain visible."
                )

        vals_sorted = sorted(values)
        spread      = vals_sorted[-1] - vals_sorted[0]

        # Variant conditions (v1.8.5): when one stated variant is
        # compared against unstated sides, the comparison is
        # provisional — the unstated side may be a different variant.
        # (Within a stated partition every side states the same variant,
        # so this only fires for the whole-group provisional case.)
        variant_conditions = ""
        if key in _BENCHMARK_KEYS:
            named_variants = {e["claim"].get("variant", "") for e in entries} - {""}
            if len(named_variants) == 1 and any(e["claim"].get("variant", "") == "" for e in entries):
                variant_conditions = (
                    f"Variant stated in some sources ({next(iter(named_variants))}) and unstated "
                    "elsewhere — comparison is provisional; the unstated side may be a different variant."
                )

        # Tolerance: for rate/yield/% values (<20) use 0.15 pp;
        # for price levels (>=20) use 2% relative.
        if all(abs(v) < 20 for v in values):
            is_conflict = spread > 0.15
        else:
            reference   = abs(vals_sorted[0]) if vals_sorted[0] != 0 else 1.0
            is_conflict = spread > reference * 0.02

        entry = {
            "entity_key":   key,
            "entity_label": label,
            "spread":       round(spread, 5),
            "values":       val_objs,
            "variant":      variant,
        }
        if key in _BENCHMARK_KEYS:
            entry["shot_count"] = next(iter(named_shots), None) if named_shots else None
            entry["evaluation_conditions"] = evaluation_note
        if variant_conditions:
            entry["variant_conditions"] = variant_conditions
            provisional_variant_comparisons.append(key)

        if is_conflict:
            entry["note"] = (
                f"Discrepancy of {spread:.3f} between sources. "
                "Verify publication timestamps — values may reflect different trading sessions."
            )
            conflicts.append(entry)
        else:
            alignment.append(entry)

    # ── Date claims (Sieve gap 2026-11-02) ───────────────────────────────
    # Dates are claims too, and rollout stories live on them. Exact dates
    # shared across docs align; years shared across docs align at year
    # granularity; differing dates are NOT conflicts — different events
    # legitimately have different dates, so a date in one doc only is
    # unique, never a discrepancy.
    date_map: Dict[str, List[Dict]] = {}
    for doc_idx, claims in enumerate(doc_dates):
        for claim in claims:
            date_map.setdefault(claim["date_key"], []).append(
                {"doc_idx": doc_idx, "claim": claim})
    shared_dates: List[Dict] = []
    shared_years: List[Dict] = []
    unique_dates: List[Dict] = []
    for date_key, entries in date_map.items():
        doc_names = [_doc_name(e["doc_idx"]) for e in entries]
        raws = [e["claim"]["raw"] for e in entries]
        kind = entries[0]["claim"]["kind"]
        entry = {
            "date_key":  date_key,
            "kind":      kind,
            "found_in":  doc_names,
            "raw":       raws,
        }
        if len(entries) == 1:
            unique_dates.append(entry)
        elif kind == "date":
            shared_dates.append(entry)
        else:
            shared_years.append(entry)

    # Follow-up source suggestions, gated (Arch review 2026-11-05,
    # v1.8.1, gap 5): a recommended follow-up source must be traceable
    # to a supported entity AND a relevant claim. Eligibility is
    # derived from the entities that actually appear in the compared
    # output — shared/unique claims, alignment, conflicts, mismatches —
    # never from raw extraction alone, so a dropped false entity (the
    # Nov 5 Dow) cannot smuggle a market-data recommendation into the
    # result. Each suggestion carries its trace: the entity keys that
    # support it. A small additional guard, not a new recommendation
    # engine.
    compared_entity_keys = (
        {c["entity_key"] for c in shared_claims}
        | {c["entity_key"] for c in unique_claims}
        | {c["entity_key"] for c in alignment}
        | {c["entity_key"] for c in conflicts}
        | {m["entity_key"] for m in unit_mismatches}
        | {m["entity_key"] for m in version_mismatches}
        | {m["entity_key"] for m in evaluation_setting_mismatches}
        | {m["entity_key"] for m in variant_mismatches}
    )
    follow_up: List[str] = []
    follow_up_trace: List[Dict] = []

    def _suggest(keys, text: str) -> None:
        hit = sorted(compared_entity_keys & set(keys))
        if hit:
            follow_up.append(text)
            follow_up_trace.append({"source": text, "entity_keys": hit})

    _suggest({"10yr-treasury-yield", "2yr-treasury-yield",
              "5yr-treasury-yield", "30yr-treasury-yield"},
             "US Treasury (treasurydirect.gov) — authoritative daily yield curve data")
    _suggest({"wti-crude", "brent-crude"},
             "EIA (eia.gov) — US and international crude pricing / inventory")
    _suggest({"cpi", "ppi", "jobless-claims", "retail-sales"},
             "BLS (bls.gov) — CPI, PPI, jobless claims primary releases")
    _suggest({"sp500", "nasdaq", "dow", "russell2000", "vix"},
             "Exchange closing prices (NYSE, Nasdaq) — authoritative end-of-day data")

    # Triage summary
    n_conflicts = len(conflicts)
    n_shared    = len(shared_claims)
    n_mismatches = len(presence_absence_mismatches)
    n_unit_mismatches = len(unit_mismatches)
    n_version_mismatches = len(version_mismatches)
    n_setting_mismatches = len(evaluation_setting_mismatches)
    n_variant_mismatches = len(variant_mismatches)
    n_shared_dates = len(shared_dates) + len(shared_years)
    timeline_notes = reconcile_timelines(analyzed_docs)
    # Agreement status (Arch review 2026-11-04): zero conflicts must not
    # imply agreement when nothing was actually compared. Only claims
    # that went through alignment/conflict comparison count as
    # comparable; unit/version mismatches are explicitly NOT comparable.
    comparable_claims = len(alignment) + len(conflicts)
    if conflicts:
        agreement_status = "conflicts_present"
    elif alignment:
        agreement_status = "aligned"
    else:
        agreement_status = "not_established"
    if n_conflicts > 0:
        triage_summary = (
            f"{n_conflicts} numeric conflict(s) detected across "
            f"{len(analyzed_docs)} documents with {n_shared} shared claim(s). "
            "Verify publication timestamps and benchmark definitions before forwarding."
        )
    elif n_shared > 0:
        triage_summary = (
            f"No conflicts found in {n_shared} shared numeric claim(s) across "
            f"{len(analyzed_docs)} documents. Values align within tolerance."
        )
    else:
        triage_summary = (
            f"No overlapping numeric claims found across "
            f"{len(analyzed_docs)} documents."
        )
    if n_mismatches > 0:
        triage_summary += (
            f" {n_mismatches} presence/absence mismatch(es): an entity present "
            "in one source is explicitly absent in another — check scope "
            "before forwarding."
        )
    if n_unit_mismatches > 0:
        triage_summary += (
            f" {n_unit_mismatches} unit mismatch(es): same entity in different "
            "units across sources — not comparable, not a conflict."
        )
    if n_version_mismatches > 0:
        triage_summary += (
            f" {n_version_mismatches} version mismatch(es): same benchmark in "
            "different versions across sources — not comparable, not a conflict."
        )
    if n_setting_mismatches > 0:
        triage_summary += (
            f" {n_setting_mismatches} evaluation-setting mismatch(es): same benchmark "
            "evaluated under different shot counts — not comparable, not a conflict."
        )
    if n_variant_mismatches > 0:
        triage_summary += (
            f" {n_variant_mismatches} variant mismatch(es): same benchmark in "
            "different variants across sources — not comparable, not a conflict."
        )
    n_provisional = sum(
        1 for c in alignment + conflicts
        if "provisional" in c.get("evaluation_conditions", "")
    )
    if n_provisional > 0:
        triage_summary += (
            f" {n_provisional} provisional comparison(s): shot counts unstated on "
            "one or both sides — see evaluation_conditions on the entries."
        )
    if provisional_variant_comparisons:
        triage_summary += (
            f" {len(provisional_variant_comparisons)} provisional variant comparison(s): "
            "variant stated on one side and unstated on the other — see "
            "variant_conditions on the entries."
        )
    if dropped_market_claims:
        triage_summary += (
            f" {len(dropped_market_claims)} market-entity claim(s) dropped: the entity "
            "fired without positive market context in the document — see "
            "dropped_market_claims."
        )
    if n_shared_dates > 0:
        triage_summary += (
            f" {n_shared_dates} shared date claim(s) across documents."
        )
    if timeline_notes:
        triage_summary += (
            f" {len(timeline_notes)} timeline reconciliation note(s): shared-year "
            "dates differ in precision — see timeline_reconciliation, not conflicts."
        )
    if comparable_claims == 0:
        triage_summary += (
            " No comparable claims were extracted, so no agreement is "
            "established — zero conflicts here means no detected "
            "disagreement, not agreement."
        )

    return {
        "doc_count":         len(analyzed_docs),
        "shared_claims":     shared_claims,
        "unique_claims":     unique_claims,
        "conflicts":         conflicts,
        "alignment":         alignment,
        "unit_mismatches":   unit_mismatches,
        "version_mismatches": version_mismatches,
        "evaluation_setting_mismatches": evaluation_setting_mismatches,
        "variant_mismatches": variant_mismatches,
        "shared_dates":      shared_dates,
        "shared_years":      shared_years,
        "unique_dates":      unique_dates,
        "timeline_reconciliation": timeline_notes,
        "agreement_status":  agreement_status,
        "comparable_claims": comparable_claims,
        "presence_absence_mismatches": presence_absence_mismatches,
        "presence_status":   presence_status,
        "crude_note":        crude_note,
        "follow_up_sources": follow_up,
        "follow_up_trace":   follow_up_trace,
        "dropped_market_claims": dropped_market_claims,
        "triage_summary":    triage_summary,
    }
