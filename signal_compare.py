"""
signal_compare.py — cross-document numeric claim extraction and comparison.

Philosophy: extract the minimum invariant representation needed for cross-
reference. Each claim is normalized to a canonical entity key so claims across
documents can be matched even when they use different surface forms
(e.g. "10-Year Treasury Yield: 4.61%" and "10yr Treasury: 4.61%").

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
_COLON_RE = re.compile(
    r"([A-Za-z&][^:\n]{2,55}):"          # label (2–55 chars)
    r"\s*"
    r"([+-]?\$?\d[\d,\.]*"               # leading digit (with optional sign/$)
    r"(?:%|bps|(?:\s*/bbl)|(?:\s*/oz)|(?:\s+per\s+(?:barrel|oz(?:ounce)?)))?"
    r"(?:\s*[-–]\s*\d[\d,\.]*(?:%|bps)?)?)",  # optional range "4.25-4.75"
    re.I,
)

# Lower-confidence inline patterns: "S&P 500 +0.02%" or "S&P 500 reached 5,894"
_INLINE_ENTITY_RE = re.compile(
    r"\b(S&P\s*500|Nasdaq(?:\s+Composite)?|Dow(?:\s+Jones)?|DJIA|Russell\s+2000|"
    r"\d{1,2}[-\s]?y(?:ear|r)(?:\s+Treasury)?|"
    r"WTI(?:\s+crude)?|Brent(?:\s+crude)?|Gold|VIX|DXY|"
    r"CPI|PPI|PCE|PMI|GDP|"
    r"GPQA(?:\s+Diamond)?|SWE[-\s]?bench(?:\s+Verified)?|HumanEval(?:\+)?|"
    r"MMLU[-\s]?Pro|MMLU|MMMU|AIME|GSM8K|MATH(?:[-\s]?500)?|ARC[-\s]?AGI|Humanity'?s\s+Last\s+Exam|"
    r"(?:initial\s+)?jobless\s+claims?|retail\s+sales?)"
    r"[^.!?\n]{0,70}?"
    r"([+-]?\$?\d[\d,\.]*"
    r"(?:%|\s*bps|\s*/bbl|\s*/oz|\s+per\s+(?:barrel|oz)|\s*,\d{3})*)",
    re.I,
)

# Benchmark-reverse pass: "61.3% on GPQA" — the score precedes the
# benchmark name, so the forward inline pattern cannot see it. Restricted
# to benchmark names + a % value so prose percentages never match.
_BENCHMARK_NAMES_RE = (
    r"GPQA(?:\s+Diamond)?|SWE[-\s]?bench(?:\s+Verified)?|HumanEval(?:\+)?|"
    r"MMLU[-\s]?Pro|MMLU|MMMU|AIME|GSM8K|MATH(?:[-\s]?500)?|ARC[-\s]?AGI|Humanity'?s\s+Last\s+Exam"
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
    r"\b(pric(?:e|ing)|per\s+(?:million|1M)\s+tokens?|tokens?)\b", re.I)
_INPUT_WORD_RE = re.compile(r"\binput\b", re.I)
_OUTPUT_WORD_RE = re.compile(r"\boutput\b", re.I)
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


def _normalize_entity_key(fragment: str) -> Optional[Tuple[str, str]]:
    """Return (canonical_key, display_label) for the first matching entity pattern."""
    for pattern, key, label in _ENTITY_KEY_MAP:
        if pattern.search(fragment):
            return key, label
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


# ── Public API ────────────────────────────────────────────────────────────────

def extract_numeric_claims(text: str) -> List[Dict]:
    """Extract normalized numeric claims from text.

    Returns a list of::

        {
          "entity_key":    "10yr-treasury-yield",
          "entity_label":  "10-Year Treasury Yield",
          "raw_value":     "4.61%",
          "numeric_value": 4.61,
          "unit":          "%",
          "context":       "10-Year Treasury Yield: 4.61%",
        }

    Deduplicates on (entity_key, numeric_value) pairs.
    """
    claims: List[Dict] = []
    seen: set = set()  # (entity_key, numeric_value) dedup

    def _add(key: str, label: str, raw_val: str, context: str, unit_override: str = "") -> None:
        numeric = _parse_value(raw_val)
        if numeric is None:
            return
        unit    = unit_override or _unit_of(raw_val)
        dedup   = (key, numeric)
        if dedup in seen:
            return
        seen.add(dedup)
        claims.append({
            "entity_key":    key,
            "entity_label":  label,
            "raw_value":     raw_val.strip(),
            "numeric_value": numeric,
            "unit":          unit,
            "context":       context[:120],
        })

    # Pass 1 — colon-delimited data lines (higher precision)
    for m in _COLON_RE.finditer(text):
        label_fragment = m.group(1).strip()
        raw_val        = m.group(2).strip()
        entity = _normalize_entity_key(label_fragment)
        if entity is None:
            continue
        key, label = entity
        # A bare number followed by a count word ("GPQA: 94 problems") is
        # a count, not an unspecified unit — without this, a count and a
        # percentage on the same entity read as one comparable series.
        count_after = bool(re.search(
            r"\b(problems|questions|items|examples|tasks|samples)\b",
            text[m.end():m.end() + 25], re.I))
        _add(key, label, raw_val, m.group(0),
             unit_override="count" if count_after else "")

    # Pass 2 — inline patterns (catches prose and snapshot lines)
    for m in _INLINE_ENTITY_RE.finditer(text):
        entity_frag = m.group(1).strip()
        raw_val     = m.group(2).strip()
        if not raw_val or raw_val == ".":
            continue
        entity = _normalize_entity_key(entity_frag)
        if entity is None:
            continue
        key, label = entity
        _add(key, label, raw_val, m.group(0))

    # Pass 3 — benchmark-reverse: "61.3% on GPQA" (score precedes the name)
    for m in _BENCHMARK_REVERSE_RE.finditer(text):
        raw_val = m.group(1).strip()
        entity = _normalize_entity_key(m.group(2).strip())
        if entity is None:
            continue
        key, label = entity
        _add(key, label, raw_val, m.group(0))

    # Pass 4 — pricing rows: a $ amount is a pricing claim only when
    # pricing words sit within ~60 chars, so market prices never match.
    # Input vs output is decided by the NEAREST input/output word, not
    # mere presence — a line naming both must not key every amount input.
    for m in _DOLLAR_RE.finditer(text):
        window_start = max(0, m.start() - 60)
        window = text[window_start:m.end() + 60]
        if not _PRICING_WORDS_RE.search(window):
            continue
        dollar_pos = m.start() - window_start
        nearest, nearest_dist = "", None
        for word_re, word_key in ((_INPUT_WORD_RE, "input"), (_OUTPUT_WORD_RE, "output")):
            for wm in word_re.finditer(window):
                dist = abs(wm.start() - dollar_pos)
                if nearest_dist is None or dist < nearest_dist:
                    nearest, nearest_dist = word_key, dist
        if nearest == "input":
            key, label = "pricing-input", "Pricing (input)"
        elif nearest == "output":
            key, label = "pricing-output", "Pricing (output)"
        else:
            key, label = "pricing", "Pricing"
        _add(key, label, m.group(1), window.strip()[:120], unit_override="$")

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
          "presence_absence_mismatches": [...],  # PRESENT vs EXPLICITLY_ABSENT
          "presence_status":   {entity_key: {doc_name: status}},
          "crude_note":        str,     # WTI/Brent note if both found
          "follow_up_sources": [...],
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
            "shared_dates":      [],
            "shared_years":      [],
            "unique_dates":      [],
            "presence_absence_mismatches": [],
            "presence_status":   {},
            "crude_note":        "",
            "follow_up_sources": [],
            "triage_summary":    "Insufficient documents for comparison.",
        }

    # Extract claims per doc
    doc_claims: List[List[Dict]] = [
        extract_numeric_claims(doc.get("text", ""))
        for doc in analyzed_docs
    ]
    doc_dates: List[List[Dict]] = [
        extract_date_claims(doc.get("text", ""))
        for doc in analyzed_docs
    ]

    def _doc_name(idx: int) -> str:
        return analyzed_docs[idx].get("source_name", f"doc_{idx + 1}")

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
    presence_absence_mismatches: List[Dict] = []

    # WTI vs Brent: note their presence but don't call it a conflict
    has_wti   = "wti-crude"   in entity_map
    has_brent = "brent-crude" in entity_map
    crude_note = (
        "WTI and Brent crude benchmarks both present — these are different contracts "
        "(WTI = US delivery; Brent = North Sea/international) and cannot be directly compared."
    ) if (has_wti and has_brent) else ""

    all_claims_flat = [c for cl in doc_claims for c in cl]

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

    for key, entries in entity_map.items():
        label      = entries[0]["claim"]["entity_label"]
        doc_names  = [_doc_name(e["doc_idx"]) for e in entries]
        values     = [e["claim"]["numeric_value"] for e in entries]
        units      = [e["claim"]["unit"]          for e in entries]
        raws       = [e["claim"]["raw_value"]      for e in entries]

        val_objs = [
            {"source": n, "value": v, "unit": u, "raw": r}
            for n, v, u, r in zip(doc_names, values, units, raws)
        ]

        if len(entries) == 1:
            # A lone claim where another doc EXPLICITLY lacks the entity
            # is a presence/absence mismatch, not a merely unique claim.
            if any(s == "EXPLICITLY_ABSENT" for s in presence_status[key].values()):
                presence_absence_mismatches.append(
                    _mismatch_entry(key, label, doc_names)
                )
                continue
            unique_claims.append({
                "entity_key":   key,
                "entity_label": label,
                "found_in":     doc_names[0],
                "value":        values[0],
                "unit":         units[0],
                "raw":          raws[0],
            })
            continue

        shared_claims.append({
            "entity_key":    key,
            "entity_label":  label,
            "found_in_docs": doc_names,
            "values":        val_objs,
        })

        # Shared key, but some compared doc explicitly lacks the entity:
        # record the mismatch alongside (not instead of) numeric comparison
        # of the PRESENT sides.
        if any(s == "EXPLICITLY_ABSENT" for s in presence_status[key].values()):
            presence_absence_mismatches.append(
                _mismatch_entry(key, label, doc_names)
            )

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
                "note": (
                    "Same entity reported in different units across sources "
                    "(e.g. a count vs a percentage) — values are not directly "
                    "comparable. Do not read as a conflict or an alignment."
                ),
            })
            continue

        vals_sorted = sorted(values)
        spread      = vals_sorted[-1] - vals_sorted[0]

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
        }

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

    # Follow-up source suggestions based on what was found
    follow_up: List[str] = []
    entity_keys_found = {c["entity_key"] for c in all_claims_flat}
    if entity_keys_found & {"10yr-treasury-yield", "2yr-treasury-yield",
                            "5yr-treasury-yield", "30yr-treasury-yield"}:
        follow_up.append("US Treasury (treasurydirect.gov) — authoritative daily yield curve data")
    if entity_keys_found & {"wti-crude", "brent-crude"}:
        follow_up.append("EIA (eia.gov) — US and international crude pricing / inventory")
    if entity_keys_found & {"cpi", "ppi", "jobless-claims", "retail-sales"}:
        follow_up.append("BLS (bls.gov) — CPI, PPI, jobless claims primary releases")
    if entity_keys_found & {"sp500", "nasdaq", "dow", "russell2000", "vix"}:
        follow_up.append("Exchange closing prices (NYSE, Nasdaq) — authoritative end-of-day data")

    # Triage summary
    n_conflicts = len(conflicts)
    n_shared    = len(shared_claims)
    n_mismatches = len(presence_absence_mismatches)
    n_unit_mismatches = len(unit_mismatches)
    n_shared_dates = len(shared_dates) + len(shared_years)
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
    if n_shared_dates > 0:
        triage_summary += (
            f" {n_shared_dates} shared date claim(s) across documents."
        )

    return {
        "doc_count":         len(analyzed_docs),
        "shared_claims":     shared_claims,
        "unique_claims":     unique_claims,
        "conflicts":         conflicts,
        "alignment":         alignment,
        "unit_mismatches":   unit_mismatches,
        "shared_dates":      shared_dates,
        "shared_years":      shared_years,
        "unique_dates":      unique_dates,
        "presence_absence_mismatches": presence_absence_mismatches,
        "presence_status":   presence_status,
        "crude_note":        crude_note,
        "follow_up_sources": follow_up,
        "triage_summary":    triage_summary,
    }
