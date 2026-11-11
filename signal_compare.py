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

# ── Benchmark table rows (v1.8.7, Arch gate point 1) ─────────────────────
# A list-item row names a benchmark and then a run of model scores:
#   "- FrontierCode 2.0 agentic coding: Fable 5.2 49.1% (Xhigh), Fable 5.1 32.4%, ..."
#   "- Humanity's Last Exam (with tools): Fable 5.2 74.2%, Fable 5.1 65.6%, ..."
# The mapped passes cannot read these rows: the inline pass meets the
# model's VERSION number first ("Fable 5.2") and stops, and the
# unmapped benchmarks (FrontierCode, CVE Rediscovery, OSWorld) are not
# in the entity map at all. A row is parsed structurally: the label
# before the first colon names the benchmark; each "Model Version %"
# pair after it is that model's score on that benchmark. The model is
# part of the claim's identity — one doc's row is a value set with
# different qualified identities, never a self-conflict.
_TABLE_ROW_RE = re.compile(r"^\s*[-•*]\s+(.+)$", re.M)
_TABLE_VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,2}\b|\bv\d+\b", re.I)


def _table_benchmark_name(label: str) -> str:
    """The benchmark name in a table-row label, versions and descriptors
    stripped: "FrontierCode 2.0 agentic coding" -> "FrontierCode",
    "CVE Rediscovery" -> "CVE Rediscovery", "Terminal-Bench-Science
    0.1" -> "Terminal-Bench-Science". Leading capitalized words only;
    a lowercase descriptor ("agentic", "business") ends the name."""
    clean = re.sub(r"\([^)]*\)", "", label)
    clean = _TABLE_VERSION_RE.sub("", clean)
    words = clean.split()
    name: List[str] = []
    for w in words:
        w = w.strip(",;")
        if not w:
            continue
        if w[0].isupper() or "-" in w:
            name.append(w)
        else:
            break
        if len(name) >= 3:
            break
    return " ".join(name)


def _parse_table_rows(text: str) -> List[Dict]:
    """Parse benchmark table rows into (benchmark, model, score) entries.

    Each entry: {"benchmark_key" (mapped key or ""), "benchmark_name",
    "model_slug", "model_display", "value" (float), "raw" ("% string"),
    "version" (mapped-benchmark version from the label, else ""),
    "variant" (mapped-benchmark variant from the label, else ""),
    "line" (the row text)}. Rows without a colon separator or without
    any model-score pair are not table rows and yield nothing."""
    entries: List[Dict] = []
    for line_m in _TABLE_ROW_RE.finditer(text):
        line = line_m.group(1).strip()
        if ":" not in line:
            continue
        label, rest = line.split(":", 1)
        label = label.strip()
        if not label:
            continue
        mapped = _normalize_entity_key(label)
        key = mapped[0] if mapped else ""
        name = _table_benchmark_name(label)
        if not key and not name:
            continue
        version = _benchmark_version(key, label) if key else ""
        variant = _benchmark_variant(key, label) if key else ""
        # v1.8.7: pairs belong to the row's first clause only. A
        # summary line ("- AA-Briefcase v1.1: all five columns match.
        # Remote Labor Index: ... RoboDojo Sim: Astra 29.8% ...")
        # is not a table row — the 29.8% is a RoboDojo score, not an
        # AA-Briefcase one, and the row parser must not reach past
        # the first sentence boundary to claim it.
        rest = re.split(r"(?<=[.!?])\s+", rest, maxsplit=1)[0]
        for pm in _TABLE_PAIR_RE.finditer(rest):
            model_display = pm.group(1).strip()
            raw = pm.group(2).strip()
            try:
                value = float(raw.rstrip("%").strip())
            except ValueError:
                continue
            entries.append({
                "benchmark_key":  key,
                "benchmark_name": name,
                "model_slug":     _model_slug(model_display),
                "model_display":  model_display,
                "value":          value,
                "raw":            raw,
                "version":        version,
                "variant":        variant,
                "line":           line,
            })
    return entries


# Pricing-row pass (Sieve gap 2026-11-01: a $15/$75 pricing row vs no row
# was invisible because pricing was not an entity). A $ amount counts as
# a pricing claim only when pricing words sit nearby, so market prices
# ("WTI Crude: $68.43/bbl") never match. Input/output words split the
# key so an input price and an output price are never conflated.
_PRICING_WORDS_RE = re.compile(
    r"\b(pric(?:e|ing)|per\s+(?:million|1M)\s+tokens?|per\s+(?:million|1M)\b|"
    r"tokens?|per[-\s]?task|cache[sd]?|batch|fast)\b", re.I)
_INPUT_WORD_RE = re.compile(r"\binput\b", re.I)
_OUTPUT_WORD_RE = re.compile(r"\boutput\b", re.I)
# v1.8.5 (Nov 6): "cached" and "per task" are their own pricing tiers —
# $0.03 cached input is not an input price and $1.18/task is not an
# output price, and "per task" must also count as a pricing word so a
# per-task price mints at all. The per-task regex requires "per" so
# "per index task" (a token count, not a price) never matches.
_CACHED_WORD_RE = re.compile(r"\bcached\b", re.I)
_PER_TASK_RE = re.compile(r"\bper[-\s]?task\b|\b/task\b", re.I)
# v1.8.7 (Arch gate point 1): fast-mode and batch prices are their own
# tiers too — "$16 per million input tokens" in a fast-mode sentence
# is a fast-input price, not the standard input price, and "$4 / $20
# batch" is a batch price. "cache" (not only "cached") names the
# cached tier ("cache reads ($0.20 against $0.25)"), and "per index
# task" IS a per-task price ("61.82 for $9.08 per index task").
_BATCH_WORD_RE = re.compile(r"\bbatch\b", re.I)
_FAST_WORD_RE = re.compile(r"\bfast\b", re.I)
_CACHE_WORD_RE = re.compile(r"\bcache\b", re.I)
_PER_TASK_RE = re.compile(r"\bper[-\s]?(?:index\s+)?task\b|\b/task\b", re.I)
# A "$A (and|/) $B ... input and output" construction assigns tiers
# POSITIONALLY: the first amount is the first tier word's price, the
# second amount the second's ("$8 and $40 per million input and
# output tokens" — $8 is input, $40 is output; the nearest-word rule
# keyed both as input). The tier words may come in either order.
_PRICING_PAIR_RE = re.compile(
    r"\$\s?(\d[\d,\.]*)\s*(?:and|/)\s*\$\s?(\d[\d,\.]*)"
    r"[^.!?\n]{0,60}?\b(input|output)\b\s*(?:and|/)\s*\b(output|input)\b",
    re.I,
)
# Price-era cues (v1.8.7, Arch gate point 2): a price is the PRIOR
# price when it is introduced as superseded ("down from $10 and
# $50"), or when it is on the second side of an "against/vs" pair in
# a sentence naming a prior model/version ("against $10 / $50 for
# Claude Fable 5.1", "cache reads ($0.20 against $0.25)" in a
# "cheaper than Fable 5.1" sentence). A current price and a prior
# price for the same tier are an expected contrast, never a conflict.
_PRIOR_CUE_RE = re.compile(
    r"\b(down from|previously|before|earlier|prior|old)\b",
    re.I,
)
_PRIOR_MODEL_RE = re.compile(
    r"\b(?:5\.1|previous|older|earlier|for\s+(?:claude\s+)?"
    r"(?:fable|mythos|opus|sonnet|gemini)\s+5\.1)\b",
    re.I,
)
_FAST_MODE_RE = re.compile(r"\bfast[-\s]?mode\b", re.I)


# A tier word that describes SPEED, not a price tier, must never
# resolve a tier: "Up to 2.5x higher output tokens per second" names
# the fast mode's output SPEED; the $16 / $80 after it are the fast
# INPUT and OUTPUT prices, not two output prices. A tier word
# followed by "tokens per second" is a speed descriptor.
_SPEED_DESCRIPTOR_RE = re.compile(r"^\s+tokens?\s+per\s+second\b", re.I)
_PAIR_AMOUNTS_RE = re.compile(r"\$\s?(\d[\d,\.]*)\s*(and|/)\s*\$\s?(\d[\d,\.]*)")


def _tier_word_is_speed(text: str, word_end: int) -> bool:
    """True when a tier word is followed by 'tokens per second' — a
    speed descriptor, not a price tier."""
    return bool(_SPEED_DESCRIPTOR_RE.match(text[word_end:word_end + 25]))


def _pricing_pair_tiers(text: str) -> Dict[int, str]:
    """Positional tier assignment for "$A (and|/) $B ... input and
    output" constructions: {amount_digit_start: tier}. The first
    amount takes the first tier word, the second amount the second.

    Mirror pairs: a "$A and $B" pair with no tier words after it
    takes the tier order of the sentence's earlier prices ("$8 and
    $40 per million input and output tokens, down from $10 and $50"
    — $10 is input, $50 is output, the same order as the sentence's
    earlier pair; "down from $10 and $50, respectively" likewise).
    When the most recent tier word before the pair is a tier FAMILY
    (batch, cached), the pair takes that family's input/output
    ("batch ($4 / $20 against $5 / $25)" — $5 is batch-input, $25
    batch-output). A slash pair ("$A / $B") with no tier words
    anywhere before it in the sentence defaults to input-then-output
    — API pricing pairs are listed input first; an "and" pair with
    no tier words stays unassigned (fail-closed: "$10 and $50 on
    standard tasks" for a different model must not become an
    input/output claim)."""
    out: Dict[int, str] = {}
    # Direct pairs: "$A and $B ... input and output".
    for m in _PRICING_PAIR_RE.finditer(text):
        out[m.start(1)] = m.group(3).lower()
        out[m.start(2)] = m.group(4).lower()
    # Mirror / default pairs.
    for m in _PAIR_AMOUNTS_RE.finditer(text):
        # Group 1 is the first amount, group 2 the separator ("and" /
        # "/"), group 3 the second amount.
        if m.start(1) in out or m.start(3) in out:
            continue
        # The sentence fragment before the pair (after the last
        # sentence break).
        before = text[max(0, m.start() - 200):m.start()]
        cut = -1
        for sep in ("\n", ". "):
            p = before.rfind(sep)
            if p != -1:
                cut = max(cut, p + len(sep))
        if cut != -1:
            before = before[cut:]
        # Tier words before the pair, in order (speed descriptors
        # excluded — "output tokens per second" is not a tier).
        tiers: List[str] = []
        for wm in re.finditer(r"\b(input|output|cached|cache|batch)\b", before, re.I):
            if _tier_word_is_speed(before, wm.end()):
                continue
            tiers.append(wm.group(1).lower())
        if tiers:
            last = tiers[-1]
            if last in ("cached", "cache"):
                out[m.start(1)] = "cached"
                out[m.start(3)] = "cached"
                continue
            if last == "batch":
                out[m.start(1)] = "batch-input"
                out[m.start(3)] = "batch-output"
                continue
            if len(tiers) >= 2:
                first, second = tiers[0], tiers[1]
                if first != second and first in ("input", "output") \
                        and second in ("input", "output"):
                    out[m.start(1)] = first
                    out[m.start(3)] = second
                    continue
        # Slash pairs with no tier words before them default to
        # input-then-output (API pricing convention). "And" pairs do
        # not default — they stay unassigned.
        if m.group(2) == "/":
            out[m.start(1)] = "input"
            out[m.start(3)] = "output"
    return out


def _pricing_era(sentence: str, amount_start: int) -> str:
    """Price era for one $ amount: "prior" when the amount is
    introduced as superseded ("down from $10") or sits on the second
    side of an "against"/"vs" pair in a sentence naming a prior
    model/version; "current" otherwise. A prior cue applies to the
    whole price list in its sentence ("Previously $10 input and $50
    output" — the $50 is prior too, not just the $10), unless a
    "now"/"currently"/"new" marker resets it after the cue."""
    before = sentence[:amount_start]
    prior_m = _PRIOR_CUE_RE.search(before)
    if prior_m:
        # A "now"/"currently"/"new" marker after the prior cue and
        # before the amount resets the era to current.
        if re.search(r"\b(?:now|currently|new)\b", before[prior_m.end():], re.I):
            return "current"
        return "prior"
    # Second side of an "against"/"vs" pair, in a sentence naming a
    # prior model/version: "$0.20 against $0.25" in a "cheaper than
    # Fable 5.1" sentence is the 5.1 price, not the 5.2 price.
    if re.search(r"\b(?:against|vs\.?)\b", sentence[:amount_start]) \
            and _PRIOR_MODEL_RE.search(sentence):
        return "prior"
    return "current"


def _pricing_subject(sentence: str, amount_start: int) -> str:
    """The model a per-task price belongs to: the nearest model token
    in the same clause before the amount ("61.82 for $9.08 per task"
    after "Fable 5.2 (High)" is Fable 5.2's per-task cost), or, when
    none precedes it, the model named just after it ("$10.04 for
    Opus 5.5 (Max)" is Opus 5.5's). "$12.82 per task" with no model
    token either side is subject-less. Only used for per-task
    partitioning — standard tiers are not split by subject."""
    clause_a, clause_b = _generic_clause_span(sentence, amount_start, amount_start)
    before = sentence[clause_a:amount_start]
    best = ""
    for m in _MODEL_TOKEN_RE.finditer(before):
        best = m.group(1)
    if best:
        return _model_slug(best)
    # No model before the amount: look just after it, where the price
    # is introduced as "for <Model>" ("$10.04 for Opus 5.5 (Max)").
    after = sentence[amount_start:clause_b]
    m = re.search(r"\bfor\s+(" + _MODEL_TOKEN_RE.pattern + r")", after)
    if m:
        return _model_slug(m.group(1))
    return ""


def _pricing_against_subject(sentence: str, amount_start: int) -> str:
    """The model a price belongs to when it is the second side of an
    "against" comparison naming a different model ("Fable 5.2 costs
    twice as much as Opus 5.5 ($8 / $40 against $4 / $20 per
    million)" — the $4/$20 are Opus 5.5's, not Fable 5.2's, and must
    not compare against Fable's $8/$40). The subject is the model in
    the sentence's "than/as <Model>" comparison phrase. Returns ""
    when the amount is not after an "against" or the sentence names
    no comparison model — the claim is then subject-less and
    partitions with the other subject-less prices of its tier."""
    before = sentence[:amount_start]
    if not re.search(r"\b(?:against|vs\.?)\b", before):
        return ""
    m = re.search(
        r"\b(?:than|as)\s+(" + _MODEL_TOKEN_RE.pattern + r")", before)
    return _model_slug(m.group(1)) if m else ""
# v1.8.5 (Nov 6): a tier word immediately before a $ amount — "input
# $15", "Input: $0.30" — is that amount's own label. Only whitespace
# or a colon may sit between the word and the $, so "output and $0.03"
# (the previous price's tier) never matches.
_BACKWARD_TIER_RE = re.compile(r"\b(input|output|cached|cache|batch|per[-\s]?task|fast)\b\s*:?\s*$", re.I)
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
    # ISO dates, with an optional time component (v1.8.6, P5): a bare
    # "2026-11-07T00:58:18Z" failed the old trailing \b (no word
    # boundary between the day digits and the "T"), so every ISO
    # timestamp was invisible to date extraction — the Nov 7 packet
    # lost both the SecurityWeek published/updated timestamps and the
    # TSN updated timestamp. The time is matched and masked with the
    # date; the date_key stays YYYY-MM-DD.
    r"|\b(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b",
    re.I,
)
_YEAR_RE = re.compile(
    r"\b(?:in|from|by|during|starting|effective|early|late|mid)[\s\-]+(20\d{2})\b",
    re.I,
)


def _date_role(text: str, start: int, end: int) -> str:
    """Role of a date occurrence: event / published / updated / fetched /
    access / unknown, from the label cues around it.

    v1.8.6 (P5): dates are not one undifferentiated pile — a release
    heading date, a published_at timestamp, and a "Fetched 2026-11-08"
    feed header are different kinds of dates, and a shared date that is
    only ever a fetched date is a pipeline artifact, not a shared claim.
    The role is read off the nearest label cue in a window BEFORE the
    match; the window ends at the match, because a "Published: X,
    updated Y" header would otherwise let "updated" bleed into X's
    window and mislabel the published date. Anything without a cue
    is "unknown" — honest, not guessed.
    """
    window = text[max(0, start - 70):end].lower()
    # Nearest cue wins: the label closest to the match is the one that
    # labels it. A "Fetched" header in the previous sentence must not
    # outrank a "Published:" immediately before this date (the earlier
    # priority-order read did exactly that: "Fetched 2026-11-08.
    # Published: November 7, 2026." mislabeled November 7 as fetched).
    cues = (
        ("fetched", ("fetch",)),
        ("updated", ("updat",)),
        ("published", ("publish",)),
        ("access", ("access",)),
        ("event", ("heading date", "release")),
    )
    best_role = "unknown"
    best_pos = -1
    for role, words in cues:
        for word in words:
            pos = window.rfind(word)
            if pos > best_pos:
                best_pos = pos
                best_role = role
    return best_role


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
        # v1.8.7: a model/benchmark version token ("Fable 5.2", "OSWorld
        # 2.1") is an identity, not a score — skip it and keep looking
        # for the real score later in the line.
        if _is_model_version_number(text, num_start, num_end):
            continue
        if _is_setting_number(text, num_start):
            continue
        if key in _BENCHMARK_KEYS and _is_following_benchmark_score(text, num_end):
            continue
        bare = _unit_of(raw_val) == ""
        count_after = bare and _count_after(text, num_end)
        # v1.8.7: the score belongs to the model named just before it
        # ("Fable 5.2 74.2%" is Fable 5.2's score) — record the
        # subject so two models' scores on one benchmark never lump.
        subject = ""
        model_tok = _model_token_before(text, num_start)
        if model_tok:
            subject = _model_slug(model_tok)
        _add(key, label, raw_val, text[m.start():num_end],
             unit_override="count" if count_after else "",
             version=_benchmark_version(key, text[m.start():num_end]),
             shot_count=_shot_count_in_window(text[m.start():num_end]),
             variant=variant,
             subject=subject)
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


# Model tokens (v1.8.7, Arch gate point 1): a model name is an
# identity, not a measurement. "Fable 5.2", "Opus 5.5", "Gemini 4
# Carbon", "GPT-6 Astra", "Claude Opus 5.5", "Astra", "Carbon" — the
# version digits belong to the name and must never mint as a score,
# a cost, or a count. A bare capitalized word ("Cyber", "Sol") is NOT
# a model token on its own — it only counts attached to a versioned
# name ("GPT-6 Cyber", "GPT-5.6 Sol") — so "Cyber Verification
# Program" never becomes a subject.
_MODEL_FAMILY_WORDS = frozenset({
    "fable", "mythos", "opus", "sonnet", "gemini", "astra", "carbon",
    "gpt", "claude", "grok", "llama", "mistral", "qwen", "deepseek",
    "inkling", "ernie",
})
_MODEL_TOKEN_RE = re.compile(
    r"\b((?:Claude\s+)?(?:Fable|Mythos|Opus|Sonnet|Gemini|Grok|Llama|Mistral|Qwen|DeepSeek|Inkling|ERNIE|Astra|Carbon)"
    r"(?:\s+\d+(?:\.\d+)?(?:\s+(?:Astra|Carbon|Cyber|Sol))?)?"
    r"|GPT-?\d+(?:\.\d+)?(?:\s+(?:Astra|Carbon|Cyber|Sol))?)\b"
)
# A token that can OWN a version number: a model token, a CamelCase
# benchmark name (FrontierCode, OSWorld, RoboDojo), or a hyphenated
# benchmark name (CL-Bench, Terminal-Bench-Science). A single
# Titlecase word ("Fable", "Opus") is a model token, not a
# version-owner on its own — the model-token regex covers those.
_VERSION_OWNER_RE = re.compile(
    r"(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,}[A-Za-z0-9._-]*"
    r"|[A-Z][A-Za-z0-9._-]*-[A-Z][A-Za-z0-9._-]*)\s*$"
)
# A "Model Version %" pair inside a table row. Defined here because
# it embeds the model-token pattern, which is defined just above.
_TABLE_PAIR_RE = re.compile(_MODEL_TOKEN_RE.pattern + r"\s+(\d+\.?\d*\s*%)")


def _model_slug(token: str) -> str:
    """Canonical slug for a model token ("Claude Opus 5.5" -> "opus-5-5").
    Bare display names resolve to their one unambiguous full model:
    "Astra" is GPT-6 Astra, "Carbon" is Gemini 4 Carbon — a bare
    "Astra 42.8%" and a "GPT-6 Astra 42.8%" are the same model's
    score, not two subjects. Versioned names (Opus, Sonnet, Fable)
    are ambiguous across versions and are NOT aliased."""
    token = re.sub(r"^claude\s+", "", token.strip(), flags=re.I)
    slug = re.sub(r"[^a-z0-9]+", "-", token.lower()).strip("-")
    return {"astra": "gpt-6-astra", "carbon": "gemini-4-carbon"}.get(slug, slug)


def _model_token_before(text: str, num_start: int, window: int = 40) -> str:
    """The model token immediately before a number (only whitespace
    between), or "" when none. Used to give a table-row score its
    measured subject ("Fable 5.2 74.2%" is Fable 5.2's score)."""
    before = text[max(0, num_start - window):num_start]
    best = ""
    for m in _MODEL_TOKEN_RE.finditer(before):
        if before[m.end():].strip() == "":
            best = m.group(1)
    return best


def _is_model_version_number(text: str, num_start: int, num_end: int) -> bool:
    """True when a number is a version token of a model/benchmark name,
    not a value. The number must be version-shaped — a dotted d.d
    ("5.2", "2.1", "1.0") or a bare integer ("4" in "Gemini 4") —
    with no unit or currency suffix, and a model token or a
    version-owner token (CamelCase/hyphenated benchmark name)
    immediately before it. A percentage ("42.8%") or a price
    ("$0.30") is never a version, whatever precedes it."""
    raw = text[num_start:num_end].rstrip(",.")
    if raw.endswith("%") or "$" in raw:
        return False
    if not re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return False
    if "." in raw:
        # Dotted: only single-digit parts read as a version
        # ("5.2", "2.1"); "48.5" and "61.82" are measurements.
        if not re.fullmatch(r"\d\.\d", raw):
            return False
    before = text[max(0, num_start - 40):num_start]
    for m in _MODEL_TOKEN_RE.finditer(before):
        if before[m.end():].strip() == "":
            return True
    if _VERSION_OWNER_RE.search(before):
        return True
    return False


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
    Returns [{date_key, date_label, kind, raw, context}] — full-date
    claims also carry date_role / date_roles (v1.8.6, P5, see
    _date_role): a document that mentions the same date more than once
    — e.g. the AI Weekly feed's 2026-11-08 as published, updated AND
    fetched — contributes one claim per date_key, with every role it
    holds. A date that is only ever "fetched" is a pipeline artifact,
    not a shared claim.
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
        role = _date_role(text, m.start(), m.end())
        if key in seen:
            for c in claims:
                if c["date_key"] == key and role not in c["date_roles"]:
                    c["date_roles"].append(role)
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
            "date_role":  role,
            "date_roles": [role],
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
             version: str = "", shot_count: Optional[int] = None, variant: str = "",
             subject: str = "", era: str = "") -> None:
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
        # v1.8.7: the same key and value for different measured
        # subjects (Fable 5.2 vs Opus 5.5 on one benchmark) or
        # different price eras (current vs prior) are DIFFERENT
        # claims too — the dedup key carries subject and era.
        dedup   = (key, numeric, variant, subject, era)
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
            "subject":       subject,
            "era":           era,
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
        subject = ""
        model_tok = _model_token_before(text, m.start(2))
        if model_tok:
            subject = _model_slug(model_tok)
        _add(key, label, raw_val, m.group(0),
             unit_override="count" if count_after else "",
             version=_benchmark_version(key, m.group(0)),
             shot_count=_shot_count_for_match(text, m, 2),
             variant=_benchmark_variant(key, ematch.group(0), label_fragment[ematch.end():]),
             subject=subject)

    # Pass 2 — inline patterns (catches prose and snapshot lines)
    for m in _INLINE_ENTITY_RE.finditer(text):
        entity_frag = m.group(1).strip()
        raw_val     = m.group(2).strip()
        if not raw_val or raw_val == ".":
            continue
        if _is_version_number(text, m.start(2)) or \
                _is_model_version_number(text, m.start(2), m.end(2)):
            # v1.8.7: the number after the entity is a version token
            # ("Humanity's Last Exam (with tools): Fable 5.2 74.2%" —
            # the 5.2 is Fable's version, not the benchmark's score).
            # Look for the real score later in the same line instead
            # of minting the version or losing the score.
            entity = _normalize_entity_key(entity_frag)
            if entity is None:
                continue
            key, label, ematch = entity
            frag = text[m.start():m.end()]
            between = frag[ematch.end():m.start(2) - m.start()]
            variant = _benchmark_variant(key, ematch.group(0), between)
            _rescan_inline_score(text, m, key, label, variant, _add)
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
        # v1.8.7: the score belongs to the model named just before it.
        subject = ""
        model_tok = _model_token_before(text, m.start(2))
        if model_tok:
            subject = _model_slug(model_tok)
        _add(key, label, raw_val, m.group(0),
             unit_override="count" if count_after else "",
             version=_benchmark_version(key, m.group(0)),
             shot_count=_shot_count_for_match(text, m, 2),
             variant=variant,
             subject=subject)

    # Pass 3 — benchmark-reverse: "61.3% on GPQA" (score precedes the name)
    for m in _BENCHMARK_REVERSE_RE.finditer(text):
        raw_val = m.group(1).strip()
        if _is_version_number(text, m.start(1)):
            continue
        entity = _normalize_entity_key(m.group(2).strip())
        if entity is None:
            continue
        key, label, ematch = entity
        subject = ""
        model_tok = _model_token_before(text, m.start(1))
        if model_tok:
            subject = _model_slug(model_tok)
        _add(key, label, raw_val, m.group(0),
             version=_benchmark_version(key, m.group(0)),
             shot_count=_shot_count_for_match(text, m, 1),
             variant=_benchmark_variant(key, ematch.group(0), "", text[m.end():m.end() + 20]),
             subject=subject)

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
    # v1.8.7 (Arch gate points 1–2): tier resolution is positional
    # first ("$8 and $40 per million input and output tokens" assigns
    # $8→input, $40→output; a "$A / $B" slash pair takes the
    # complementary tier of its first element), then backward-adjacent,
    # then forward, then clause-level (the nearest tier word after the
    # previous $ in the same clause — "cached tokens ... cost $0.20",
    # "batch ($4 / $20 ...)"). Fast-mode and batch prices are their
    # own tiers; a price carries an era (current vs prior — "down
    # from $10", "against $10 / $50 for Fable 5.1"); a per-task price
    # carries its measured subject. A $ amount with no resolvable tier
    # mints as bare "pricing" and is never compared (fail-closed).
    pair_tiers = _pricing_pair_tiers(text)
    prev_amount: Dict = {}  # the previous $ amount's tier/era, for slash pairs
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
        # The sentence containing this amount, for era / subject /
        # fast-mode / clause-level tier checks.
        sent_start = max(text.rfind("\n", 0, m.start()),
                         text.rfind(". ", 0, m.start())) + 1
        sent_end = len(text)
        for sep in ("\n", ". "):
            p = text.find(sep, m.end())
            if p != -1:
                sent_end = min(sent_end, p)
        sentence = text[sent_start:sent_end]
        amount_off = m.start(1) - sent_start
        # Tier ownership (v1.8.5 + v1.8.7): positional pair first,
        # then backward-adjacent, then forward, then clause-level.
        # A slash pair ("$4 / $20") takes the complementary tier of
        # its first element when the first element resolved a tier.
        tier = pair_tiers.get(m.start(1), "")
        if not tier:
            between_prev = ""
            if prev_amount and prev_amount.get("end") is not None:
                between_prev = text[prev_amount["end"]:m.start()]
            if prev_amount.get("tier") and re.fullmatch(r"\s*/\s*", between_prev or ""):
                prev_tier = prev_amount["tier"]
                tier = {
                    "input": "output", "output": "input",
                    "fast-input": "fast-output", "fast-output": "fast-input",
                    "batch-input": "batch-output", "batch-output": "batch-input",
                    "cached": "cached", "per-task": "per-task",
                }.get(prev_tier, "")
        if not tier:
            back = _BACKWARD_TIER_RE.search(window[max(0, dollar_pos - 16):dollar_pos])
            if back and not _tier_word_is_speed(
                    window[max(0, dollar_pos - 16):dollar_pos], back.end()):
                raw_tier = back.group(1).lower()
                if raw_tier in ("cache", "cached"):
                    tier = "cached"
                elif raw_tier.replace(" ", "-") in ("per-task",):
                    tier = "per-task"
                elif raw_tier in ("input", "output", "batch"):
                    tier = raw_tier
                # "fast" alone is not a tier — it only qualifies
                # input/output, which the fast-mode check below handles.
        if not tier:
            nearest_dist = None
            for word_re, word_key in ((_INPUT_WORD_RE, "input"), (_OUTPUT_WORD_RE, "output"),
                                      (_CACHED_WORD_RE, "cached"), (_CACHE_WORD_RE, "cached"),
                                      (_BATCH_WORD_RE, "batch"), (_FAST_WORD_RE, "fast"),
                                      (_PER_TASK_RE, "per-task")):
                for wm in word_re.finditer(window, amount_end, seg_end):
                    if _tier_word_is_speed(window, wm.end()):
                        continue
                    # A tier word that directly introduces the NEXT
                    # price ("batch ($4 ...") belongs to that price,
                    # not this one — skip it when only whitespace/an
                    # open paren sits between it and the next $.
                    if seg_end < len(window) and window[seg_end:seg_end + 1] == "$" \
                            and re.fullmatch(r"\s*\(\s*", window[wm.end():seg_end]):
                        continue
                    dist = wm.start() - dollar_pos
                    if dist <= 30 and (nearest_dist is None or dist < nearest_dist):
                        tier, nearest_dist = word_key, dist
        if not tier:
            # Clause-level: the nearest tier word after the previous $
            # in the same clause ("cached tokens ... cost $0.20",
            # "batch ($4 / $20 ...)"). A tier word that belongs to a
            # previous price (separated by another $) never counts.
            # The previous $ is the one before THIS amount's own $
            # (at amount_off - 1) — rfind up to amount_off would find
            # the amount's own $ and leave an empty clause.
            own_dollar = m.start() - sent_start
            prev_dollar = sentence.rfind("$", 0, own_dollar)
            clause_start = prev_dollar + 1 if prev_dollar != -1 else 0
            clause_before = sentence[clause_start:amount_off]
            best_pos, best_tier = -1, ""
            for word_re, word_key in ((_INPUT_WORD_RE, "input"), (_OUTPUT_WORD_RE, "output"),
                                      (_CACHED_WORD_RE, "cached"), (_CACHE_WORD_RE, "cached"),
                                      (_BATCH_WORD_RE, "batch"), (_PER_TASK_RE, "per-task")):
                for wm in word_re.finditer(clause_before):
                    if _tier_word_is_speed(clause_before, wm.end()):
                        continue
                    if wm.start() > best_pos:
                        best_pos, best_tier = wm.start(), word_key
            if best_tier:
                tier = best_tier
        if not tier:
            # "Against"-pair inheritance: the second element of an
            # "X against Y" pair takes the first element's tier when
            # the pair names one price list ("cache reads ($0.20
            # against $0.25)" — both are cache-read prices). It does
            # NOT inherit when the amount is qualified as another
            # model's price ("$0.20 per million against $1.00 for
            # Astra" — the $1.00 is Astra's, not a cache price).
            if prev_amount.get("tier") and prev_amount.get("end") is not None:
                between_prev = text[prev_amount["end"]:m.start()]
                if re.fullmatch(r"\s*(?:against|vs\.?)\s*", between_prev or ""):
                    after_amt = sentence[amount_off:amount_off + 30] \
                        if amount_off < len(sentence) else ""
                    if not re.search(r"\bfor\s+" + _MODEL_TOKEN_RE.pattern, after_amt):
                        tier = prev_amount["tier"]
        if not tier:
            # Sentence-level per-task: a "$X" introduced by "for" in
            # a sentence that names a per-task price elsewhere ("61.82
            # for $9.08 per task, against 59.11 for $10.04 ... (48.21
            # for $2.45 against 56.41 for $3.80)") is itself a
            # per-task price — the tier word belongs to the sentence's
            # comparison, not only to the first amount's clause.
            own_dollar = m.start() - sent_start
            before_amt = sentence[max(0, own_dollar - 6):own_dollar]
            if re.search(r"\bfor\s*$", before_amt) and _PER_TASK_RE.search(sentence):
                tier = "per-task"
        # Fast mode qualifies an input/output tier; a fast-mode price
        # with no input/output word is unresolvable (bare, never compared).
        is_fast = bool(_FAST_MODE_RE.search(sentence))
        if tier in ("input", "output") and is_fast:
            tier = "fast-" + tier
        elif tier == "batch":
            # A bare batch amount is the first of a batch pair (or a
            # lone batch price): batch-input. Its slash partner takes
            # the complement (batch-output) via the pair rule above.
            tier = "batch-input"
        era = _pricing_era(sentence, amount_off)
        # A slash-pair complement inherits the first element's era.
        if prev_amount.get("tier") and re.fullmatch(r"\s*/\s*", text[prev_amount.get("end", m.start()):m.start()] or ""):
            era = prev_amount.get("era", era)
        subject = _pricing_subject(sentence, amount_off) if tier == "per-task" else ""
        if not subject:
            # A standard-tier price on the second side of an "against"
            # comparison naming a different model belongs to that
            # model ("$8 / $40 against $4 / $20" for Opus 5.5) — it
            # must not compare against the first model's prices.
            subject = _pricing_against_subject(sentence, amount_off)
        if tier == "input":
            key, label = "pricing-input", "Pricing (input)"
        elif tier == "output":
            key, label = "pricing-output", "Pricing (output)"
        elif tier == "cached":
            key, label = "pricing-cached", "Pricing (cached)"
        elif tier == "per-task":
            key, label = "pricing-per-task", "Pricing (per task)"
        elif tier == "fast-input":
            key, label = "pricing-fast-input", "Pricing (fast input)"
        elif tier == "fast-output":
            key, label = "pricing-fast-output", "Pricing (fast output)"
        elif tier == "batch-input":
            key, label = "pricing-batch-input", "Pricing (batch input)"
        elif tier == "batch-output":
            key, label = "pricing-batch-output", "Pricing (batch output)"
        else:
            key, label = "pricing", "Pricing"
        _add(key, label, m.group(1), window.strip()[:120], unit_override="$",
             subject=subject, era=era)
        prev_amount = {"tier": tier, "era": era, "end": m.end()}

    # Pass 5 — benchmark table rows (v1.8.7, Arch gate point 1): a
    # list-item row names a benchmark and then a run of model scores.
    # Mapped benchmarks mint mapped claims (subject = the model);
    # unmapped benchmarks are handled by the generic fallback's
    # table-row pass in extract_generic_quantitative_claims.
    for row in _parse_table_rows(text):
        if not row["benchmark_key"]:
            continue
        key = row["benchmark_key"]
        label = _normalize_entity_key(row["benchmark_name"])
        display = label[1] if label else row["benchmark_name"]
        _add(key, display, row["raw"], row["line"][:120],
             version=row["version"], variant=row["variant"],
             subject=row["model_slug"])

    if dropped_out is not None:
        dropped_out.extend(dropped)

    return claims


# ── Generic quantitative-claim fallback (v1.8.6, P2) ──────────────────────
# Benchmark/pricing extraction only covers mapped entities. Vendor
# announcements and release notes repeat figures — benchmark scores,
# counts, percentages, costs — that no mapped entity catches, so compare
# used to report zero shared claims on documents that agree everywhere
# (Nov 7 GPT-6 Cyber packet, Nov 8 ADK 2.14.0 packet: 0 shared, 0
# conflicts, 0 aligned, on docs agreeing on every figure). The fallback
# extracts bounded generic quantitative claims from ordinary sentences
# and compares only compatible identities: same metric family, same
# measured subject (when stated), same context numbers (denominator),
# same unit. It never expands the benchmark map, and it never calls
# repetition "independent verification" — shared entries carry their
# attribution, and a single-originator figure says so on the entry.
#
# Bounds, by design:
#   - a number needs a metric cue (a verb or count noun naming what it
#     measures) — metric-less numbers are skipped, never guessed;
#   - versions, years, times, hyphenated modifiers (30-node, 200-turn),
#     multipliers ("five times"), error codes and URLs are not claims;
#   - number words only for three–nineteen, tens and hyphenated
#     compounds, only next to a count noun (one/two excluded);
#   - currency forces family "cost" (or "grants" when the clause names
#     a grant); a per-unit denominator separates totals from per-unit
#     figures ($1,279 per LPE exploit is not $1,213,430 total);
#   - the measured subject is the nearest product token in the claim's
#     own segment (since the previous number / until the next one), so
#     "99.93% ... compared to 94.49% for GPT-5.6 Cyber" does not pin
#     the 99.93% on GPT-5.6;
#   - the benchmark name is recorded on the claim but is NOT part of
#     the identity: a benchmark named in one doc and unnamed in another
#     ("RemoteExploitBench" in the TSN 95% sentence, unnamed in the
#     primary's) is the same claim. Two different NAMED benchmarks in
#     one group is a benchmark mismatch instead (see compare_documents).
_GENERIC_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_GENERIC_CLAUSE_SPLIT_RE = re.compile(r"[,;]|\s+and\s+")
_GENERIC_DIGIT_RE = re.compile(
    # One greedy digit run (commas allowed, must END on a digit):
    # "2026" must match whole — the old \d{1,3} alternative split it
    # into "202" + "6" and minted two spurious claims, and a trailing
    # comma ("July 2026,") must not join the match or the year guard
    # fails. "1,724", "2.8", "48.5", "0.30" all match. "trillion"
    # joins million/billion/thousand as a magnitude suffix (v1.8.7:
    # "$2 trillion" is $2,000,000,000,000, not $2).
    r"([$£€]?)(\d(?:[\d,]*\d)?(?:\.\d+)?)"
    r"(\s*(?:%|million|billion|trillion|thousand))?"
)
_GENERIC_NUMBER_WORDS = {
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_GENERIC_NUMBER_WORD_RE = re.compile(
    r"\b((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
    r"(?:one|two|three|four|five|six|seven|eight|nine)"
    r"|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
    r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty"
    r"|forty|fifty|sixty|seventy|eighty|ninety)\b",
    re.I,
)
# Product tokens: an ALLCAPS-ish name, optionally versioned, optionally
# followed by a display word ("GPT-6 Cyber", "GPT-5.6 Sol", "GPT-6").
# CamelCase names (ExploitGym, OpenAI, RemoteExploitBench) are not
# product tokens — they are benchmarks or orgs, caught elsewhere.
_GENERIC_PRODUCT_RE = re.compile(
    r"\b([A-Z]{2,}(?:[-.][A-Za-z0-9]+)*(?:\s+(?:[A-Z][a-zA-Z]+|[A-Z]{2,})){0,2})\b"
)
_GENERIC_PRODUCT_BASE_STOPLIST = {
    "lpe", "api", "url", "cve", "http", "https", "html", "css", "sql",
    "json", "sdk", "gpu", "cpu", "ram", "tls", "ssl", "dns", "ai", "ml",
    "llm", "cli", "tbd", "faq", "pdf", "doc", "jpg", "png", "svg", "xml",
    "csv", "io", "os", "vm",
}
# Metric cues, in list order only as a tiebreak — the metric function
# picks by NEARNESS to the number, not by list position (a cue that
# belongs to another number in the same sentence must not win).
_GENERIC_METRIC_CUES = [
    (re.compile(r"\brecall\b", re.I), "recall"),
    (re.compile(r"percentage[- ]points?", re.I), "percentage-point gap"),
    (re.compile(r"relative improvement", re.I), "relative improvement"),
    (re.compile(r"\bfaster\b", re.I), "speed"),
    (re.compile(r"\bmemory\b", re.I), "memory"),
    (re.compile(r"\bsampl\w*\b", re.I), "samples"),
    (re.compile(r"out-of-scope", re.I), "out-of-scope avoidance"),
    (re.compile(r"refrain\w*\s+.{0,30}exploit|exploit\w*\s+.{0,30}refrain", re.I),
     "out-of-scope avoidance"),
    (re.compile(r"proofs? of concept|\bpocs?\b", re.I), "proofs of concept"),
    (re.compile(r"\btechniques?\b", re.I), "techniques"),
    (re.compile(r"disclos\w*", re.I), "findings disclosed"),
    (re.compile(r"reported\b.{0,60}\bto maintainers\b", re.I), "findings disclosed"),
    # "exploit" as a whole word only: "ExploitGym" must not fire it
    # (a benchmark name is not an exploit count — "48.5% on
    # ExploitGym" is a score), and "exploitation" as an adjective
    # ("10 new exploitation techniques") must not fire it either.
    (re.compile(r"\bexploit(?:s|ed|ing)?\b", re.I), "exploits"),
    (re.compile(r"\blpe\b", re.I), "exploits"),
    (re.compile(r"reproduc\w*", re.I), "vulnerabilities"),
    (re.compile(r"\bpatch\w*\b|\bstaging\b", re.I), "patches"),
    (re.compile(r"\bmodules?\b", re.I), "modules"),
    (re.compile(r"\btoolsets?\b|\bintegrations?\b", re.I), "features"),
    (re.compile(r"\bfeatures?\b", re.I), "features"),
    (re.compile(r"\bcomponents?\b", re.I), "components"),
    (re.compile(r"breaking changes?", re.I), "breaking changes"),
    (re.compile(r"vulnerabilit\w*", re.I), "vulnerabilities"),
    # v1.8.7: "1,829 CVEs" is a vulnerability count, not a score —
    # the CVE token right after the number names the family.
    (re.compile(r"\bCVEs?\b", re.I), "vulnerabilities"),
    (re.compile(r"\bflaws?\b|\bissues?\b", re.I), "vulnerabilities"),
    (re.compile(r"\bfindings?\b", re.I), "findings"),
    (re.compile(r"\brefus\w*\b", re.I), "refusal"),
    (re.compile(r"\bcomplet\w*\b", re.I), "completion"),
    (re.compile(r"\bscores?\b|\bscored\b|\bachiev\w*\b|\breached\b", re.I), "score"),
    # v1.8.7: "mean rewards", "performance", "graded criteria" name a
    # score just as "score" does — a bare decimal under one of these
    # cues is a measurement, not a task count.
    (re.compile(r"\brewards?\b", re.I), "score"),
    (re.compile(r"\bperformance\b", re.I), "score"),
    (re.compile(r"\bcriteria\b", re.I), "score"),
    # v1.8.7: count-noun families for the run 10/11 headline figures
    # ("41 of 57 facts", "20,000 tokens per question", "55% fewer
    # blocks", "1,000 agents", "128 dreaming sessions").
    (re.compile(r"\bfacts?\b", re.I), "facts"),
    (re.compile(r"\btokens?\b", re.I), "tokens"),
    (re.compile(r"\bblocks?\b", re.I), "blocks"),
    (re.compile(r"\bsubagents?\b|\bagents?\b", re.I), "agents"),
    (re.compile(r"\bsessions?\b", re.I), "sessions"),
    (re.compile(r"\bretention\b", re.I), "retention"),
    # "48.5% on ExploitGym": a number followed by "on <Benchmark>" is
    # a score on that benchmark.
    (re.compile(r"\bon\s+[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"), "score"),
    (re.compile(r"\btasks?\b", re.I), "tasks"),
    (re.compile(r"\btargets?\b", re.I), "targets"),
    (re.compile(r"\bgrants?\b", re.I), "grants"),
    (re.compile(r"\bcandidates?\b", re.I), "candidates"),
    (re.compile(r"lines? of code|code ?base", re.I), "code"),
    (re.compile(r"\bcost\b|\bcredits?\b", re.I), "cost"),
]
_GENERIC_FAMILY_LABELS = {
    "score": "score", "recall": "recall", "speed": "workflow speed",
    "memory": "peak memory", "modules": "modules", "features": "features",
    "components": "components", "breaking changes": "breaking changes",
    "exploits": "exploits", "vulnerabilities": "vulnerabilities",
    "findings": "findings", "findings disclosed": "critical findings disclosed",
    "patches": "patches", "refusal": "task refusal",
    "completion": "task completion",
    "out-of-scope avoidance": "out-of-scope avoidance rate",
    "relative improvement": "relative improvement",
    "percentage-point gap": "percentage-point gap",
    "cost": "cost", "grants": "grants", "candidates": "candidate findings",
    "targets": "audit targets", "techniques": "exploitation techniques",
    "proofs of concept": "proofs of concept", "tasks": "tasks",
    "code": "code base share", "samples": "sampled items",
    "facts": "facts", "tokens": "tokens", "blocks": "blocks",
    "agents": "agents", "sessions": "sessions", "retention": "retention",
    "size": "size", "duration": "duration",
}
# Count-noun families (v1.8.7): a non-integer bare decimal (0.838,
# 61.82) is a measurement, never a count of these — the decimal guard
# in _generic_metric excludes them for such numbers, so a version
# fragment or a score is never read as a task count.
_GENERIC_COUNT_NOUN_FAMILIES = {
    "tasks", "findings", "components", "modules", "features",
    "techniques", "proofs of concept", "samples", "exploits",
    "vulnerabilities", "patches", "grants", "candidates", "targets",
    "facts", "tokens", "blocks", "agents", "sessions", "breaking changes",
    "code",
}
# A month-day date ("April 7", "November 6", "Nov. 9") is a date, never
# a claim value (v1.8.7: the 7 in "April 7" must never become a
# critical-findings count, and the 6 in "November 6" never a component
# count).
_GENERIC_MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|"
    r"Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)\s*$", re.I)
_GENERIC_MONTH_AFTER_RE = re.compile(
    r"^\s*(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|"
    r"Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)\b", re.I)
# A duration mints only under a completion verb (v1.8.7): "took 30
# hours", "completed ... in 18 hours", "took 11 days" are durations;
# "within 12 hours", "after 10 days", "for eight days" stay skipped.
_GENERIC_COMPLETION_VERB_RE = re.compile(
    r"\b(?:took|completed?|finished|spent|lasted|formalized?|formalised?|built|solved|proved|achieved)\b|\bin just\b", re.I)
_GENERIC_SIZE_UNIT_RE = re.compile(r"\s*(KB|MB|GB|TB)\b", re.I)
_GENERIC_CTX_NUM_RE = re.compile(
    # Denominators only: node/turn/run/test/task/workflow/user. Day and
    # hour are durations, not denominators ("within 12 hours" is not a
    # claim context — the ledger lists durations as not extracted).
    r"(\d+(?:\.\d+)?)[- ](node|turn|run|test|task|workflow|user)s?\b",
    re.I,
)
_GENERIC_DENOM_RE = re.compile(
    # At most two words after per/of — a longer capture swallows the
    # rest of the clause (\"per LPE exploit by OpenAI's arithmetic\")
    # and splits one denominator into two different keys.
    r"\b(?:per|out of|of)\s+([A-Za-z][\w\-]{0,20}(?:\s+[A-Za-z][\w\-]{0,20})?)"
)
_GENERIC_QUALIFIER_RE = re.compile(
    r"\b(about|approximately|around|roughly|nearly|almost|up to|more than"
    r"|over|at least|just over|just under)\b|~",
    re.I,
)
_GENERIC_UPPER_BOUND_QUALS = {"up to", "at most", "no more than"}
_GENERIC_APPROX_QUALS = {"about", "approximately", "~", "roughly", "nearly", "around", "almost"}
# Verb-driven families: on a near-tie the governing verb names the
# count ("sampled 166 exploits" is a sample count). Derived families
# never hand their metric down to a baseline (a percentage-point gap
# is not the metric of the score it measures).
_GENERIC_VERB_DRIVEN_FAMILIES = {"samples", "findings disclosed"}
_GENERIC_DERIVED_FAMILIES = {"percentage-point gap", "relative improvement"}
_GENERIC_COMPARISON_MARKER_RE = re.compile(
    r"\bcompared (?:with|to)\b|\bversus\b|\bvs\.?\b|\bup from\b|\bdown from\b"
    r"|\bhigher than\b|\blower than\b|\bagainst\b",
    re.I,
)
_GENERIC_PRODUCTION_VERB_RE = re.compile(
    r"\b(?:produced|developed|found|identified|generated|created|reported"
    r"|disclosed|scored|achieved|reached|built|ran|conducted)\b",
    re.I,
)
_GENERIC_CAMEL_RE = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+")
_GENERIC_BENCH_ORG_STOPLIST = {
    "securityweek", "github", "gitlab", "huggingface", "coderabbit",
    "openai", "google", "microsoft", "mozilla", "meta", "amazon",
    "anthropic", "deepmind", "mistral", "cisco", "teamsrunner",
    "multirunconfig", "toolargsmatcher",
}
_GENERIC_ASSERT_RE = re.compile(
    r"\b([A-Z][A-Za-z]{2,})\s+(?:says|said|claims|claimed|reports|reported"
    r"|states|stated|announced|confirmed)\b"
)
_GENERIC_ACCORDING_RE = re.compile(r"according to\s+([A-Z][A-Za-z]{2,})")
_GENERIC_HEADER_ORG_RE = re.compile(
    r"^(?:PRIMARY SOURCE|PRIMARY COMPANION|SECONDARY(?: SOURCE)?|TERTIARY(?: SOURCE)?|COVERAGE)\s*[—\-–]\s*([A-Z][A-Za-z0-9_-]{1,30}(?: +[A-Z][a-z][A-Za-z0-9_-]{0,29}){0,3})"
)
_GENERIC_FIRST_PERSON_RE = re.compile(
    r"\b(?:we|our)\b\s+(?:are|'re|built|conducted|used|found|developed"
    r"|introduced|disclosed|tested|ran|report|reports|also|spent|sampled"
    r"|identified|also)\b"
    r"|\bOur\s+(?:researchers|prototype|system|model|team)\b"
)
_GENERIC_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _generic_number_word_value(text: str) -> Optional[int]:
    parts = text.lower().replace("-", " ").split()
    if len(parts) == 1:
        return _GENERIC_NUMBER_WORDS.get(parts[0])
    if len(parts) == 2:
        tens = _GENERIC_NUMBER_WORDS.get(parts[0])
        ones = _GENERIC_NUMBER_WORDS.get(parts[1])
        if tens is not None and ones is not None and tens >= 20 and ones < 10:
            return tens + ones
    return None


def _generic_is_version(sentence: str, start: int, end: int) -> bool:
    """True when a matched number is a version token, not a value.

    _is_version_number only catches a v/version prefix, which misses
    "Python 2.14.0" and "the version is 2.29.5". The fallback also
    treats a number followed by ".digit" as a version fragment, a
    number preceded by ".digit" as a version fragment, a dotted number
    right after a version-ish word ("ADK 2.14", "Python 2.14") as a
    version, and a number that is part of a model identifier
    ("GPT-6", "GPT-5.6" — a digit right after "-" or "." that follows
    a letter) as a version fragment. A dotted measurement ("48.5%",
    "2.8 million", "$0.30") is not a version: it carries a unit or a
    magnitude, and the word before it is not version-ish.
    """
    if _is_version_number(sentence, start):
        return True
    # v1.8.7: a model/benchmark version token ("Fable 5.2", "Opus
    # 5.5", "FrontierCode 2.0", "OSWorld 2.1", "CL-Bench 1.0") is an
    # identity, not a value — the version-ish word list above lacks
    # model/benchmark names, so delegate to the model-version check.
    if _is_model_version_number(sentence, start, end):
        return True
    if re.match(r"\.\d", sentence[end:]):
        return True
    if start >= 2 and sentence[start - 1] == "." and sentence[start - 2].isdigit():
        return True
    # Model identifier: a digit right after "-" or "." that follows a
    # letter ("GPT-6", "GPT-5.6", "v2.14" without the v caught above).
    if start >= 2 and sentence[start - 1] in "-." and sentence[start - 2].isalpha():
        return True
    if "." in sentence[start:end]:
        before = sentence[max(0, start - 12):start]
        if re.search(
                r"\b(?:python|release|version|v|build|sdk|api|adk|gpt|gemini"
                r"|claude|chrome|node|npm|pip|java|go|rust|app|os|ios|android"
                r"|linux|windows|macos)\s*$", before, re.I):
            return True
    return False


_GENERIC_CTX_UNIT_RE = re.compile(
    r"[- ](?:node|turn|run|test|task|workflow|user)s?\b", re.I)
_GENERIC_DURATION_RE = re.compile(
    r"\s*(?:hours?|minutes?|seconds?|days?|weeks?|months?|years?)\b", re.I)


def _generic_numbers_in(sentence: str) -> List[Dict]:
    """Claim-candidate numbers in one sentence: digit numbers and
    number words, with positions, values, units and currency.

    Excluded (they are context, not claims): version tokens, bare
    years, times, hyphenated/attached modifiers (30-node, 200-turn,
    "200 turns" — recorded separately as context numbers), durations
    ("within 12 hours", "after 10 days" — the ledger lists durations
    as not extracted), multipliers ("five times"), numbers followed by
    a letter (570s, 5x — except a magnitude letter after currency,
    $5B), and numbers inside URLs.
    """
    numbers: List[Dict] = []
    for m in _GENERIC_DIGIT_RE.finditer(sentence):
        currency, digits, suffix = m.group(1), m.group(2), (m.group(3) or "").strip()
        start, end = m.start(), m.end()
        if _generic_is_version(sentence, start, end):
            continue
        after = sentence[end:end + 1]
        before = sentence[start - 1:start] if start > 0 else ""
        if after == "-" or after == ":" or before == ":":
            # v1.8.7: "30-day retention" is a retention claim, not a
            # hyphenated context modifier ("30-node", "200-turn") —
            # the hyphen skip does not fire for an N-day retention
            # phrase.
            if after == "-" and re.match(
                    r"-\s*days?\s+(?:data\s+)?retention\b",
                    sentence[end:end + 25], re.I):
                pass
            else:
                continue
        # Range endpoint ("lines 0-98"): a number right after "-" that
        # follows a digit is the second half of a range, not a claim.
        if start >= 2 and sentence[start - 1] == "-" and sentence[start - 2].isdigit():
            continue
        # Attached context modifier ("30-node", "200 turns") or a
        # duration ("12 hours", "10 days") — context, not a claim.
        if _GENERIC_CTX_UNIT_RE.match(sentence[end:]):
            continue
        # v1.8.7: a duration mints only under a completion verb
        # ("took 30 hours", "completed ... in 18 hours", "took 11
        # days"); "within 12 hours", "after 10 days", "for eight
        # days" stay skipped.
        value = float(digits.replace(",", ""))
        dur_m = _GENERIC_DURATION_RE.match(sentence[end:])
        if dur_m:
            # The completion verb may sit well before the duration in
            # a long sentence ("formalized Fermat's Last Theorem in
            # the Lean proof assistant in 30 hours") — check the whole
            # sentence before the number, not a 40-char window.
            before = sentence[:start]
            if not _GENERIC_COMPLETION_VERB_RE.search(before):
                continue
            numbers.append({
                "start": start, "end": end + len(dur_m.group(0)),
                "value": value, "raw": sentence[start:end + len(dur_m.group(0))].strip(),
                "unit": dur_m.group(0).strip().lower(), "currency": currency,
                "duration": True,
            })
            continue
        # v1.8.7: a month-day date ("April 7", "November 6", "Nov. 9",
        # "7 April") is a date, never a claim value.
        if _GENERIC_MONTH_RE.search(sentence[max(0, start - 15):start]) or \
                _GENERIC_MONTH_AFTER_RE.match(sentence[end:end + 12]):
            continue
        # v1.8.7: a denominator number ("41 of 57 facts" — the 57; "41
        # of the 57 facts") is part of the numerator's denominator,
        # not a claim of its own. The numerator mints with the
        # denominator recorded; the denominator never mints.
        if re.search(r"\d+\s+of\s+(?:the\s+)?$", sentence[max(0, start - 20):start]):
            continue
        # v1.8.7: a question count in a sentence that also carries a
        # percentage is the set size, not a claim ("Across 25
        # evaluation questions, wiki-only performance increased from
        # 13% to 48%" — the 25 is the evaluation set; "On 24 merger
        # questions, Claude scored 87%..." — the 24 is the merger
        # set). Two different sets' sizes (24 vs 25) are not a
        # disagreement, and the percentages carry the claim.
        if not suffix and "%" in sentence:
            after_words = re.findall(r"[A-Za-z]+", sentence[end:end + 40])[:3]
            if "questions" in [w.lower() for w in after_words]:
                continue
        if re.search(r"https?://\S*$", sentence[:start]):
            continue
        if suffix == "%":
            unit = "percent"
        elif suffix:
            value *= {"million": 1e6, "billion": 1e9, "trillion": 1e12,
                      "thousand": 1e3}[suffix.lower()]
            unit = "count"
        elif currency:
            unit = currency
        else:
            unit = "count"
        if not currency and not suffix and re.fullmatch(r"\d{4}", digits) \
                and 1900 <= int(digits) <= 2099:
            continue
        # v1.8.7: a size unit ("283 KB", "25 KB") is a size claim, not
        # a bare count — it bypasses metric cues (family "size").
        size_m = _GENERIC_SIZE_UNIT_RE.match(sentence[end:])
        if size_m and not currency and not suffix:
            numbers.append({
                "start": start, "end": end + len(size_m.group(0)),
                "value": value, "raw": sentence[start:end + len(size_m.group(0))].strip(),
                "unit": size_m.group(1).upper(), "currency": currency,
                "size": True,
            })
            continue
        if after.isalpha():
            if currency and after in "BMK":
                value *= {"B": 1e9, "M": 1e6, "K": 1e3}[after]
                end += 1
            elif not currency and after in "BMK":
                # v1.8.7: a magnitude letter on a non-currency number
                # followed by a count noun ("420M tokens", "2B
                # parameters") is a magnitude, not a version fragment.
                # The next word must be a count noun, not a preposition
                # or verb ("420M in revenue" is not a count).
                nxt = re.match(r"\s+([A-Za-z]+)", sentence[end + 1:])
                if nxt and nxt.group(1).lower() not in (
                        "per", "in", "on", "at", "for", "of", "and", "or",
                        "to", "is", "was", "are", "were", "with", "by",
                        "from", "as", "than"):
                    value *= {"B": 1e9, "M": 1e6, "K": 1e3}[after]
                    end += 1
                else:
                    continue
            else:
                continue
        if re.match(r"\s*(?:times|fold)\b", sentence[end:], re.I):
            continue
        numbers.append({
            "start": start, "end": end, "value": value,
            "raw": sentence[start:end].strip(), "unit": unit, "currency": currency,
        })
    for m in _GENERIC_NUMBER_WORD_RE.finditer(sentence):
        value = _generic_number_word_value(m.group(1))
        if value is None:
            continue
        start, end = m.start(), m.end()
        # A hyphenated fraction ("three-quarters", "two-thirds") is a
        # proportion, not a count of three — skip.
        if re.match(r"-(?:quarters?|thirds?|halves|half)\b", sentence[end:], re.I):
            continue
        if re.match(r"\s*(?:times|fold)\b", sentence[end:], re.I):
            continue
        numbers.append({
            "start": start, "end": end, "value": float(value),
            "raw": m.group(1), "unit": "count", "currency": "",
        })
    numbers.sort(key=lambda n: n["start"])
    return numbers


def _generic_product_base(token: str) -> str:
    return token.split()[0].split("-")[0].split(".")[0].lower()


# Subject stoplists (v1.8.7): a benchmark base ("mmlu", "gpqa",
# "hle", "frontiercode", "osworld", "cl-bench"), a size unit ("mb",
# "kb", "gb", "tb"), or a feed-header label ("coverage", "primary",
# "secondary") is never a measured subject — "coverage" came from a
# "COVERAGE — alfw.ai" header line, "mb"/"kb" from size units, and
# "cl-bench" from a benchmark name. Model family words are valid
# subjects on their own ("Fable", "Opus").
_GENERIC_SUBJECT_STOPLIST = {
    "mmlu", "gpqa", "hle", "humaneval", "frontiercode", "osworld",
    "cl", "cl-bench", "aa", "swe", "swe-bench", "math", "aime", "gsm8k",
    "mb", "kb", "gb", "tb", "coverage", "primary", "secondary",
    "tertiary", "benchmark", "score", "index", "lab",
}


def _generic_subject_ok(display: str) -> bool:
    """True when a candidate subject token is a measured system, not
    a benchmark base, a size unit, or a header label."""
    base = display.split()[0].split("-")[0].split(".")[0].lower()
    return base not in _GENERIC_SUBJECT_STOPLIST


def _generic_subject(sentence: str, num: Dict, numbers: List[Dict]) -> Tuple[str, str]:
    """The measured subject of one number: the nearest model token in
    the segment since the previous number (a model token — "Fable
    5.2", "Opus 5.5" — outranks a product token), else the nearest
    product token, else a bare capitalized model-family word
    ("Fable", "Opus"), else the first product token after the number
    introduced by "for"/"by" before the next number. Returns (slug,
    display), ("", "") when the sentence names no measured system for
    this number — the claim is then subject-less and the merge pass in
    compare_documents attaches it to the one stated-subject group with
    the same identity, or leaves it standing on its own."""
    prev_end = 0
    for n in numbers:
        if n["end"] <= num["start"] and n["end"] > prev_end:
            prev_end = n["end"]
    next_start = len(sentence)
    for n in numbers:
        if n["start"] >= num["end"] and n["start"] < next_start:
            next_start = n["start"]
    seg_before = sentence[prev_end:num["start"]]
    seg_after = sentence[num["end"]:next_start]

    def _slug(display: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", display.lower()).strip("-")

    # Model tokens first: "Fable 5.2 49.1%" is Fable 5.2's score.
    model_before = list(_MODEL_TOKEN_RE.finditer(seg_before))
    if model_before:
        display = model_before[-1].group(1)
        return _model_slug(display), display
    tokens_before = [
        m for m in _GENERIC_PRODUCT_RE.finditer(seg_before)
        if _generic_product_base(m.group(1)) not in _GENERIC_PRODUCT_BASE_STOPLIST
        and _generic_subject_ok(m.group(1))
    ]
    if tokens_before:
        display = tokens_before[-1].group(1)
        return _slug(display), display
    # Bare capitalized model-family word ("Fable", "Opus", "Mythos").
    fam_before = [m for m in re.finditer(r"\b([A-Z][a-z]+)\b", seg_before)
                  if m.group(1).lower() in _MODEL_FAMILY_WORDS]
    if fam_before:
        display = fam_before[-1].group(1)
        return _slug(display), display
    for m in _MODEL_TOKEN_RE.finditer(seg_after):
        before = seg_after[max(0, m.start() - 12):m.start()]
        if re.search(r"\b(?:for|by)\s*$", before):
            display = m.group(1)
            return _model_slug(display), display
    for m in _GENERIC_PRODUCT_RE.finditer(seg_after):
        if _generic_product_base(m.group(1)) in _GENERIC_PRODUCT_BASE_STOPLIST:
            continue
        if not _generic_subject_ok(m.group(1)):
            continue
        before = seg_after[max(0, m.start() - 12):m.start()]
        if re.search(r"\b(?:for|by)\s*$", before):
            display = m.group(1)
            return _slug(display), display
    return ("", "")


def _generic_metric(sentences: List[str], sent_idx: int, num: Dict,
                    numbers: List[Dict], _seen: frozenset = frozenset()) -> Optional[str]:
    """Metric family for one number, or None (metric-less numbers are
    skipped — the fallback is bounded, not a number vacuum).

    Region 1: the nearest cue within 20 chars — a cue before the
    number counts double (the metric usually follows the number:
    "16 percentage points", "81 findings", "9% ... memory"), and a
    cue belongs to its NEAREST number: a cue that sits between two
    numbers is claimed by the nearer one ("23% faster with 9%" — the
    "faster" belongs to the 23%, not the 9%). A cue whose span wraps
    the number ("reported 304 ... to maintainers") counts as adjacent.
    On a near-tie (within 2) the before-cue wins only when its family
    is verb-driven ("sampled 166 exploits" is a sample count;
    "achieved 40.3% recall" is a recall, not a generic score).
    Region 2: a comparison marker ("compared with", "up from",
    "versus") between the nearest preceding number and this one hands
    this number the preceding number's metric — a baseline inherits
    its metric — except derived families (a percentage-point gap, a
    relative improvement) never hand their metric down: the walk
    continues to the value number behind them.
    Region 3: the nearest cue in the same clause within 60 chars,
    then the whole sentence (a "modules" cue two clauses over still
    belongs to a bare "26 auth" when no closer cue exists).
    Region 4: a bare count sentence ("produced 166.") inherits the
    previous sentence's most recently mentioned figure's metric, when
    a production verb precedes the number in the same clause.
    A "refrain" in the sentence reframes "exploit" as avoidance
    ("refrained from exploiting them in 99.93%") — the plain exploit
    cue is excluded there, so the figure is an avoidance rate, not an
    exploit count.
    """
    sentence = sentences[sent_idx]
    refrain = bool(re.search(r"\brefrain\w*\b", sentence))
    # v1.8.7 Region 0 (fraction): a numerator followed by "of <number>
    # <count-noun>" takes that noun's family ("41 of 57 facts" — the
    # 41 is a facts count; the "facts" cue sits after the 57, so the
    # ordinary cue logic would hand it to the 57 instead).
    frac_m = re.match(r"\s*of\s+\d+(?:\.\d+)?\s+([A-Za-z]+)",
                      sentence[num["end"]:num["end"] + 30])
    if frac_m:
        frac_word = frac_m.group(1).lower().rstrip("s")
        frac_family = {
            "fact": "facts", "token": "tokens", "block": "blocks",
            "agent": "agents", "subagent": "agents", "session": "sessions",
            "task": "tasks", "question": "tasks", "criteria": "score",
            "criterion": "score", "test": "tasks", "finding": "findings",
            "vulnerabilitie": "vulnerabilities", "exploit": "exploits",
            "patche": "patches", "patch": "patches",
            "component": "components", "module": "modules",
            "feature": "features",
        }.get(frac_word)
        if frac_family:
            return frac_family
    # v1.8.7 Region 0 (CVE count): a number immediately followed by
    # a CVE token or a vulnerabilities noun names a vulnerability
    # count — "1,829 previously disclosed vulnerabilities" is a CVE
    # count, not a "findings disclosed" count. The "disclosed" cue
    # sits closer to the number than "vulnerabilities" in that
    # phrasing, so without this rule the same 1,829 mints under two
    # families in two docs and never aligns.
    if re.match(r"\s+(?:previously\s+)?(?:disclosed\s+)?(?:CVEs?|vulnerabilit\w*)",
                sentence[num["end"]:num["end"] + 40]):
        return "vulnerabilities"
    # v1.8.7 decimal guard: a non-integer bare decimal (0.838, 61.82)
    # is a measurement, never a count of a count-noun family — a
    # version fragment or a score must not read as a task count.
    # "sessions" is count-only: a percentage never takes it.
    raw = num.get("raw", "")
    is_bare_decimal = (num.get("unit") == "count" and "." in str(raw)
                       and float(num["value"]) != int(num["value"]))
    is_percent = num.get("unit") == "percent"
    cues = []
    for rx, family in _GENERIC_METRIC_CUES:
        if refrain and family == "exploits":
            continue
        if is_bare_decimal and family in _GENERIC_COUNT_NOUN_FAMILIES:
            continue
        if is_percent and family == "sessions":
            continue
        for m in rx.finditer(sentence):
            cues.append((m.start(), m.end(), family))
    if not cues:
        return None

    def _edge_dist(cs: int, ce: int, n: Dict) -> int:
        if ce <= n["start"]:
            return n["start"] - ce
        if cs >= n["end"]:
            return cs - n["end"]
        return 0

    def _claimed_by_other(cs: int, ce: int) -> bool:
        """True when another number in this sentence is strictly
        nearer to the cue than the number being scored — the cue
        belongs to that number, not this one. (\"23% faster with 9%\":
        \"faster\" belongs to the 23%; \"completed 95%\": \"completed\"
        belongs to the 95% it directly precedes.) A before-cue in a
        different clause than the number belongs to that clause's
        number ("43 had staging patches, and 299 were being
        reported" — "patches" is the 43's, not the 299's)."""
        if ce <= num["start"] and not (clause_a <= cs < clause_b):
            return True
        mine = _edge_dist(cs, ce, num)
        for n in numbers:
            if n is num:
                continue
            if _edge_dist(cs, ce, n) < mine:
                return True
        return False

    def _between_has_number(a: int, b: int) -> bool:
        return any(n["start"] >= a and n["end"] <= b for n in numbers
                   if n is not num)

    clause_a, clause_b = _generic_clause_span(sentence, num["start"], num["end"])

    # Region 1: the nearest cue within 20 chars, in the number's own
    # clause first (the clause scope uses a wider limit, 60: the clause
    # bounds the search, so a cue 29 chars out in the number's own
    # clause ("81 novel high-severity Firefox issues") is still that
    # number's). A cue in another clause — \"candidate patches;
    # 81 novel issues\" — belongs to its own clause's number, not to
    # a number across the semicolon), then the whole sentence.
    for scope_a, scope_b, limit in ((clause_a, clause_b, 60), (0, len(sentence), 20)):
        best_before = None  # (effective_dist, family)
        best_after = None
        for cs, ce, family in cues:
            if not (scope_a <= cs < scope_b):
                continue
            if _claimed_by_other(cs, ce):
                continue
            if ce <= num["start"]:
                if _between_has_number(ce, num["start"]):
                    continue
                dist = (num["start"] - ce) * 2
                if dist <= limit and (best_before is None or dist < best_before[0]):
                    best_before = (dist, family)
            elif cs >= num["end"]:
                dist = cs - num["end"]
                if dist <= limit and (best_after is None or dist < best_after[0]):
                    best_after = (dist, family)
            else:
                # The cue's span wraps the number ("reported 304 ... to
                # maintainers") — adjacent, dist 0.
                if best_before is None or 0 < best_before[0]:
                    best_before = (0, family)
        if best_before and best_after:
            if abs(best_before[0] - best_after[0]) <= 2:
                if best_before[1] in _GENERIC_VERB_DRIVEN_FAMILIES:
                    return best_before[1]
                return best_after[1] if best_after[0] < best_before[0] else best_before[1]
            return best_before[1] if best_before[0] < best_after[0] else best_after[1]
        if best_before:
            return best_before[1]
        if best_after:
            return best_after[1]
    # Region 2: baseline inheritance, skipping derived families.
    # v1.8.7 pair inheritance: a "from X to Y" range ("13% to 48%",
    # "54% to 61%") and an "A and B" pair ("0.655 and 0.838",
    # "87% vs 73%") hand the metric to the second number — the two
    # numbers are one measurement, not two unrelated figures. The
    # separator between them must be ONLY the separator word ("and",
    # "to", "vs", ","); anything else is a different construction.
    # "to" counts only when the previous number is introduced by
    # "from" ("from 13% to 48%") — a bare "to" elsewhere ("scored
    # 49.1% to 32.4%") is not a range marker.
    prev = None
    for n in numbers:
        if n["end"] <= num["start"] and (prev is None or n["end"] > prev["end"]):
            prev = n
    while prev is not None:
        between = sentence[prev["end"]:num["start"]]
        between_stripped = between.strip().lower().rstrip(".")
        is_pair = between_stripped in ("and", "to", "vs", ",")
        if is_pair and between_stripped == "to":
            before_prev = sentence[max(0, prev["start"] - 12):prev["start"]]
            is_pair = bool(re.search(r"\bfrom\s*$", before_prev, re.I))
        if _GENERIC_COMPARISON_MARKER_RE.search(between) or is_pair:
            if (sent_idx, prev["start"]) in _seen:
                break
            family = _generic_metric(sentences, sent_idx, prev, numbers,
                                     _seen | {(sent_idx, num["start"])})
            if family and family not in _GENERIC_DERIVED_FAMILIES:
                return family
            if family:
                # Derived family: walk back to the value number behind it.
                earlier = None
                for n in numbers:
                    if n["end"] <= prev["start"] and (earlier is None or n["end"] > earlier["end"]):
                        earlier = n
                prev = earlier
                continue
        break
    # Region 3: same clause first, then the whole sentence. Claiming
    # applies to BEFORE-cues only (a \"vulnerabilities\" cue that
    # belongs to a 1,724 must not leak into a neighbouring 74.7%).
    # An AFTER-cue before the next number is the scored number's own
    # list noun (\"26 auth, ..., cache modules, 16 ...\" — \"modules\"
    # is the 26's even though the 16 sits nearer), so claiming does
    # not apply to it.
    clause_a, clause_b = _generic_clause_span(sentence, num["start"], num["end"])
    next_start = len(sentence)
    for n in numbers:
        if n["start"] >= num["end"] and n["start"] < next_start:
            next_start = n["start"]
    for scope_a, scope_b in ((clause_a, clause_b), (0, len(sentence))):
        best = None
        for cs, ce, family in cues:
            if not (scope_a <= cs < scope_b):
                continue
            if ce <= num["start"]:
                if _claimed_by_other(cs, ce):
                    continue
                if _between_has_number(ce, num["start"]):
                    continue
                dist = num["start"] - ce
            elif cs >= num["end"] and cs < next_start:
                dist = cs - num["end"]
            elif cs < num["start"] and ce > num["end"]:
                # The cue's span genuinely wraps the number — adjacent.
                if _claimed_by_other(cs, ce):
                    continue
                dist = 0
            else:
                continue
            if dist <= 60 and (best is None or dist < best[0]):
                best = (dist, family)
        if best:
            return best[1]
    # v1.8.7 backward pair inheritance: when the number has no metric
    # of its own, a following number in the same "A and B" / "A to B"
    # pair hands its metric BACK ("0.655 and 0.838 mean rewards" —
    # the cue sits after the pair, nearer the 0.838, but both numbers
    # are the same measurement). Recursion guard: never resolve a
    # number already on the resolution path.
    next_num = None
    for n in numbers:
        if n["start"] >= num["end"] and (next_num is None or n["start"] < next_num["start"]):
            next_num = n
    if next_num is not None and (sent_idx, next_num["start"]) not in _seen:
        between = sentence[num["end"]:next_num["start"]]
        if between.strip().lower().rstrip(".") in ("and", "to", "vs", ","):
            family = _generic_metric(sentences, sent_idx, next_num, numbers,
                                     _seen | {(sent_idx, num["start"])})
            if family and family not in _GENERIC_DERIVED_FAMILIES:
                return family
    # Region 4: a production verb precedes the number in the same
    # clause — the bare count inherits the previous sentence's most
    # recently mentioned figure's metric.
    if sent_idx > 0:
        before_in_clause = sentence[clause_a:num["start"]]
        if _GENERIC_PRODUCTION_VERB_RE.search(before_in_clause[-25:]):
            prev_numbers = _generic_numbers_in(sentences[sent_idx - 1])
            if prev_numbers:
                return _generic_metric(sentences, sent_idx - 1, prev_numbers[-1], prev_numbers,
                                      _seen | {(sent_idx, num["start"])})
    return None


def _generic_clause_span(sentence: str, start: int, end: int) -> Tuple[int, int]:
    """The (start, end) span of the comma/and clause containing a position."""
    parts = []
    last = 0
    for m in _GENERIC_CLAUSE_SPLIT_RE.finditer(sentence):
        parts.append((last, m.start()))
        last = m.end()
    parts.append((last, len(sentence)))
    for a, b in parts:
        if a <= start < b:
            return (a, b)
    return (0, len(sentence))


def _generic_clause(sentence: str, start: int, end: int) -> str:
    """The comma/and clause containing a position (cue scoping)."""
    a, b = _generic_clause_span(sentence, start, end)
    return sentence[a:b]


def _generic_context_numbers(sentence: str, num: Dict) -> Tuple[str, str]:
    """Hyphenated/attached context numbers in the claim's clause
    (30-node, 200-turn): part of the identity, so a 30-node/200-turn
    run is not the same claim as a bare run. Returns (key part,
    display)."""
    clause = _generic_clause(sentence, num["start"], num["end"])
    found: Dict[str, str] = {}
    for m in _GENERIC_CTX_NUM_RE.finditer(clause):
        unit = m.group(2).lower()
        plural = unit if unit.endswith("s") else unit + "s"
        found[plural] = m.group(1)
    key = ";".join(f"{k}={found[k]}" for k in sorted(found))
    display = "; ".join(f"{found[k]}-{k[:-1]}" for k in sorted(found))
    return key, display


def _generic_denominator(sentence: str, num_end: int,
                         numbers: List[Dict] = None) -> str:
    """The denominator after a number ("of tests", "per LPE exploit"),
    from the ~40 chars after it. The window stops at the next number:
    a "per" phrase that belongs to a later amount in the same
    sentence ("$1,213,430 ... $1,279 per LPE exploit") must not be
    read as this amount's denominator. Recorded on every claim; only
    a per-unit denominator on a cost claim enters the identity key."""
    # v1.8.7 fraction rule: "41 of 57 facts" mints the numerator with
    # denominator "of 57 facts" — the denominator is the claim's
    # context, and the compare loop's value-set handling keeps the
    # 41s and the 57s as separate aligned groups. The fraction check
    # runs on the untruncated window: the next number IS the
    # denominator's number, so truncating at it would cut the match.
    wide = sentence[num_end:num_end + 45]
    fm = re.match(r"\s*of\s+(\d+(?:\.\d+)?\s+[A-Za-z][\w\-]{0,20})", wide)
    if fm:
        return f"of {fm.group(1).strip()}"
    limit = num_end + 45
    if numbers:
        for n in numbers:
            if n["start"] >= num_end and n["start"] < limit:
                limit = n["start"]
                break
    window = sentence[num_end:limit]
    m = _GENERIC_DENOM_RE.search(window)
    return m.group(0).strip() if m else ""


def _generic_qualifier(sentence: str, num_start: int) -> str:
    """The qualifier nearest before a number ("about", "up to", "~"),
    from the ~30 chars before it. Recorded, never silently dropped:
    an "up to" ceiling and an "about" approximation change what a
    value means when compared."""
    window = sentence[max(0, num_start - 30):num_start]
    found = list(_GENERIC_QUALIFIER_RE.finditer(window))
    if not found:
        return ""
    q = found[-1].group(0).lower().strip()
    return "approximately" if q == "~" else q


def _generic_benchmark(sentence: str, num_start: int = 0) -> str:
    """The CamelCase token nearest the number, excluding org names —
    the benchmark the claim belongs to, when the sentence names one.
    Recorded on the claim and (v1.8.7) part of the identity: two
    scores on different benchmarks are different claims, even for the
    same model. A benchmark named in one doc and unnamed in another
    is the same claim — the compare loop's benchmark merge attaches
    the unnamed side when the values match. The nearest token wins,
    not the first in the sentence: in a long sentence that mentions
    several benchmarks, the first one is arbitrary (\"about 55% fewer
    blocks ... its ProgramBench result\" — the 55% is not a
    ProgramBench score)."""
    best = ""
    best_dist = 10 ** 9
    for m in _GENERIC_CAMEL_RE.finditer(sentence):
        if m.group(0).lower() in _GENERIC_BENCH_ORG_STOPLIST:
            continue
        dist = min(abs(m.start() - num_start), abs(m.end() - num_start))
        if dist < best_dist:
            best_dist = dist
            best = m.group(0)
    return best if best_dist <= 80 else ""


def _generic_header_org(text: str) -> str:
    """The pipeline feed header's org ("PRIMARY SOURCE — OpenAI"),
    first word, lowercased. The header is the caller's own label for
    who the document is from — the doc-level attribution fallback for
    first-person docs."""
    first_line = text.split("\n", 1)[0]
    m = _GENERIC_HEADER_ORG_RE.match(first_line)
    return m.group(1).lower() if m else ""


def _generic_attribution(sentence: str, doc_attribution: str) -> str:
    """Who the sentence says asserted the figure: a sentence-level
    asserter verb ("OpenAI says 48.5%", "according to OpenAI"), else
    the doc-level attribution (first-person voice + header org), else
    "" — unstated, never guessed. "Deemed"/"classified" deliberately
    excluded: they attribute a classification, not the number."""
    m = _GENERIC_ASSERT_RE.search(sentence)
    if m:
        return m.group(1).lower()
    m = _GENERIC_ACCORDING_RE.search(sentence)
    if m:
        return m.group(1).lower()
    return doc_attribution


def _generic_role(num: Dict) -> str:
    """The claim's role (v1.8.7, Arch gate point 1): what KIND of
    number this is, part of the identity key so two different kinds
    of number never lump. A percentage is a percentage; a currency
    amount is a cost; a size (KB/MB) is a size; a duration is a
    duration; a non-integer bare decimal is a measurement; anything
    else is a task count. (The run-10 false conflict was a
    task_count 128 "cost" against a size 283 "cost" — the role
    separation is what stops that.)"""
    if num.get("unit") == "percent":
        return "percentage"
    if num.get("currency"):
        return "cost"
    if num.get("size"):
        return "size"
    if num.get("duration"):
        return "duration"
    raw = str(num.get("raw", ""))
    if "." in raw and float(num["value"]) != int(num["value"]):
        return "measurement"
    return "task_count"


def _generic_first_cue_family(sentence: str) -> str:
    """The family of the earliest metric cue in the sentence, or ""
    when the sentence names no metric. Used by the list rule: a
    number preceded by a model token in a sentence that names a
    metric is that metric's figure."""
    best = None
    for rx, family in _GENERIC_METRIC_CUES:
        m = rx.search(sentence)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), family)
    return best[1] if best else ""


def extract_generic_quantitative_claims(
    text: str,
    source_name: str = "",
    existing_claims: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Bounded generic quantitative-claim fallback (v1.8.6, P2; v1.8.7
    roles).

    Extracts numbers that carry a metric cue from ordinary sentences
    and returns them in the same shape as extract_numeric_claims, plus
    claim_kind "generic_quantitative" and the Arch fields: source span,
    entity, metric, value, unit, context/denominator, qualifier,
    variant, time, attribution. Dedupes against *existing_claims*
    (mapped extraction) by (sentence, value), so a mapped benchmark
    score is never double-counted. Returns [] for empty text.

    v1.8.7: the entity key carries the number's ROLE —
    generic|{role}|{family}|{ctx}|{denom}|{subject} — so a percentage,
    a cost, a size, a duration, a measurement and a task count are
    different identities even when they share a metric family.
    """
    if not text or not text.strip():
        return []
    header_org = _generic_header_org(text)
    # v1.8.7: skip the first line when it is a feed header ("COVERAGE
    # — TechTimes", "PRIMARY SOURCE — Anthropic") — the header's own
    # words ("coverage") must never become a claim subject, and the
    # header line is a label, not a sentence.
    body = text
    first_line = text.split("\n", 1)[0]
    if _GENERIC_HEADER_ORG_RE.match(first_line):
        body = text.split("\n", 1)[1] if "\n" in text else ""
    sentences = [s for s in _GENERIC_SENT_SPLIT_RE.split(body) if s.strip()]
    first_person_hits = len(_GENERIC_FIRST_PERSON_RE.findall(text))
    # Doc-level attribution only for docs that speak in the first
    # person at least twice ("We're introducing", "Our researchers") —
    # a coverage doc's stray "we asked" must not re-attribute the
    # vendor's figures to the coverage publisher.
    doc_attribution = header_org if (header_org and first_person_hits >= 2) else ""
    existing_pairs = set()
    if existing_claims:
        for c in existing_claims:
            raw = str(c.get("raw_value", "")).strip().lstrip("$").rstrip("%")
            val = c.get("numeric_value")
            if val is None or not raw:
                continue
            for si, s in enumerate(sentences):
                if raw in s:
                    existing_pairs.add((si, float(val)))
    claims: List[Dict] = []
    for si, sentence in enumerate(sentences):
        numbers = _generic_numbers_in(sentence)
        prev_subject: Tuple[str, str] = ("", "")
        prev_num: Optional[Dict] = None
        for num in numbers:
            if (si, num["value"]) in existing_pairs:
                continue
            if num.get("currency"):
                clause = _generic_clause(sentence, num["start"], num["end"])
                family = "grants" if re.search(r"\bgrants?\b", clause, re.I) else "cost"
            elif num.get("size"):
                # A size ("283 KB", "25 KB") bypasses metric cues —
                # it is a size, whatever the sentence's nouns say.
                family = "size"
            elif num.get("duration"):
                # A duration under a completion verb ("took 30 hours").
                family = "duration"
            else:
                family = _generic_metric(sentences, si, num, numbers)
            if not family:
                # v1.8.7 list rule: a number preceded by a model token
                # within 20 chars, in a sentence that names a metric,
                # is that metric's figure ("Fable 5.2 49.1%" in prose
                # is a score; "Opus 5.5 0.838" is a score). The
                # sentence's first cue names the metric.
                before20 = sentence[max(0, num["start"] - 20):num["start"]]
                if _MODEL_TOKEN_RE.search(before20):
                    family = _generic_first_cue_family(sentence)
                if not family:
                    continue
            subject_slug, subject_display = _generic_subject(sentence, num, numbers)
            # v1.8.7 pair-subject inheritance: the second member of a
            # pair ("Opus 5.5 0.655 and 0.838", "Sonnet 4.6 0.330 and
            # 0.364") inherits the first member's subject — the model
            # token sits before the first number, not the second, and
            # without it the two models' scores collapse into one
            # unstated group and read as a conflict.
            if not subject_slug and prev_num is not None and prev_subject[0]:
                between = sentence[prev_num["end"]:num["start"]]
                if re.fullmatch(r"\s*(?:and|to|vs\.?|,)\s*", between):
                    subject_slug, subject_display = prev_subject
            ctx_key, ctx_display = _generic_context_numbers(sentence, num)
            denominator = _generic_denominator(sentence, num["end"], numbers)
            qualifier = _generic_qualifier(sentence, num["start"])
            benchmark = _generic_benchmark(sentence, num["start"])
            attribution = _generic_attribution(sentence, doc_attribution)
            # v1.8.7: the benchmark is part of the identity — two
            # scores on different benchmarks are different claims,
            # even for the same model (the run-11 table rows all share
            # the key "generic|percentage|score|||fable-5-2" and
            # collapse into one benchmark-mismatch group). The
            # benchmark slug rides in the ctx slot, ahead of any
            # context numbers.
            bench_slug = re.sub(r"[^a-z0-9]+", "-", benchmark.lower()).strip("-") if benchmark else ""
            ctx_parts = ([f"bench={bench_slug}"] if bench_slug else []) + ([ctx_key] if ctx_key else [])
            ctx_key = ";".join(ctx_parts)
            denom_key = (
                re.sub(r"[^a-z0-9]+", "-", denominator.lower()).strip("-")
                if family == "cost" and denominator.lower().startswith("per")
                else ""
            )
            role = _generic_role(num)
            label = _GENERIC_FAMILY_LABELS.get(family, family)
            if benchmark:
                label = f"{benchmark} {label}"
            if denominator and family == "cost":
                label = f"{label} ({denominator})"
            if subject_display:
                label = f"{label} — {subject_display}"
            year_m = _GENERIC_YEAR_RE.search(sentence)
            context = "; ".join(
                p for p in [ctx_display, denominator] if p
            )
            claims.append({
                "entity_key": f"generic|{role}|{family}|{ctx_key}|{denom_key}|{subject_slug or 'unstated'}",
                "entity_label": label,
                "numeric_value": float(num["value"]),
                "unit": num["unit"],
                "raw_value": num["raw"],
                "version": "",
                "shot_count": None,
                "variant": "",
                "claim_kind": "generic_quantitative",
                "subject": subject_slug,
                "subject_display": subject_display,
                "metric": family,
                "benchmark": benchmark,
                "context": context,
                "context_numbers": ctx_key,
                "denominator": denominator,
                "qualifier": qualifier,
                "time": year_m.group(0) if year_m else "",
                "attribution": attribution,
                "span": sentence.strip()[:200],
            })
            prev_subject = (subject_slug, subject_display)
            prev_num = num
    # v1.8.7 table-row generic claims: a list-item benchmark row whose
    # benchmark is NOT in the mapped entity map (FrontierCode, CVE
    # Rediscovery, OSWorld, Terminal-Bench-Science) still states model
    # scores — mint them as generic score claims with the benchmark
    # name and the model subject, so the headline numbers surface
    # instead of vanishing. Mapped benchmarks are handled by the
    # mapped table-row pass; this pass only covers the unmapped ones.
    for row in _parse_table_rows(text):
        if row["benchmark_key"]:
            continue
        bench = row["benchmark_name"]
        if not bench:
            continue
        subject = row["model_slug"]
        # The sentence loop may already have minted this score from
        # the same row (with the benchmark unnamed and the subject
        # unstated — "WANDR" is all-caps, not a CamelCase benchmark,
        # and the model token sits before the number). Upgrade that
        # claim's subject/benchmark instead of minting a duplicate.
        bench_slug = re.sub(r"[^a-z0-9]+", "-", bench.lower()).strip("-")
        upgraded = False
        for c in claims:
            if c["numeric_value"] == float(row["value"]) and \
                    c.get("span", "")[:200] == row["line"][:200]:
                c["subject"] = subject
                c["subject_display"] = row["model_display"]
                c["benchmark"] = bench
                c["entity_key"] = (
                    f"generic|percentage|score|bench={bench_slug}||{subject or 'unstated'}")
                c["entity_label"] = f"{bench} score — {row['model_display']}"
                upgraded = True
                break
        if upgraded:
            continue
        label = f"{bench} score — {row['model_display']}"
        claims.append({
            "entity_key": f"generic|percentage|score|bench={bench_slug}||{subject or 'unstated'}",
            "entity_label": label,
            "numeric_value": float(row["value"]),
            "unit": "percent",
            "raw_value": row["raw"],
            "version": "",
            "shot_count": None,
            "variant": "",
            "claim_kind": "generic_quantitative",
            "subject": subject,
            "subject_display": row["model_display"],
            "metric": "score",
            "benchmark": bench,
            "context": "",
            "context_numbers": "",
            "denominator": "",
            "qualifier": "",
            "time": "",
            "attribution": doc_attribution,
            "span": row["line"][:200],
        })
    return claims


def _generic_unique_entry(key: str, label: str, doc_name: str, claim: Dict,
                          variant: str = "", note: str = "") -> Dict:
    """A unique_claims entry for a generic quantitative claim, carrying
    the Arch fields (span, metric, benchmark, context, qualifier, time,
    attribution) so the caller can see what the figure is, not just
    that it is unique."""
    entry = {
        "entity_key":   key,
        "entity_label": label,
        "found_in":     doc_name,
        "value":        claim["numeric_value"],
        "unit":         claim["unit"],
        "raw":          claim["raw_value"],
        "variant":      variant,
        "claim_kind":   "generic_quantitative",
        "metric":       claim.get("metric", ""),
        "benchmark":    claim.get("benchmark", ""),
        "context":      claim.get("context", ""),
        "qualifier":    claim.get("qualifier", ""),
        "time":         claim.get("time", ""),
        "attribution":  claim.get("attribution", ""),
        "span":         claim.get("span", ""),
    }
    if note:
        entry["note"] = note
    return entry


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
          "benchmark_mismatches": [...],  # same generic identity, different named benchmarks (v1.8.6)
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
            "benchmark_mismatches": [],
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
        base_claims = extract_numeric_claims(doc.get("text", ""), dropped_out=dropped)
        # Generic quantitative fallback (v1.8.6, P2): numbers that carry
        # a metric cue in ordinary sentences — benchmark scores, counts,
        # percentages, costs — deduped against the mapped extraction
        # above by (sentence, value), so nothing is double-counted.
        base_claims += extract_generic_quantitative_claims(
            doc.get("text", ""), _doc_name(doc_idx), base_claims)
        doc_claims.append(base_claims)
        for entry in dropped:
            dropped_market_claims.append({**entry, "source": _doc_name(doc_idx)})
    # v1.8.7 extraction coverage (Arch gate point 4): per doc, the
    # numbers the text carries vs the claims minted — a recall
    # diagnostic, counted separately from conflicts/alignment. The
    # "seen" count is the digit numbers that are not 4-digit years
    # and not version tokens; it over-counts (a number that carries
    # no metric cue is not a claim), so coverage is a floor, not a
    # precision measure — and a low floor is a recall defect, never
    # permission to fabricate a conflict.
    extraction_coverage: List[Dict] = []
    for doc_idx, doc in enumerate(analyzed_docs):
        text = doc.get("text", "")
        seen = 0
        for m in _GENERIC_DIGIT_RE.finditer(text):
            digits = m.group(2)
            if re.fullmatch(r"\d{4}", digits) and 1900 <= int(digits) <= 2099:
                continue
            if _generic_is_version(text, m.start(), m.end()):
                continue
            seen += 1
        minted = len(doc_claims[doc_idx])
        extraction_coverage.append({
            "source":         _doc_name(doc_idx),
            "numbers_seen":   seen,
            "claims_minted":  minted,
            "coverage":       round(minted / seen, 3) if seen else 1.0,
        })
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

    # ── Generic-claim subject merge (v1.8.6, P2) ─────────────────────────
    # A subject-less generic claim attaches to the stated-subject group
    # with the same (family, context, denominator) identity — a doc
    # that states a figure without naming the measured system in the
    # claim's own segment is still stating that figure (\"the model
    # avoided exploiting out-of-scope services in 99.93% of tests\" is
    # GPT-6 Cyber's 99.93% when the primary names it). The VALUE
    # disambiguates when two subjects share a family (GPT-6 Cyber
    # 99.93% vs GPT-5.6 Cyber 94.49% avoidance; GPT-6 Cyber 40.3%
    # vs GPT-6 Astra 32.0% recall): an unstated entry attaches to the
    # one stated group whose values include a value within tolerance
    # of its own. Zero or two-plus matching stated groups: the entry
    # stands alone, never guessed. Moved entries are flagged
    # subject_attached so the attachment stays visible.
    def _generic_close(a: float, b: float) -> bool:
        if abs(a) < 20 and abs(b) < 20:
            return abs(a - b) <= 0.15
        ref = max(abs(a), abs(b)) or 1.0
        return abs(a - b) <= ref * 0.02

    for key in [k for k, es in entity_map.items()
                if es and es[0]["claim"].get("claim_kind") == "generic_quantitative"
                and k.endswith("|unstated")]:
        parts = key.split("|")
        if len(parts) != 6:
            continue
        prefix = "|".join(parts[:5])
        stated = {k: es for k, es in entity_map.items()
                  if k.startswith(prefix + "|") and not k.endswith("|unstated")
                  and es and es[0]["claim"].get("claim_kind") == "generic_quantitative"}
        if not stated:
            continue
        attached: List[Dict] = []
        for e in list(entity_map[key]):
            val = e["claim"].get("numeric_value")
            matches = [k for k, es in stated.items()
                       if any(_generic_close(x["claim"].get("numeric_value", 0), val)
                              for x in es)]
            if len(matches) == 1:
                target = matches[0]
                target_label = entity_map[target][0]["claim"]["entity_label"]
                e["claim"] = {**e["claim"], "entity_key": target,
                              "entity_label": target_label,
                              "subject_attached": True}
                entity_map[target].append(e)
                attached.append(e)
        if attached:
            entity_map[key] = [e for e in entity_map[key] if e not in attached]
            if not entity_map[key]:
                del entity_map[key]

    # ── Generic-claim benchmark merge (v1.8.7) ───────────────────────
    # A generic score that names no benchmark attaches to the one
    # stated-benchmark group with the same (role, family, denominator,
    # subject, non-benchmark context) identity whose values include a
    # value within tolerance — a benchmark named in one doc and unnamed
    # in another is the same claim. Zero or two-plus matching stated
    # groups: the entry stands alone, never guessed.
    def _ctx_bench(ctx: str) -> Tuple[str, str]:
        parts = [p for p in ctx.split(";") if p]
        bench = next((p[6:] for p in parts if p.startswith("bench=")), "")
        rest = ";".join(p for p in parts if not p.startswith("bench="))
        return bench, rest

    for key in [k for k, es in entity_map.items()
                if es and es[0]["claim"].get("claim_kind") == "generic_quantitative"]:
        parts = key.split("|")
        if len(parts) != 6:
            continue
        bench, rest = _ctx_bench(parts[3])
        if bench:
            continue
        # Identity without the benchmark: role, family, rest-ctx,
        # denominator. The subject must be COMPATIBLE (equal, or
        # unstated on either side) — a bench-less claim may also be
        # subject-less ("a 51% relative improvement"), and a benched
        # group may be subject-less while the bench-less side states
        # it. The value match below is the disambiguator, and exactly
        # one value-matching stated group is required. Ignoring the
        # subject entirely over-attaches: benchmark scores from
        # different models are numerically close (74.6 vs 75.3) and
        # would collapse into one group.
        ident = (parts[1], parts[2], rest, parts[4])
        # The key's subject part is the literal "unstated" when no
        # subject was extracted — treat that as empty, not as a
        # stated subject, or the compatibility check below never fires.
        subj = parts[5] if parts[5] != "unstated" else ""
        stated: Dict[str, List[Dict]] = {}
        for k2, es2 in entity_map.items():
            if k2 == key:
                continue
            p2 = k2.split("|")
            if len(p2) != 6:
                continue
            b2, r2 = _ctx_bench(p2[3])
            if not b2:
                continue
            if (p2[1], p2[2], r2, p2[4]) != ident:
                continue
            subj2 = p2[5] if p2[5] != "unstated" else ""
            if subj and subj2 and subj != subj2:
                continue
            stated.setdefault(k2, []).extend(es2)
        if not stated:
            continue
        attached: List[Dict] = []
        renamed: Dict[str, str] = {}
        for e in list(entity_map[key]):
            val = e["claim"].get("numeric_value")
            matches = [k for k, es in stated.items()
                       if any(_generic_close(x["claim"].get("numeric_value", 0), val)
                              for x in es)]
            if len(matches) == 1:
                target = renamed.get(matches[0], matches[0])
                if target not in entity_map:
                    continue
                target_label = entity_map[target][0]["claim"]["entity_label"]
                target_bench = entity_map[target][0]["claim"].get("benchmark", "")
                # Unify the subject too: the merged identity takes the
                # stated subject from either side (the bench-less side
                # may state it while the benched group does not, and
                # vice versa). The merged key carries that subject, so
                # the later (variant, subject) partition does not split
                # the group again.
                merged_subject = e["claim"].get("subject", "") or \
                    entity_map[target][0]["claim"].get("subject", "")
                tparts = target.split("|")
                merged_key = "|".join(tparts[:5] + [merged_subject or "unstated"])
                if merged_key != target:
                    for ge in entity_map[target]:
                        ge["claim"] = {**ge["claim"], "entity_key": merged_key,
                                       "subject": merged_subject}
                    entity_map[merged_key] = entity_map.pop(target)
                    renamed[matches[0]] = merged_key
                    target = merged_key
                e["claim"] = {**e["claim"], "entity_key": target,
                              "entity_label": target_label,
                              "subject": merged_subject,
                              "benchmark": target_bench,
                              "benchmark_attached": True}
                entity_map[target].append(e)
                attached.append(e)
        if attached:
            entity_map[key] = [e for e in entity_map[key] if e not in attached]
            if not entity_map[key]:
                del entity_map[key]

    # ── Mapped-claim subject merge (v1.8.7, Arch gate point 2) ─────────
    # A benchmark/pricing claim that states no subject attaches to the
    # one stated-subject group with the same (key, variant/era)
    # identity when its value matches that group's values within
    # tolerance — mirroring the generic merge above. Partitioning
    # benchmark groups by subject risks splitting a real alignment
    # when one doc states the subject and another does not (Nov 6
    # SWE-bench Pro 86.3%: no model token before the number in either
    # doc, so both sides are unstated and the merge is a no-op there).
    # Zero or two-plus matching stated groups: the entry stands alone.
    for key, entries in list(entity_map.items()):
        if not entries or entries[0]["claim"].get("claim_kind") == "generic_quantitative":
            continue
        if key not in _BENCHMARK_KEYS and not key.startswith("pricing"):
            continue
        unstated = [e for e in entries if not e["claim"].get("subject")]
        if not unstated:
            continue
        stated_subjects: Dict[str, List[Dict]] = {}
        for e in entries:
            subj = e["claim"].get("subject", "")
            if subj:
                stated_subjects.setdefault(subj, []).append(e)
        if not stated_subjects:
            continue
        for e in list(unstated):
            val = e["claim"].get("numeric_value")
            matches = [s for s, es in stated_subjects.items()
                       if any(_generic_close(x["claim"].get("numeric_value", 0), val)
                              for x in es)]
            if len(matches) == 1:
                e["claim"] = {**e["claim"], "subject": matches[0],
                              "subject_attached": True}

    shared_claims: List[Dict] = []
    unique_claims: List[Dict] = []
    conflicts:     List[Dict] = []
    # v1.8.7 fail-closed gate: claims that minted but could not be
    # typed for comparison (bare pricing) — span + reason each.
    unmatched_claims: List[Dict] = []
    alignment:     List[Dict] = []
    unit_mismatches: List[Dict] = []
    version_mismatches: List[Dict] = []
    evaluation_setting_mismatches: List[Dict] = []
    presence_absence_mismatches: List[Dict] = []
    variant_mismatches: List[Dict] = []
    # Benchmark mismatches (v1.8.6, P2): one generic identity reported
    # under two different NAMED benchmarks is not comparable at all —
    # neither a conflict nor an alignment.
    benchmark_mismatches: List[Dict] = []
    # Single-originator generic figures (v1.8.6, P2): orgs whose
    # figures are repeated across sources, for the triage summary.
    vendor_repeat_orgs: set = set()
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
    # v1.8.7 (Arch gate point 2): the partition is (variant, subject)
    # for benchmark keys and (era, subject) for pricing keys — two
    # models' scores on one benchmark, two models' per-task costs, and
    # a current price against a prior price are different qualified
    # identities, never a conflict. Everything else partitions by
    # variant alone, as before.
    grouped: List[Tuple[str, List[Dict], str]] = []
    for key, entries in entity_map.items():
        # v1.8.7 (Arch gate point 2): benchmark claims partition by
        # (variant, subject) and pricing claims by (era, subject) —
        # two models' scores, two models' per-task costs, and a
        # current vs a prior price are different qualified identities,
        # never a conflict. The variant splits ONLY when two or more
        # variants are stated: a stated variant against an unstated
        # one is a provisional comparison (v1.8.5), not a split.
        # Everything else partitions by variant alone, as before.
        stated_variants = {e["claim"].get("variant", "")
                           for e in entries if e["claim"].get("variant")}
        split_variant = len(stated_variants) > 1
        def _part_key(e):
            c = e["claim"]
            if key in _BENCHMARK_KEYS:
                v = c.get("variant", "") if split_variant else ""
                return (v, c.get("subject", ""))
            if key.startswith("pricing"):
                return (c.get("era", ""), c.get("subject", ""))
            return (c.get("variant", ""), "")
        partitions: Dict[Tuple[str, str], List[Dict]] = {}
        for e in entries:
            partitions.setdefault(_part_key(e), []).append(e)
        if split_variant:
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
        group_variant = next(iter(stated_variants), "") if not split_variant else ""
        if len(partitions) > 1:
            for (variant, _subject), part in partitions.items():
                grouped.append((key, part, variant if split_variant else group_variant))
        else:
            grouped.append((key, entries, group_variant))

    # A presence/absence mismatch is recorded once per key, even when
    # the key was split into several variant partitions.
    _presence_mismatch_done: set = set()

    for key, entries, variant in grouped:
        # ── Generic-claim group shaping (v1.8.6, P2) ─────────────────────
        is_generic = bool(entries) and entries[0]["claim"].get("claim_kind") == "generic_quantitative"
        if is_generic:
            # (b) Benchmark guard: two different NAMED benchmarks in
            # one group is a benchmark mismatch, not a comparison.
            # Compared case-insensitively: "FrontierCode" and
            # "frontiercode" are the same benchmark (the table-row pass
            # records the display name, the benchmark merge the slug).
            named_benches = {e["claim"].get("benchmark", "").lower()
                             for e in entries} - {""}
            if len(named_benches) > 1:
                benchmark_mismatches.append({
                    "entity_key":   key,
                    "entity_label": entries[0]["claim"]["entity_label"],
                    "benchmarks":   sorted({e["claim"].get("benchmark", "")
                                            for e in entries} - {""}),
                    "values":       [
                        {"source": _doc_name(e["doc_idx"]), "value": e["claim"]["numeric_value"],
                         "unit": e["claim"]["unit"], "raw": e["claim"]["raw_value"],
                         "benchmark": e["claim"].get("benchmark", "")}
                        for e in entries
                    ],
                    "note": (
                        "Same generic identity reported under different named benchmarks "
                        "across sources — figures from different benchmarks are not directly "
                        "comparable. Do not read as a conflict or an alignment."
                    ),
                })
                continue
            # (c) A group compares ACROSS documents, never within one:
            # a single-doc group emits one unique claim per distinct
            # value (a doc that states 61 patched and 43 in staging is
            # not disagreeing with itself).
            doc_ids = {e["doc_idx"] for e in entries}
            if len(doc_ids) == 1:
                if any(s == "EXPLICITLY_ABSENT" for s in presence_status[key].values()):
                    if key not in _presence_mismatch_done:
                        presence_absence_mismatches.append(
                            _mismatch_entry(key, entries[0]["claim"]["entity_label"],
                                            [_doc_name(e["doc_idx"]) for e in entries])
                        )
                        _presence_mismatch_done.add(key)
                    continue
                seen_vals: Dict[float, Dict] = {}
                for e in entries:
                    seen_vals.setdefault(e["claim"]["numeric_value"], e)
                for v, e in seen_vals.items():
                    unique_claims.append(_generic_unique_entry(
                        key, entries[0]["claim"]["entity_label"],
                        _doc_name(e["doc_idx"]), e["claim"], variant))
                continue
            # (c) Multi-doc groups compare VALUE SETS, not modals: one
            # identity can hold two distinct claims in the same document
            # (1,724 reproduced vulnerabilities AND 81 novel Firefox
            # vulnerabilities; 948 post-dedup AND 1,169 pre-dedup
            # exploits). Each value reported by 2+ docs is its own
            # aligned group; a value in one doc only splits out as a
            # unique. A doc's modal, when a single shared value is
            # compared, is the most frequent value ANOTHER doc also
            # reports (not merely its own most frequent value).
            by_value: Dict[float, List[Dict]] = {}
            for e in entries:
                by_value.setdefault(e["claim"]["numeric_value"], []).append(e)
            shared_vals = [v for v, es in by_value.items()
                           if len({e["doc_idx"] for e in es}) >= 2]
            if len(shared_vals) >= 2:
                for v in sorted(shared_vals):
                    ves = by_value[v]
                    vobjs = []
                    for e in ves:
                        vo = {"source": _doc_name(e["doc_idx"]),
                              "value": e["claim"]["numeric_value"],
                              "unit": e["claim"]["unit"],
                              "raw": e["claim"]["raw_value"],
                              "version": e["claim"].get("version", ""),
                              "shot_count": e["claim"].get("shot_count"),
                              "variant": e["claim"].get("variant", ""),
                              "span": e["claim"].get("span", ""),
                              "metric": e["claim"].get("metric", ""),
                              "benchmark": e["claim"].get("benchmark", ""),
                              "context": e["claim"].get("context", ""),
                              "qualifier": e["claim"].get("qualifier", ""),
                              "time": e["claim"].get("time", ""),
                              "attribution": e["claim"].get("attribution", "")}
                        vobjs.append(vo)
                    shared_claims.append({
                        "entity_key": key,
                        "entity_label": entries[0]["claim"]["entity_label"],
                        "found_in_docs": list(dict.fromkeys(
                            _doc_name(e["doc_idx"]) for e in ves)),
                        "values": vobjs,
                        "variant": variant,
                        "claim_kind": "generic_quantitative",
                    })
                    attr_orgs = sorted(
                        {e["claim"].get("attribution", "") for e in ves} - {""})
                    align_entry: Dict = {
                        "entity_key": key,
                        "entity_label": entries[0]["claim"]["entity_label"],
                        "spread": 0.0,
                        "values": vobjs,
                        "variant": variant,
                        "claim_kind": "generic_quantitative",
                        "attribution": attr_orgs,
                    }
                    if len(attr_orgs) == 1:
                        align_entry["attribution_note"] = (
                            f"All compared sources that name an attributor attribute this figure to "
                            f"{attr_orgs[0]} — the figure is repeated across sources, not "
                            "independently verified.")
                        vendor_repeat_orgs.add(attr_orgs[0])
                    alignment.append(align_entry)
                for v, es in by_value.items():
                    if v not in shared_vals:
                        for e in es:
                            unique_claims.append(_generic_unique_entry(
                                key, entries[0]["claim"]["entity_label"],
                                _doc_name(e["doc_idx"]), e["claim"], variant))
                continue
            by_doc: Dict[int, List[Dict]] = {}
            for e in entries:
                by_doc.setdefault(e["doc_idx"], []).append(e)
            cmp_entries: List[Dict] = []
            for doc_idx, doc_entries in by_doc.items():
                vals = [e["claim"]["numeric_value"] for e in doc_entries]
                if len(set(vals)) == 1:
                    cmp_entries.extend(doc_entries)
                    continue
                counts: Dict[float, int] = {}
                for v in vals:
                    counts[v] = counts.get(v, 0) + 1
                other_vals = {e["claim"]["numeric_value"] for e in entries
                              if e["doc_idx"] != doc_idx}
                ranked = sorted(counts, key=lambda v: -counts[v])
                modal = next((v for v in ranked if v in other_vals), ranked[0])
                cmp_entries.extend(
                    [e for e in doc_entries if e["claim"]["numeric_value"] == modal])
                for e in doc_entries:
                    if e["claim"]["numeric_value"] != modal:
                        unique_claims.append(_generic_unique_entry(
                            key, entries[0]["claim"]["entity_label"],
                            _doc_name(doc_idx), e["claim"], variant,
                            note=(
                                f"Also reported in {_doc_name(doc_idx)} alongside the compared "
                                f"value {modal}: a different count for this identity in the same "
                                "document, not compared."
                            )))
            entries = cmp_entries

        # ── Mapped-claim shaping (v1.8.7, Arch gate point 3) ─────────
        # A group compares ACROSS documents, never within one: one
        # document can never self-conflict. All entries from one doc →
        # one unique per distinct value (a benchmark table row is a
        # value set with different qualified identities, not a dispute).
        # A bare "pricing" claim (no resolvable tier) is fail-closed:
        # it is never compared — it lands as a unique with a note.
        if not is_generic:
            doc_ids = {e["doc_idx"] for e in entries}
            if len(doc_ids) == 1:
                # A lone mapped claim where another doc EXPLICITLY
                # lacks the entity is a presence/absence mismatch,
                # not a merely unique claim.
                if any(s == "EXPLICITLY_ABSENT" for s in presence_status[key].values()):
                    if key not in _presence_mismatch_done:
                        presence_absence_mismatches.append(
                            _mismatch_entry(key, entries[0]["claim"]["entity_label"],
                                            [_doc_name(e["doc_idx"]) for e in entries])
                        )
                        _presence_mismatch_done.add(key)
                    continue
                seen_vals: Dict[float, Dict] = {}
                for e in entries:
                    seen_vals.setdefault(e["claim"]["numeric_value"], e)
                for v, e in seen_vals.items():
                    unique_claims.append({
                        "entity_key":   key,
                        "entity_label": entries[0]["claim"]["entity_label"],
                        "found_in":     _doc_name(e["doc_idx"]),
                        "value":        v,
                        "unit":         e["claim"]["unit"],
                        "raw":          e["claim"]["raw_value"],
                        "variant":      variant,
                    })
                continue
            if key == "pricing":
                for e in entries:
                    unique_claims.append({
                        "entity_key":   key,
                        "entity_label": entries[0]["claim"]["entity_label"],
                        "found_in":     _doc_name(e["doc_idx"]),
                        "value":        e["claim"]["numeric_value"],
                        "unit":         e["claim"]["unit"],
                        "raw":          e["claim"]["raw_value"],
                        "variant":      variant,
                        "note": (
                            "Unresolved pricing tier (no input/output/cached/"
                            "per-task/fast/batch word could be assigned) — "
                            "fail-closed: never compared, listed for the "
                            "reader to resolve."
                        ),
                    })
                    # v1.8.7 fail-closed gate: the abstention is visible
                    # as its own list (span + reason), not only buried
                    # in unique_claims.
                    unmatched_claims.append({
                        "entity_key": key,
                        "source":     _doc_name(e["doc_idx"]),
                        "value":      e["claim"]["numeric_value"],
                        "span":       e["claim"].get("span", "")[:200],
                        "reason":     "unresolved pricing tier — no input/output/cached/per-task/fast/batch word assigned",
                    })
                continue

        label      = entries[0]["claim"]["entity_label"]
        doc_names  = [_doc_name(e["doc_idx"]) for e in entries]
        values     = [e["claim"]["numeric_value"] for e in entries]
        units      = [e["claim"]["unit"]          for e in entries]
        raws       = [e["claim"]["raw_value"]      for e in entries]

        val_objs = []
        for e in entries:
            vo = {"source": _doc_name(e["doc_idx"]), "value": e["claim"]["numeric_value"],
                  "unit": e["claim"]["unit"], "raw": e["claim"]["raw_value"],
                  "version": e["claim"].get("version", ""),
                  "shot_count": e["claim"].get("shot_count"),
                  "variant": e["claim"].get("variant", "")}
            if e["claim"].get("claim_kind") == "generic_quantitative":
                vo.update({
                    "span": e["claim"].get("span", ""),
                    "metric": e["claim"].get("metric", ""),
                    "benchmark": e["claim"].get("benchmark", ""),
                    "context": e["claim"].get("context", ""),
                    "qualifier": e["claim"].get("qualifier", ""),
                    "time": e["claim"].get("time", ""),
                    "attribution": e["claim"].get("attribution", ""),
                })
            val_objs.append(vo)

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
            if is_generic:
                unique_claims.append(_generic_unique_entry(
                    key, label, doc_names[0], entries[0]["claim"], variant))
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

        shared_entry = {
            "entity_key":    key,
            "entity_label":  label,
            "found_in_docs": list(dict.fromkeys(doc_names)),
            "values":        val_objs,
            "variant":       variant,
        }
        if is_generic:
            shared_entry["claim_kind"] = "generic_quantitative"
        shared_claims.append(shared_entry)

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

        # Qualifier rules (v1.8.6, P2): an "up to"/"at most" ceiling
        # makes lower values compatible, not a conflict; an approximate
        # qualifier ("about", "~") widens the tolerance for that pair.
        qualifier_note = ""
        if is_generic and is_conflict:
            quals = [e["claim"].get("qualifier", "") for e in entries]
            if any(q in _GENERIC_UPPER_BOUND_QUALS for q in quals):
                max_val = vals_sorted[-1]
                max_entries = [e for e in entries
                               if e["claim"]["numeric_value"] == max_val]
                if all(e["claim"].get("qualifier", "") in _GENERIC_UPPER_BOUND_QUALS
                       for e in max_entries):
                    is_conflict = False
                    qualifier_note = (
                        "Upper-bound qualifier ('up to'/'at most') on the highest value — "
                        "lower values are compatible with the ceiling, not a conflict."
                    )
            if is_conflict and any(q in _GENERIC_APPROX_QUALS for q in quals):
                reference = abs(vals_sorted[0]) if vals_sorted[0] != 0 else 1.0
                if spread <= max(reference * 0.10, 0.15):
                    is_conflict = False
                    qualifier_note = (
                        "Approximate qualifier ('about'/'~') on one side — values agree "
                        "within the widened tolerance for an approximate figure."
                    )

        # Generic-claim attribution (v1.8.6, P2): the orgs the compared
        # sentences actually name as asserters. When every asserter
        # names one org, the figure is repeated across sources, not
        # independently verified — say so on the entry, and once in the
        # triage summary. "Four sources repeated the vendor claim" must
        # never be phrased as "four independently verified it."
        attribution_note = ""
        attribution_orgs: List[str] = []
        if is_generic:
            attribution_orgs = sorted(
                {e["claim"].get("attribution", "") for e in entries} - {""})
            if len(attribution_orgs) == 1:
                attribution_note = (
                    f"All compared sources that name an attributor attribute this figure to "
                    f"{attribution_orgs[0]} — the figure is repeated across sources, not "
                    "independently verified."
                )
                vendor_repeat_orgs.add(attribution_orgs[0])

        entry = {
            "entity_key":   key,
            "entity_label": label,
            "spread":       round(spread, 5),
            "values":       val_objs,
            "variant":      variant,
        }
        if is_generic:
            entry["claim_kind"] = "generic_quantitative"
            entry["attribution"] = attribution_orgs
            if attribution_note:
                entry["attribution_note"] = attribution_note
            if qualifier_note:
                entry["qualifier_note"] = qualifier_note
        if key in _BENCHMARK_KEYS:
            entry["shot_count"] = next(iter(named_shots), None) if named_shots else None
            entry["evaluation_conditions"] = evaluation_note
        if variant_conditions:
            entry["variant_conditions"] = variant_conditions
            provisional_variant_comparisons.append(key)

        if is_conflict:
            # v1.8.7: the "different trading sessions" note belongs to
            # MARKET entities only (WTI/Brent, S&P, Nasdaq, Treasury).
            # A benchmark or generic discrepancy is not a market
            # session — the note stays neutral there.
            if key in _MARKET_ENTITY_KEYS or key in _CRUDE_KEYS:
                entry["note"] = (
                    f"Discrepancy of {spread:.3f} between sources. "
                    "Verify publication timestamps — values may reflect different trading sessions."
                )
            else:
                entry["note"] = (
                    f"Discrepancy of {spread:.3f} between sources. "
                    "Verify publication timestamps and measurement conditions — "
                    "values may reflect different evaluation settings or versions."
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
        # v1.8.6, P5: per-document roles for the date, so a caller can
        # tell a shared published date from a shared fetched header.
        # Year claims carry no role (the year pass has no label cues).
        roles = {
            _doc_name(e["doc_idx"]): sorted(e["claim"].get("date_roles", ["unknown"]))
            for e in entries
        }
        entry = {
            "date_key":  date_key,
            "kind":      kind,
            "found_in":  doc_names,
            "raw":       raws,
            "roles":     roles,
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
        | {m["entity_key"] for m in benchmark_mismatches}
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
    n_benchmark_mismatches = len(benchmark_mismatches)
    n_vendor_repeat = sum(
        1 for c in alignment + conflicts if c.get("attribution_note")
    )
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
    if n_benchmark_mismatches > 0:
        triage_summary += (
            f" {n_benchmark_mismatches} benchmark mismatch(es): same generic identity "
            "reported under different named benchmarks — not comparable, not a conflict."
        )
    if n_vendor_repeat > 0:
        triage_summary += (
            f" {n_vendor_repeat} shared generic claim(s) repeat figures attributed to a "
            f"single originator ({', '.join(sorted(vendor_repeat_orgs))}) — repetition "
            "across sources, not independent verification."
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
    if unmatched_claims:
        triage_summary += (
            f" {len(unmatched_claims)} claim(s) abstained: a price minted with no "
            "resolvable tier — fail-closed, never compared. See unmatched_claims "
            "for the spans and reasons."
        )
    low_coverage = [c for c in extraction_coverage if c["coverage"] < 0.5]
    if low_coverage:
        triage_summary += (
            f" Extraction coverage below half in {len(low_coverage)} document(s) "
            f"({', '.join(c['source'] for c in low_coverage)}): headline numeric "
            "claims may be unextracted — a recall defect, not evidence of "
            "agreement or disagreement. See extraction_coverage."
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
        "benchmark_mismatches": benchmark_mismatches,
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
        "unmatched_claims":  unmatched_claims,
        "extraction_coverage": extraction_coverage,
        "triage_summary":    triage_summary,
    }
