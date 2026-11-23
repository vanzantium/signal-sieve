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

# v1.8.8: benchmark names the CamelCase detector misses — "CVE
# Rediscovery" is all-caps + one hump, "Terminal-Bench-Science" is
# hyphenated, "Humanity's Last Exam" is title-case words. A phrase
# match here is what lets a prose percentage ("identified 89.6% of
# vulnerabilities in its internal CVE Rediscovery evaluation") read
# as a score on that benchmark instead of a count of the noun after
# the number. Only names that actually appear in the corpus with
# scores attached; list items with no scores (OSWorld, FrontierSWE)
# stay out.
_GENERIC_BENCHMARK_PHRASES = (
    "CVE Rediscovery",
    "Terminal-Bench-Science",
    # v1.8.9: the spaced form — the primary writes "Terminal-Bench
    # Science (70.5%)", not the all-hyphen form, so the paren rule
    # needs this spelling too.
    "Terminal-Bench Science",
    "Chartography",
    "Humanity's Last Exam",
    # v1.8.8 r2 (run-17 gap 4): hyphenated / all-caps / multi-word
    # benchmark and index names the CamelCase detector misses —
    # "Terminal-Bench" (pipe-row scores never keyed), "DeepSWE",
    # "OSWorld", "Last Secure Code Bench" (22%→31.8% across three
    # docs), "Artificial Analysis" (the AA Index).
    "Terminal-Bench",
    "DeepSWE",
    "OSWorld",
    "Last Secure Code Bench",
    "Artificial Analysis",
    # v1.8.8 r2 (run-17 gap 3, follow-on): "SWE-bench" — without it
    # a SWE-bench score is bench-less and the merge absorbs it into
    # the nearest benched group by value alone (76.2 landed on
    # DeepSWE 75.7, a false alignment; before that it landed on
    # Terminal-Bench, a false conflict).
    "SWE-bench",
    # "RoboDojo Sim" is half-detected ("RoboDojo" is CamelCase) — the
    # phrase is the fuller name the primary's row uses, so a medium
    # "RoboDojo Sim: Astra 29.8%" keys onto it instead of a bare
    # "RoboDojo" group that can never merge.
    "RoboDojo Sim",
    # v1.8.9 (run-20 gap 4): "SWE-sweep" — the harness-sensitive
    # benchmark whose 39.0% (Antigravity) vs 12.8% (shared mini-SWE
    # harness) pair never aligned because the name is invisible to
    # the CamelCase detector (all-caps + hyphen).
    "SWE-sweep",
    # v1.8.9 (run-20 gap 10): "Agent Arena" — the Arena board score
    # ("debuts #2 globally with a score of 1336 on Arena's new Agent
    # Arena Index") is bench-less without it and pools with any other
    # bench-less score in the doc.
    "Agent Arena",
    # v1.8.9 (run-20 gap 14): "Vals Index" — the Vals ranked scores
    # ("Neon Extra High second at 70.87% on Vals Index v2.2") never
    # key without it, so Neon's 70.87 never mints at all (no metric
    # cue in its clause) while Fable/Sonnet/Astra mint bench-less —
    # and the stated ±0.94pp uncertainty has no benchmark group to
    # attach to for the separability check.
    "Vals Index",
    # v1.8.9 (run-20 gap 12): "Agents' Last Exam" — the Kingy
    # comparison rows ("Agents' Last Exam v1.1: 75.8% vs 71.6% vs
    # 73.0%.") never key without it, so the 75.8% minted as a bare
    # "agents" figure and pooled with the 99.2% pass rate.
    "Agents' Last Exam",
    # v1.8.9 (run-20 gap 14): "Coding Agent Index" — Artificial
    # Analysis runs TWO indexes (Intelligence Index and Coding Agent
    # Index) and the org name alone cannot tell them apart. Without
    # the index name, a sentence naming "Artificial Analysis's
    # separate Coding Agent Index" reads as naming exactly one
    # benchmark (the org), and the next sentence's scores inherit the
    # Intelligence Index — a false conflict (Fable 58 vs 65).
    "Coding Agent Index",
)
_GENERIC_BENCHMARK_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _GENERIC_BENCHMARK_PHRASES) + r")\b",
    re.I,
)


def _generic_benchmark_phrase(text: str) -> str:
    """The benchmark phrase in text, if any, from the phrase list
    (_generic_benchmark misses these: CVE Rediscovery, hyphenated
    and title-case names). Returns the canonical phrase as listed,
    or "" when none matches."""
    m = _GENERIC_BENCHMARK_PHRASE_RE.search(text)
    if not m:
        return ""
    hit = m.group(0)
    for p in _GENERIC_BENCHMARK_PHRASES:
        if p.lower() == hit.lower():
            return p
    return hit


def _span_bench_slugs(span: str) -> set:
    """Benchmark slugs named in a claim's sentence span (phrase list
    + CamelCase tokens, stoplist excluded). Used by the merge's
    bench-attach guard: a bench-less score whose sentence names
    benchmarks must not attach to one it never names."""
    slugs = set()
    for p in _GENERIC_BENCHMARK_PHRASES:
        if re.search(r"\b" + re.escape(p) + r"\b", span, re.I):
            slugs.add(re.sub(r"[^a-z0-9]+", "-", p.lower()).strip("-"))
    for m in _GENERIC_CAMEL_RE.finditer(span):
        if m.group(0).lower() not in _GENERIC_BENCH_ORG_STOPLIST:
            slugs.add(re.sub(r"[^a-z0-9]+", "-", m.group(0).lower()).strip("-"))
    return slugs

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
        # v1.8.8 r2: a row with a single bare number after the colon
        # and no model-score pair ("- DeepSWE v1.1 (agentic coding):
        # 75.7", "- CodeArena, Elo (web development): 1750") still
        # states a score — the bare number IS the row's value, subject
        # unstated. Only for unmapped benchmarks: a mapped row's bare
        # number already has the mapped pass's colon handling, and the
        # generic table-row pass below only covers the unmapped ones.
        # A rest with two numbers ("22 to 31.8") is a range, not a
        # single value, and yields nothing here.
        if not key:
            bare = re.search(r"^\s*(\d+\.?\d*)\s*$", rest.strip())
            if bare:
                entries.append({
                    "benchmark_key":  "",
                    "benchmark_name": name,
                    "model_slug":     "",
                    "model_display":  "",
                    "value":          float(bare.group(1)),
                    "raw":            bare.group(1),
                    "version":        version,
                    "variant":        variant,
                    "line":           line,
                })
    # v1.8.9 (run-20 gap 6): "vs" comparison rows — a bare line
    # "Benchmark: v1% vs v2% vs v3%." whose model order comes from a
    # preceding "(Model vs Model vs Model)" header ("Selected
    # November results published by DeepMind (Neon vs Opus 5.5 vs
    # GPT-6.2 Sol):"). The values are the models' scores in header
    # order; without a header the row is unparseable (no model per
    # value) and stays a sentence-loop claim. "(unreported)" values
    # are skipped. Percentage rows only: the generic row pass
    # upgrades these as percentage/score claims, and a bare-number
    # row (AA Index "61 vs 59 vs 57") already aligns through the
    # sentence loop — rewriting its role would split it. Only a
    # label that resolves to a benchmark (mapped key, phrase list,
    # CamelCase, hyphenated) counts — a prose line with the same
    # shape never does. The header must sit within 15 lines above
    # the row, so a stale header cannot claim a later table.
    _VS_ROW_RE = re.compile(
        r"^(.+?):\s*(\d[\d,]*\.?\d*\s*%|\(unreported\))\s+vs\.?\s+"
        r"(\d[\d,]*\.?\d*\s*%|\(unreported\))\s+vs\.?\s+"
        r"(\d[\d,]*\.?\d*\s*%|\(unreported\))\.?\s*$", re.I)
    lines = text.split("\n")
    header_models: List[str] = []
    header_idx = -100
    for li, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        hm = re.search(r"\(([^)]+)\)", stripped)
        if hm:
            parts = [p.strip() for p in
                     re.split(r"\s+vs\.?\s+", hm.group(1), flags=re.I)]
            if len(parts) >= 2 and all(
                    _MODEL_TOKEN_RE.fullmatch(p) for p in parts):
                header_models = parts
                header_idx = li
                continue
        m = _VS_ROW_RE.match(stripped)
        if not m or not header_models or li - header_idx > 15:
            continue
        label = m.group(1).strip()
        values = [m.group(2), m.group(3), m.group(4)]
        if len(values) != len(header_models):
            continue
        mapped = _normalize_entity_key(label)
        key = mapped[0] if mapped else ""
        name = _table_benchmark_name(label)
        if not key and not name:
            continue
        if not key and not _generic_benchmark_phrase(label) \
                and not _GENERIC_CAMEL_RE.search(name) \
                and "-" not in name:
            continue
        version = _benchmark_version(key, label) if key else ""
        variant = _benchmark_variant(key, label) if key else ""
        for model_display, raw in zip(header_models, values):
            if raw is None or "unreported" in raw.lower():
                continue
            try:
                value = float(raw.rstrip("%").replace(",", "").strip())
            except ValueError:
                continue
            entries.append({
                "benchmark_key":  key,
                "benchmark_name": name,
                "model_slug":     _model_slug(model_display),
                "model_display":  model_display,
                "value":          value,
                "raw":            raw.strip(),
                "version":        version,
                "variant":        variant,
                "line":           stripped,
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


def _tier_word_trails_amount(text_before: str) -> bool:
    """True when the text immediately before a tier word ends with a
    $ amount ("$0.2 cached" — the 'cached' trails the $0.2 price and
    belongs to it, not to the next price). v1.8.8 r2: the backward
    and clause-level tier branches used to steal a previous price's
    trailing qualifier ("[$4 in · $0.2 cached · $20 out / 1M]" gave
    $20 the cached tier) — a tier word that directly trails a $
    amount is that amount's qualifier, never the next amount's."""
    return bool(re.search(r"\$\s*[\d,]+(?:\.\d+)?\s*$", text_before))


_DOT_LIST_TIER_WORDS = {
    "in": "input", "out": "output", "cached": "cached",
    "cache": "cached", "batch": "batch",
}


def _dot_list_tier(text: str, amount_start: int) -> str:
    """Tier from a "·"-separated price list: "[$4 in · $0.2 cached ·
    $20 out / 1M]" — each amount takes the tier word in its own
    segment, after the amount. "in"/"out" are abbreviations the
    full-word tier regexes never see, and the "·" separators mean the
    pair rule ("and"/"/" between amounts) never fires either. Only
    fires on a line with "·" and at least two $ amounts, so a lone
    "$4 in tokens" never reads "in" as input. Returns "" when the
    amount sits in no such list or its segment names no tier."""
    line_start = text.rfind("\n", 0, amount_start) + 1
    nl = text.find("\n", amount_start)
    line = text[line_start:nl if nl != -1 else len(text)]
    if "·" not in line or line.count("$") < 2:
        return ""
    amt_off = amount_start - line_start
    pos = 0
    for part in line.split("·"):
        seg_start = pos
        seg_end = pos + len(part)
        pos = seg_end + 1  # skip the "·"
        if not (seg_start <= amt_off < seg_end):
            continue
        after = part[amt_off - seg_start:]
        m = re.search(r"\b(in|out|cached|cache|batch)\b", after, re.I)
        if m:
            return _DOT_LIST_TIER_WORDS[m.group(1).lower()]
        return ""
    return ""


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
    # Worker B (v1.8.9 correction, Nov 10 unmatched): "against" pairs.
    # "$A against $B <tier-word>" — the tier word immediately after
    # the second amount names BOTH amounts' tier ("$6.721 against
    # $12.82 per task" — both are per-task prices; the first amount
    # carries no tier word of its own and used to mint as bare
    # pricing, i.e. an unmatched claim). The tier word must follow
    # the second amount directly (an optional "per million" may
    # intervene); a tier word in a later clause never counts
    # ("$0.20 against $0.25) and batch ..." — the $0.25 is cached,
    # not batch, so the lookahead stops at the closing paren).
    for m in re.finditer(
            r"\$\s?(\d[\d,\.]*)\s+against\s+\$\s?(\d[\d,\.]*)", text, re.I):
        if m.start(1) in out or m.start(2) in out:
            continue
        after = text[m.end():m.end() + 30]
        tm = re.match(
            r"\s*(?:per\s+(?:million|1M)\s+)?"
            r"(input|output|cached|cache|batch|per[-\s]?task)\b",
            after, re.I)
        if not tm:
            continue
        tier = tm.group(1).lower().replace(" ", "-")
        if tier == "cache":
            tier = "cached"
        out[m.start(1)] = tier
        out[m.start(2)] = tier
    return out


_FUTURE_MONTH_RE = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
# A price taking effect at a later date: "starting/from/effective
# <Month> <day>" ("$4 starting December 1", "$4 / $20 from Dec. 1"),
# or "takes effect (on) <Month> <day>" ("the new rate takes effect
# December 1" — run-20 gap 3). The day is optional: the extractor's
# "sentence" is ". "-delimited, so an abbreviated month ("Nov.",
# "Dec.") truncates the sentence at the abbreviation's period and the
# day never reaches the era check ("then $4 / $20 from Dec" — the "1"
# sits in the next fragment).
_FUTURE_EFFECTIVE_RE = re.compile(
    r"\b(?:starting|from|effective|takes?\s+effect(?:\s+on)?)\s+" + _FUTURE_MONTH_RE + r"\.?(?:\s+\d{1,2})?",
    re.I,
)
# A price that runs through a stated end date ("rates run through
# November 30", "the $2 rate holds through Nov. 30") is the price in
# effect now, ending on that date — current, not future. The verb
# must be a continuation verb: a change verb ("rise to") in the same
# clause means the price AFTER the date, which the future rule owns.
_THROUGH_CURRENT_RE = re.compile(
    r"\b(?:runs?|lasts?|holds?|remains?|stays?|valid|good|applies|apply)\b[^.!?\n]{0,60}?\bthrough\s+"
    + _FUTURE_MONTH_RE + r"\.?(?:\s+\d{1,2})?",
    re.I,
)
# A sentence that opens with a date and then changes a price
# ("December 1, 2026: the API price becomes $4 ...") — the date is
# the new price's effective date, not the old one's. An optional list
# marker ("- December 1, 2026: ...") does not change that.
_FUTURE_LEADING_DATE_RE = re.compile(
    r"^\s*(?:[-•*]\s*)?" + _FUTURE_MONTH_RE + r"\.?\s+\d{1,2},?\s+\d{4}\s*:",
    re.I,
)
_FUTURE_CHANGE_VERB_RE = re.compile(
    r"\b(?:becomes?|changes?\s+to|(?:rises?|increases?|drops?|falls?|moves?|switch(?:es)?)\s+to)\b",
    re.I,
)


# An explicit current-era marker: the price is stated as the price in
# effect now ("costs $2 today", "now costs $2", "currently $2",
# "the new price is $2", "$2 at present"). The set mirrors the
# prior-cue reset markers the era check already treated as current
# ("now"/"currently"/"new"), plus "today" and "at present". A price
# with no era marker at all is era "unknown", never "current" — an
# unstated effective date must fail closed (Arch audit 2026-11-13:
# an undated "$4 per million tokens" must not contradict a dated
# "$2 today", and must not silently align with it either).
_CURRENT_MARKER_RE = re.compile(
    r"\b(?:today|now|currently|new|at\s+present)\b",
    re.I,
)


def _pricing_era(sentence: str, amount_start: int) -> str:
    """Price era for one $ amount: "future" when the amount is a price
    taking effect at a later date, "prior" when the amount is
    introduced as superseded ("down from $10") or sits on the second
    side of an "against"/"vs" pair in a sentence naming a prior
    model/version, "current" ONLY when the sentence states the price
    is in effect now (an explicit current marker before the amount,
    or just after it — "costs $2 today"), and "unknown" otherwise.
    "Unknown" is the fail-closed default: a price that states no era
    never stands in for a stated one, so it partitions apart from
    both "current" and "future" prices instead of contradicting or
    silently aligning with them. Two unstated prices still compare
    with each other (unknown == unknown) — that is the November 11
    replay's behaviour, where all four $2/$10 reports state no era
    and align. A prior cue applies to the whole price list in its
    sentence ("Previously $10 input and $50 output" — the $50 is
    prior too, not just the $10), unless a current marker resets it
    after the cue. Future is checked first: a sequenced change
    ("through November 30, then $4 starting December 1") names the
    new price, not a superseded one, even when the sentence also
    carries a prior cue elsewhere.
    The future cases, all relative to the amount's position:
    (a) after "then" in a sentence that also names a future-effective
    date ("then $4 and $20 starting December 1"); (b) immediately
    followed by "starting/from/effective <Month>" before the next $
    ("$4.00 starting December 1, 2026"), or by "takes effect (on)
    <Month>"; (c) in a sentence that opens with a date and changes
    the price ("December 1, 2026: the API price becomes $4 ...");
    (d) introduced by a change verb ("rise to", "becomes") after a
    dated end in the same sentence ("rates run through November 30,
    after which they rise to $4"). A price whose own clause says it
    runs through a stated end date ("the $2 rate runs through
    November 30") is current — in effect now, ending on that date."""
    before = sentence[:amount_start]
    # (a) Sequenced change: "then" before the amount, and the
    # sentence names the future-effective date the change lands on.
    if re.search(r"\bthen\b", before, re.I) and _FUTURE_EFFECTIVE_RE.search(sentence):
        return "future"
    # (b) Immediately followed by a future-effective date, before the
    # next $ (a later price's date never counts for this one).
    after = sentence[amount_start:amount_start + 30]
    dollar = after.find("$")
    if dollar != -1:
        after = after[:dollar]
    if _FUTURE_EFFECTIVE_RE.search(after):
        return "future"
    # (c) Sentence opens with the effective date and changes the price.
    if _FUTURE_LEADING_DATE_RE.match(sentence) and _FUTURE_CHANGE_VERB_RE.search(before):
        return "future"
    # (d) v1.8.9 (run-20 gap 3): a price introduced by a change verb
    # after a dated end ("promotional rates run through November 30,
    # after which they rise to $4 ...") is the price AFTER that date
    # — future. The date must precede the amount and the change verb
    # must precede it too; a past-tense change ("dropped to $2 after
    # the November sale") never matches the verb regex, so it cannot
    # be misread as a future price.
    if _FUTURE_CHANGE_VERB_RE.search(before) and re.search(
            r"\b(?:through|after|on)\s+" + _FUTURE_MONTH_RE + r"\.?(?:\s+\d{1,2})?",
            before, re.I):
        return "future"
    prior_m = _PRIOR_CUE_RE.search(before)
    if prior_m:
        # A current marker after the prior cue and before the amount
        # resets the era to current.
        if _CURRENT_MARKER_RE.search(before[prior_m.end():]):
            return "current"
        return "prior"
    # Second side of an "against"/"vs" pair, in a sentence naming a
    # prior model/version: "$0.20 against $0.25" in a "cheaper than
    # Fable 5.1" sentence is the 5.1 price, not the 5.2 price.
    if re.search(r"\b(?:against|vs\.?)\b", sentence[:amount_start]) \
            and _PRIOR_MODEL_RE.search(sentence):
        return "prior"
    # Explicit current marker: in the amount's own clause before it
    # ("now costs $2") — clause-scoped, so a marker belonging to an
    # earlier price in the same sentence ("costs $2 today, ... $4 ...")
    # never leaks onto a later price — or just after it in the same
    # clause ("costs $2 today", "costs $2 per million tokens at
    # present"). The clause bound is the scope: a later price in a
    # different clause never lends this one its marker.
    clause_a, clause_b = _generic_clause_span(sentence, amount_start, amount_start)
    if _CURRENT_MARKER_RE.search(sentence[clause_a:amount_start]) \
            or _CURRENT_MARKER_RE.search(sentence[amount_start:clause_b]):
        return "current"
    # v1.8.9 (run-20 gap 3): a price whose own clause says it runs
    # through a stated end date ("the $2 rate runs through November
    # 30") is the price in effect now, ending on that date — current.
    # Clause-scoped: a "through" in an earlier clause belongs to the
    # old rate, not to a later price introduced by a change verb
    # ("... run through November 30, after which they rise to $4" —
    # the $4's clause has no "through", and rule (d) already claimed
    # it as future).
    if _THROUGH_CURRENT_RE.search(sentence[clause_a:clause_b]):
        return "current"
    # No era stated: unknown, never "current". An unstated price must
    # not contradict a stated one, and must not silently align with
    # one either — it partitions apart from both.
    return "unknown"


# Thinking level for a per-task price: "at high/medium/low (thinking)"
# just after the amount, or a parenthesized level just before it.
_EFFORT_AFTER_RE = re.compile(r"\bat\s+(high|medium|low)\b", re.I)
_EFFORT_BEFORE_RE = re.compile(r"\(\s*(high|medium|low)\s*\)", re.I)


def _pricing_effort(sentence: str, amount_start: int) -> str:
    """Thinking level for one per-task $ amount: high/medium/low.
    After-match first ("$7.76 per task on Carbon at high thinking",
    "$6.59 at medium") — the level named just after the amount,
    within 40 chars and before the next $ (a later price's level
    never counts). Else a parenthesized level just before it in the
    same sentence ("60.0 (high), second, $7.76 per task"). "" when
    the sentence names no level — the price is then level-less and
    partitions with the other level-less per-task prices of its
    subject. Only used for per-task partitioning: a standard-tier
    price does not vary by thinking level, so it never carries one."""
    after = sentence[amount_start:amount_start + 40]
    dollar = after.find("$")
    if dollar != -1:
        after = after[:dollar]
    m = _EFFORT_AFTER_RE.search(after)
    if m:
        return m.group(1).lower()
    before = sentence[max(0, amount_start - 40):amount_start]
    m = _EFFORT_BEFORE_RE.search(before)
    if m:
        return m.group(1).lower()
    return ""


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
    # Worker B (v1.8.9 correction): a "for <Model>" naming the price's
    # owner just after the amount ("$10.04 for Opus 5.5 (Max)") is a
    # stronger, more local signal than a model at the clause's start
    # ("Fable 5.2 (High) scores 61.82 for $9.08 ... for $10.04 for
    # Opus 5.5" — the $10.04 is Opus's, not Fable's), so check it
    # first. The window ends at the next "$": a "for <Model>" after
    # another price belongs to that price, not this one ("$9.08 per
    # index task against 59.11 for $10.04 for Opus 5.5" — the Opus
    # mention is past the $10.04, so $9.08 keeps its clause subject).
    after = sentence[amount_start:amount_start + 40]
    dollar = after.find("$")
    window = after if dollar < 0 else after[:dollar]
    m = re.search(r"\bfor\s+(" + _MODEL_TOKEN_RE.pattern + r")", window)
    if m:
        return _model_slug(m.group(1))
    best = ""
    for m in _MODEL_TOKEN_RE.finditer(before):
        best = m.group(1)
    if best:
        return _model_slug(best)
    return ""



_BASELINE_MARKER_RE = re.compile(
    r"\b(?:than|compared\s+with|compared\s+to|versus|vs\.?|against|much\s+as|same\s+as)\b",
    re.I)


def _model_token_is_baseline(before: str, tok_start: int,
                             after: str = "") -> bool:
    """True when a model token is the object of a comparison phrase
    rather than the measured subject: the nearest comparison marker
    (than / compared with|to / versus / vs / against / much as /
    same as) before the token has nothing but other model tokens,
    "and", and whitespace between it and this token ("faster than
    Claude Fable 5.1 and Claude Mythos 5.1" — both are baselines;
    "twice as much as Opus 5.5" — Opus is the baseline). A comma or
    any other word in between ends the comparison list, so a model
    named in a later clause ("faster than Opus, and Mythos costs
    $8") is NOT a baseline. A number + "for" in between ("compared
    to 94.49% for GPT-5.6") is not a bare model list either — that
    shape is the exactly-one fallback's job, not this helper's.
    Worker B: a possessive right after the token ("compared with
    Argon's 50") makes the model the OWNER of the following figure,
    not a baseline — Argon's 50 is Argon's score, even though the
    comparison is against it. `after` is the text after the token
    up to the number; a leading "'s" vetoes the baseline reading."""
    if after.lstrip().startswith("'s"):
        return False
    markers = list(_BASELINE_MARKER_RE.finditer(before[:tok_start]))
    if not markers:
        return False
    rest = before[markers[-1].end():tok_start]
    while True:
        rest = rest.strip()
        if not rest:
            return True
        if rest.lower().startswith("and"):
            rest = rest[3:]
            continue
        m = _MODEL_TOKEN_RE.match(rest)
        if m:
            rest = rest[m.end():]
            continue
        return False


def _pricing_is_against_second_side(sentence: str, amount_start: int) -> bool:
    """True when a $ amount sits on the second side of an "against" /
    "vs" comparison ("$8 / $40 against $4 / $20", "$0.20 against
    $0.25", "$10 / $50 for Claude Fable 5.1" after "against $10") —
    i.e. an "against"/"vs" precedes it in the sentence AND a $ amount
    precedes that "against". The second side belongs to the compared-
    against model, never to the sentence's subject; the first side
    belongs to the sentence's subject. Worker B (v1.8.9 correction):
    the old code let the nearest-model rule claim second-side prices
    for the first side's model (Medium's prior $10/$50 read as
    Fable 5.2's because "Fable 5.2" precedes them)."""
    before = sentence[:amount_start]
    m = None
    for cand in re.finditer(r"\b(?:against|vs\.?)\b", before, re.I):
        m = cand
    if m is None:
        return False
    if "$" not in before[:m.start()]:
        return False
    # The "against" must belong to THIS amount's pair: between the
    # "against" and the amount there may be only the pair's own
    # figures — a bare number or a "$A / $B" slash list ("$8 / $40
    # against $4 / $20": both $4 and $20 are second side). Any
    # other word in between means the "against" belongs to an
    # earlier pair ("$9.08 per index task against 59.11 for $10.04
    # ... (48.21 for $2.45 ..." — the $2.45 is not the second side
    # of the $9.08 pair, the $10.04 is). amount_start points at
    # the digits, so the amount's own "$" is the last char of
    # `before` — strip it before checking the gap.
    gap = before[m.end():]
    if gap.endswith("$"):
        gap = gap[:-1]
    return (re.fullmatch(r"\s*\$?[\d.,]*\s*(?:/\s*\$?[\d.,]*\s*)*", gap)
            is not None
            and "\n" not in gap and ";" not in gap)


def _pricing_against_subject(sentence: str, amount_start: int) -> str:
    """The model a price belongs to when it is the second side of an
    "against" comparison naming a different model ("Fable 5.2 costs
    twice as much as Opus 5.5 ($8 / $40 against $4 / $20 per
    million)" — the $4/$20 are Opus 5.5's, not Fable 5.2's, and must
    not compare against Fable's $8/$40). The subject is the model in
    the sentence's "than/as <Model>" comparison phrase, or — Worker B
    (v1.8.9 correction) — the model named just after the amount as
    "for <Model>" ("$10 / $50 for Claude Fable 5.1", "$1.00 for
    Astra"). Returns "" when the amount is not after an "against" or
    the sentence names no comparison model — the claim is then
    subject-less and partitions with the other subject-less prices of
    its tier."""
    before = sentence[:amount_start]
    if not re.search(r"\b(?:against|vs\.?)\b", before):
        return ""
    m = re.search(
        r"\b(?:than|as)\s+(" + _MODEL_TOKEN_RE.pattern + r")", before)
    if m:
        return _model_slug(m.group(1))
    after = sentence[amount_start:amount_start + 40]
    m = re.search(r"\bfor\s+(" + _MODEL_TOKEN_RE.pattern + r")", after)
    return _model_slug(m.group(1)) if m else ""



def _pricing_standard_subject(sentence: str, amount_start: int,
                              text_before: str = "") -> str:
    """The model a standard-tier (input/output/cached/...) price
    belongs to: the nearest model token before the amount in the
    sentence ("GPT-6.2 Sol, which costs $2 per million input
    tokens" is Sol's $2; "Claude Opus 5.5, which is priced at $4
    ..." is Opus 5.5's $4). Sentence-scoped, not clause-scoped:
    the model sits in the clause before the price clause in exactly
    these list sentences. v1.8.9 (run-20 gap 3): standard tiers
    were never split by subject, so three models' input prices
    (Neon $3, Sol $2, Opus $4) pooled under one pricing-input
    identity and read as a conflict. A price with no model token
    before it stays subject-less (fail-closed).
    Worker B (v1.8.9 correction): a model named as a comparison
    BASELINE is never the subject — "faster than Claude Fable 5.1
    and Claude Mythos 5.1, and are available at a lower price: $8"
    is Fable/Mythos 5.2's $8, not Mythos 5.1's (the nearest token
    before the amount). Baseline tokens are skipped; the nearest
    non-baseline token wins. A bare family name ("Fable's API
    price") resolves to the versioned model of the same family
    named earlier in the sentence — or, when the sentence was
    truncated at an abbreviation ("Nov. 9, cutting Fable's API
    price"), earlier in the document (text_before) — so the bare
    name and the versioned name meet instead of splitting. The old
    "against follows the amount" guard is gone: it blanked
    first-side prices too ("cached tokens for Fable 5.2 cost $0.20
    per million against $1.00" — the $0.20 IS Fable 5.2's);
    second-side prices are the against rule's job, and the pricing
    pass now routes them there first."""
    before = sentence[:amount_start]
    tokens = list(_MODEL_TOKEN_RE.finditer(before))
    best_m = None
    for m in reversed(tokens):
        if _model_token_is_baseline(before, m.start(), before[m.end():]):
            continue
        best_m = m
        break
    if best_m:
        best = best_m.group(1)
        # Bare family name: resolve to the versioned model of the
        # same family named earlier in the sentence, if any.
        if not re.search(r"\d", best):
            fam = best.strip().lower()
            for m in tokens:
                if m.start() >= best_m.start():
                    break
                tok = m.group(1)
                if re.search(r"\d", tok) \
                        and _model_slug(tok).startswith(fam + "-"):
                    best = tok
            if not re.search(r"\d", best) and text_before:
                # The sentence may have been truncated at an
                # abbreviation ("Nov. 9, ...") — look earlier in
                # the document for the versioned model.
                for m in _MODEL_TOKEN_RE.finditer(text_before):
                    tok = m.group(1)
                    if re.search(r"\d", tok) \
                            and _model_slug(tok).startswith(fam + "-"):
                        best = tok
        return _model_slug(best)
    m = re.search(r"\bfor\s+(" + _MODEL_TOKEN_RE.pattern + r")",
                  sentence[amount_start:amount_start + 40])
    if m:
        return _model_slug(m.group(1))
    return ""
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
    r"|\b(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
    # v1.8.8 r2: slash dates ("2026/11/15") — the primary's release
    # date in run 17 never minted, so it never joined the shared-date
    # set (gap 11). Groups 10/11/12; the ISO branch's 7/8/9 are
    # untouched.
    r"|\b(\d{4})/(\d{2})/(\d{2})\b",
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
    "argon", "neon", "hydrogen",
    "gpt", "claude", "grok", "llama", "mistral", "qwen", "deepseek",
    "inkling", "ernie",
})
_MODEL_TOKEN_RE = re.compile(
    r"\b((?:Claude\s+)?(?:Fable|Mythos|Opus|Sonnet|Gemini|Grok|Llama|Mistral|Qwen|DeepSeek|Inkling|ERNIE|Astra|Carbon|Argon|Neon|Hydrogen)"
    r"(?:\s+\d+(?:\.\d+)?(?:\s+(?:Astra|Carbon|Argon|Neon|Hydrogen|Cyber|Sol))?)?"
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
    "Astra" is GPT-6 Astra, "Carbon" is Gemini 4 Carbon, "Argon" is
    Gemini 4 Argon, "Neon" is Gemini 4 Neon, "Hydrogen" is Gemini 4
    Hydrogen — a bare "Astra 42.8%" and a "GPT-6 Astra 42.8%"
    are the same model's score, not two subjects. Versioned names
    (Opus, Sonnet, Fable) are ambiguous across versions and are NOT
    aliased."""
    token = re.sub(r"^claude\s+", "", token.strip(), flags=re.I)
    slug = re.sub(r"[^a-z0-9]+", "-", token.lower()).strip("-")
    return {"astra": "gpt-6-astra", "carbon": "gemini-4-carbon",
            "argon": "gemini-4-argon", "neon": "gemini-4-neon",
            "hydrogen": "gemini-4-hydrogen"}.get(slug, slug)


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


def _number_in_benchmark_parens(text: str, num_start: int) -> bool:
    """True when a number sits in a benchmark's parenthesized score
    ("RiemannBench (84.0%)") — the number belongs to the benchmark
    named before the open paren, not to a benchmark named after it
    in the same sentence. v1.8.9 (run-20 gap 6): the reverse pass
    claimed the 84.0% for HLE because "HLE-Diamond" followed it,
    while its own benchmark sat in the parens before it."""
    before = text[max(0, num_start - 40):num_start]
    m = re.search(r"([A-Za-z][\w\-']*(?:\s+[A-Za-z][\w\-']*){0,2})\s*\(\s*$", before)
    if not m:
        return False
    name = m.group(1)
    if _generic_benchmark_phrase(name):
        return True
    if re.search(r"\b(" + _BENCHMARK_NAMES_RE + r")\b", name, re.I):
        return True
    cm = _GENERIC_CAMEL_RE.search(name)
    return bool(cm and cm.group(0).lower() not in _GENERIC_BENCH_ORG_STOPLIST)


def _number_followed_by_benchmark(text: str, num_end: int) -> bool:
    """True when a matched number is directly followed by a benchmark
    name ("the 84.0% RiemannBench result") — the number is that
    benchmark's score, not the preceding entity's. v1.8.9 (run-20
    gap 6): the forward inline pattern claimed the 84.0% for HLE
    (named earlier in the sentence) although RiemannBench follows
    it. The reverse pass cannot mint it (RiemannBench is unmapped),
    so the generic pass owns it; the preceding entity must not claim
    it. Org names are excluded by the stoplist, so "71.3% DeepMind
    reports" never trips it."""
    after = text[num_end:num_end + 40]
    m = re.match(r"\s+", after)
    rest = after[m.end():] if m else after
    # A model token is a measured system, not a benchmark — "MMLU
    # (EM) 5-shot: ERNIE 93.3, DeepSeek-V4-Flash 89.4" must not read
    # DeepSeek as the following benchmark (v1.8.9: the guard as first
    # written swallowed the MMLU 93.3 this way).
    if _MODEL_TOKEN_RE.match(rest):
        return False
    if _GENERIC_BENCHMARK_PHRASE_RE.match(rest):
        return True
    cm = _GENERIC_CAMEL_RE.match(rest)
    return bool(cm and cm.group(0).lower() not in _GENERIC_BENCH_ORG_STOPLIST)


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
        elif m.group(10):
            # v1.8.8 r2: slash date (YYYY/MM/DD).
            year, month, day = int(m.group(10)), int(m.group(11)), int(m.group(12))
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

# A stock exchange can be the venue of a listing without being a measured index.
_VENUE_MENTION = re.compile(
    r"\bNasdaq(?:\s+Global\s+(?:Select\s+)?Market|\s+debut|\s+under\s+(?:the\s+)?(?:ticker|symbol))\b", re.I)
_SINCE_INDEX = re.compile(
    r"\b(since\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:,?\s+\d{4})?)\s*,\s*(?:the\s+)?$", re.I)
_SINCE_AFTER = re.compile(
    r"^\s+(since\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:,?\s+\d{4})?)\b", re.I)


def _windowed_index(key, label, text, index_start, value_end):
    if key in ("sp500", "nasdaq", "dow"):
        window = (_SINCE_INDEX.search(text[max(0, index_start-72):index_start])
                  or _SINCE_AFTER.match(text[value_end:value_end+64]))
        if window:
            period = window.group(1).lower().replace(".", "")
            return f"{key}|{period}", f"{label} ({period})"
    return key, label


_DEBUT_CLOSE = re.compile(
    r"\b(?:closed its first (?:day|session) at|(?:finished|ended) its first (?:day|session)[^.!?\n]{0,80}?, closing at)\s*"
    r"\$(\d+(?:\.\d{1,2})?)\s+(?:a|per) share\b", re.I)
_LISTING_NAME = re.compile(
    r"(?:^|[.!?]\s+)([A-Z][\w-]*(?: [A-Z][\w-]*){0,3})(?: (?:finished|ended) its first (?:day|session)|, [^.!?]{1,90}? made its (?:Nasdaq|NYSE) debut)")


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
    seen: Dict[Tuple, int] = {}  # (entity_key, numeric_value, variant, subject, era, effort) -> index into claims
    dropped: List[Dict] = []

    def _add(key: str, label: str, raw_val: str, context: str, unit_override: str = "",
             version: str = "", shot_count: Optional[int] = None, variant: str = "",
             subject: str = "", era: str = "", effort: str = "") -> None:
        numeric = _parse_value(raw_val)
        if numeric is None:
            return
        if key == "nasdaq" and _VENUE_MENTION.search(context):
            dropped.append(dict(entity_key=key, entity_label=label, raw_value=raw_val.strip(),
                                reason="exchange named as a listing venue"))
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
        # different price eras (current vs prior vs unknown) are
        # DIFFERENT claims too — the dedup key carries subject and era.
        # v1.8.8: the same per-task price at different thinking
        # levels (high vs medium) is a DIFFERENT claim too — the
        # dedup key carries effort.
        dedup   = (key, numeric, variant, subject, era, effort)
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
            "effort":        effort,
            "context":       context[:120],
        })

    text = _without_metadata(text)
    # Pass 1 — colon-delimited data lines (higher precision)
    for m in _COLON_RE.finditer(text):
        label_fragment = m.group(1).strip()
        raw_val        = m.group(2).strip()
        entity = _normalize_entity_key(label_fragment)
        if entity is None:
            continue
        key, label, ematch = entity
        key, label = _windowed_index(key, label, text, m.start(1) + m.group(1).find(ematch.group()), m.end(2))
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
        key, label = _windowed_index(key, label, text, m.start(), m.end(2))
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
        if key in _BENCHMARK_KEYS and _number_followed_by_benchmark(text, m.end()):
            # v1.8.9 (run-20 gap 6): "the 84.0% RiemannBench result"
            # — the number is the FOLLOWING benchmark's score; the
            # preceding entity must not claim it.
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
        if _number_in_benchmark_parens(text, m.start(1)):
            # v1.8.9 (run-20 gap 6): "RiemannBench (84.0%), and
            # HLE-Diamond" — the 84.0% is RiemannBench's parenthesized
            # score, not HLE's; the reverse pass must not claim it.
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
    # from $10", "against $10 / $50 for Fable 5.1"); a price that
    # states no era is era "unknown" and partitions apart from stated
    # eras instead of standing in for one. A per-task price
    # carries its measured subject. A $ amount with no resolvable tier
    # mints as bare "pricing" and is never compared (fail-closed).
    pair_tiers = _pricing_pair_tiers(text)
    close_amounts = set()
    for paragraph in re.finditer(r"[^\n]+", text):
        body = paragraph.group()
        closes = list(_DEBUT_CLOSE.finditer(body))
        issuers = list(_LISTING_NAME.finditer(body))
        tickers = set(re.findall(r'\b(?:ticker|symbol|on (?:Nasdaq|NYSE) as)\s+["“]?([A-Z]{1,6})\b', body))
        if len(closes) != 1 or len(issuers) != 1 or len(tickers) != 1:
            continue
        if re.search(r"\b(?:if|forecast|expected|projected|denied|incorrectly|hypothetical|would|could)\b", body, re.I):
            continue
        close = closes[0]
        owner = issuers[0]
        if close.start() < owner.start(1):
            continue
        if close.group().lower().startswith("closed"):
            # Only bind the directly following stock/ticker clause to the issuer.
            bridge = re.sub(r"(?<=\d)\.(?=\d)", "_", body[owner.end():close.start()])
            if not re.fullmatch(r'[^.!?]{0,90}\.\s+The stock(?:,? (?:trading under the |with )ticker ["“]?[A-Z]{1,6}(?:,?["”])?,?)? ', bridge):
                continue
        issuer = owner.group(1)
        ticker = next(iter(tickers))
        close_amounts.add(paragraph.start() + close.start(1))
        _add(f"share-close|{issuer.casefold()}|{ticker}|ipo-day-one", f"{issuer} ({ticker}) first-session close",
             close.group(1), close.group(), unit_override="$/share")
    prev_amount: Dict = {}  # the previous $ amount's tier/era, for slash pairs
    for m in _DOLLAR_RE.finditer(text):
        if m.start(1) in close_amounts:
            continue
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
            # v1.8.8 r2: a "·"-separated price list ("[$4 in · $0.2
            # cached · $20 out / 1M]") — each amount takes the tier
            # word in its own segment, including the "in"/"out"
            # abbreviations the full-word regexes never see.
            tier = _dot_list_tier(text, m.start())
        if not tier:
            back_seg = window[max(0, dollar_pos - 16):dollar_pos]
            back = _BACKWARD_TIER_RE.search(back_seg)
            # v1.8.8 r2: a tier word that directly trails a $ amount
            # belongs to that amount, not this one ("$0.2 cached $20"
            # must not give $20 the cached tier).
            if back and not _tier_word_is_speed(back_seg, back.end()) \
                    and not _tier_word_trails_amount(back_seg[:back.start()]):
                raw_tier = back.group(1).lower()
                if raw_tier in ("cache", "cached"):
                    tier = "cached"
                elif raw_tier.replace(" ", "-") in ("per-task",):
                    tier = "per-task"
                elif raw_tier in ("input", "output", "batch"):
                    # "Cached input: $0.10" — a cached/cache word
                    # immediately before the tier word qualifies it:
                    # the price is a cached price, not a standard one.
                    # Without this the nearest tier word wins and the
                    # cached price conflates with the standard input
                    # price of the same model.
                    if raw_tier in ("input", "output") and re.search(
                            r"\b(?:cached|cache)\s+$", back_seg[:back.start()], re.I):
                        tier = "cached"
                    else:
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
                    # v1.8.8 r2: a tier word that directly trails a $
                    # amount belongs to that amount, not this one —
                    # but ONLY across a "·" list separator ("$0.2
                    # cached · $20" — the "cached" is $0.2's). In a
                    # comma/and series ("$7.76 per task ..., $6.59 at
                    # medium") the tier word IS the series' tier for
                    # the later amounts, which carry none of their own.
                    abs_pos = clause_start + wm.start()
                    if _tier_word_trails_amount(
                            sentence[max(0, abs_pos - 20):abs_pos]) \
                            and "·" in clause_before[wm.end():]:
                        continue
                    if wm.start() > best_pos:
                        best_pos, best_tier = wm.start(), word_key
            if best_tier:
                # Same cached qualifier as the backward branch: a
                # cached/cache word immediately before the winning
                # tier word makes it a cached price ("cached tokens
                # ... cost $0.20" is a cached price, not an input one).
                if best_tier in ("input", "output") and re.search(
                        r"\b(?:cached|cache)\s+$", clause_before[:best_pos], re.I):
                    tier = "cached"
                else:
                    tier = best_tier
        if not tier:
            # "Against"-pair inheritance: the second element of an
            # "X against Y" pair takes the first element's tier when
            # the pair names one price list ("cache reads ($0.20
            # against $0.25)" — both are cache-read prices; "cached
            # tokens ... cost $0.20 per million against $1.00 for
            # Astra" — the $1.00 is Astra's cached-token price, so
            # it inherits the cached tier too, and the "for Astra"
            # names its SUBJECT, not a different tier). Worker B
            # (v1.8.9 correction): the old check required the text
            # between the two amounts to be exactly "against", so a
            # unit phrase ("per million against") broke the
            # inheritance and the second price minted as bare
            # pricing — an unmatched claim. Any short gap carrying
            # "against"/"vs" and no other $ now inherits.
            if prev_amount.get("tier") and prev_amount.get("end") is not None:
                between_prev = text[prev_amount["end"]:m.start()]
                if len(between_prev) <= 40 and "$" not in between_prev \
                        and "\n" not in between_prev \
                        and re.search(r"\b(?:against|vs\.?)\b", between_prev):
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
        subject = ""
        if _pricing_is_against_second_side(sentence, amount_off):
            # Worker B (v1.8.9 correction): a second-side price
            # belongs to the compared-against model — the against
            # rule runs FIRST, before the nearest-model rule can
            # claim it for the first side's model (Medium's prior
            # $10/$50 "against $10 / $50 for Claude Fable 5.1" read
            # as Fable 5.2's; the $4/$20 "against" Opus 5.5's
            # would read as Fable 5.2's the same way).
            subject = _pricing_against_subject(sentence, amount_off)
        elif tier == "per-task":
            subject = _pricing_subject(sentence, amount_off)
        else:
            # v1.8.9 (run-20 gap 3): standard tiers partition by
            # subject too — the partition already is (era, subject),
            # it was just always "" for standard tiers. Worker B:
            # a bare "pricing" claim also takes its subject here
            # ("GPT-6 Astra costs $10 and $50 on standard tasks" —
            # Astra's prices, tier-less but owned, so they stand as
            # Astra uniques instead of unmatched abstentions).
            # text[:m.start()] is passed so a bare family name can
            # resolve against the versioned model named earlier in
            # the document when the sentence itself was truncated
            # at an abbreviation ("Nov. 9, ...").
            subject = _pricing_standard_subject(
                sentence, amount_off, text[:m.start()])
        # Thinking level, per-task prices only: a standard-tier price
        # does not vary by thinking level, so it never carries one.
        effort = _pricing_effort(sentence, amount_off) if tier == "per-task" else ""
        if not subject:
            # A standard-tier price on the second side of an "against"
            # comparison naming a different model belongs to that
            # model ("$8 / $40 against $4 / $20" for Opus 5.5) — it
            # must not compare against the first model's prices.
            subject = _pricing_against_subject(sentence, amount_off)
        if subject == "mythos-5-2":
            # Worker B (v1.8.9 correction): Fable 5.2 and Mythos 5.2
            # are the same underlying model at the same $8/$40 price
            # (the primary says so, and lists the same price for
            # both) — a Mythos 5.2 price IS the Fable 5.2 price for
            # comparison, so it canonicalizes to fable-5-2 and the
            # primary's $8/$40 join the other docs' $8/$40 group
            # instead of partitioning under mythos-5-1 (the baseline
            # the old nearest-model rule read). Mythos 5.1 is a
            # different generation and never canonicalizes.
            subject = "fable-5-2"
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
             subject=subject, era=era, effort=effort)
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
# v1.8.8 r2: a feed-header label line ("Source: ...", "Published:
# ...", "Fetched: ...", "URL: ...") is a label, not a sentence — the
# InfoWorld header's "62%" minted as a bench=infoworld claim (run-17
# gap 8). Only the label prefix is skipped; a body sentence that
# happens to contain "Source:" mid-sentence is untouched.
_GENERIC_FEED_HEADER_LINE_RE = re.compile(
    r"^(?:Source|Published|Fetched|URL|Author|Length|Coverage)\s*:", re.I)
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
    r"|\bhigher than\b|\blower than\b|\bagainst\b|\bsplit\s+as\b",
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
    # v1.8.8 r2: a qualifier, not a benchmark ("SWE-bench Verified",
    # "OSWorld-Verified") — left in the named set it defeats the
    # exactly-one rule and blocks the real benchmark attaching.
    "verified",
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
        # v1.8.8 r2: a digit run glued to a preceding letter is part
        # of an identifier ("Work360", "Qwen3.8", "H100"), not a
        # claim value — the "360" in "Work360 benchmark score" is
        # the benchmark's name, and minting it manufactures a
        # task-count conflict (run-17 gap 3).
        if start > 0 and sentence[start - 1].isalpha():
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
        # v1.8.8 r2: a slash-date fragment ("2026/11/15" — the 11 and
        # the 15; a bare "11/15") is a date, never a claim value, just
        # as a hyphen date fragment is not (the range rule above
        # covers "-"). The day guard needs two digits so a genuine
        # fraction ("3/4") keeps minting.
        if start >= 2 and sentence[start - 1] == "/" and sentence[start - 2].isdigit():
            continue
        if re.match(r"/\d{2}\b", sentence[end:]):
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
        _inf_dur_m = None
        if not dur_m:
            # "41.8 inference minutes per task" — the duration unit
            # is "inference minutes", not bare "minutes".
            _inf_dur_m = re.match(
                r"\s+inference\s+(minutes?|hours?|seconds?)\b",
                sentence[end:], re.I)
        if dur_m or _inf_dur_m:
            dur_span = (dur_m.group(0) if dur_m
                        else _inf_dur_m.group(0))
            dur_unit = (dur_m.group(0).strip().lower() if dur_m
                        else _inf_dur_m.group(1).lower())
            # The completion verb may sit well before the duration in
            # a long sentence ("formalized Fermat's Last Theorem in
            # the Lean proof assistant in 30 hours") — check the whole
            # sentence before the number, not a 40-char window.
            before = sentence[:start]
            if not _GENERIC_COMPLETION_VERB_RE.search(before):
                # v1.8.9 (run-20 gap 11): a task duration in a model
                # table row ("Gemini 4 Neon Extra High: 61, $6.44 per
                # task, 21.5 minutes") carries no completion verb —
                # the "per task" + minutes + model-row context IS the
                # duration claim. Without it the minutes never mint
                # and no cross-unit speed check is possible. Only when
                # the sentence names a model token and a per-task
                # phrase, so "took 5 minutes" prose and "within 12
                # hours" context stay out.
                if not (re.search(r"\bper task\b", sentence, re.I)
                        and _MODEL_TOKEN_RE.search(sentence)):
                    continue
            numbers.append({
                "start": start, "end": end + len(dur_span),
                "value": value, "raw": sentence[start:end + len(dur_span)].strip(),
                "unit": dur_unit, "currency": currency,
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
        # v1.8.8 r2: a "%" terminates the number — a following word is
        # a qualifier ("67.7%with tools", "89.0%with tools"), not part
        # of the number. Without this the generic pass skips every
        # glued percentage the mapped pass does not already cover
        # (run-17 gap 4: Chartography 89.0%, OSWorld 81.8%).
        if after.isalpha() and suffix != "%":
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
    # Worker B (v1.8.9 correction): a model named as a comparison
    # BASELINE is never the subject — "about 60% fewer blocks with
    # Fable 5.2 compared with Fable 5.1" is Fable 5.2's 60%, not
    # Fable 5.1's (the nearest token before the number). Baseline
    # tokens (immediately preceded by than / compared with|to /
    # versus / vs / against / as) are skipped; the nearest
    # non-baseline token wins. If every token in the segment is a
    # baseline, fall through to the other subject sources below
    # instead of returning the baseline.
    model_before = list(_MODEL_TOKEN_RE.finditer(seg_before))
    for m in reversed(model_before):
        if _model_token_is_baseline(seg_before, m.start(),
                                    seg_before[m.end():]):
            continue
        display = m.group(1)
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
    # v1.8.9 (run-20 gap 14): a stated uncertainty is not a gap —
    # "Its published uncertainty for Neon is plus or minus 0.94
    # percentage points" is the published margin on a score, not a
    # percentage-point gap between two figures. The "percentage
    # points" cue below would claim it as a gap, so the uncertainty
    # check fires first: the number right after "plus or minus" /
    # "±" / "margin of error", or the number in an "uncertainty"
    # sentence followed by "percentage point(s)".
    _before30 = sentence[max(0, num["start"] - 30):num["start"]]
    if _GENERIC_UNCERTAINTY_BEFORE_RE.search(_before30):
        return "uncertainty"
    if _GENERIC_UNCERTAINTY_SENT_RE.search(sentence) and \
            re.search(r"percentage\s+points?\b",
                      sentence[num["end"]:num["end"] + 25], re.I):
        return "uncertainty"
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
    # v1.8.8 r2 Region 0 (index points): a number immediately followed
    # by "points" where the sentence names a rating index
    # ("Artificial Analysis gives it 43 points", "43 points on its
    # index") is a score on that index — the AA Index never minted
    # at all (run-17 gap 6). "points" alone is not a cue (a "talking
    # points" count is not a score), so the index name is required.
    if re.match(r"\s+points?\b", sentence[num["end"]:num["end"] + 10]):
        clause_a, clause_b = _generic_clause_span(sentence, num["start"], num["end"])
        clause_text = sentence[clause_a:clause_b]
        if "artificial analysis" in sentence.lower() or \
                re.search(r"\bindex\b", clause_text, re.I):
            return "score"
    # v1.8.9 (run-20 gap 10): a board/ranking score — "Arena's
    # November 18 board moved Neon High to third overall at 1,333,
    # three points behind Opus 5.5 Max" — a bare integer in a
    # sentence naming a board (arena/board/leaderboard) with a rank
    # cue (overall/ranked/third/second/first/place) is a score on
    # that board. The value floor (>= 100) keeps small counts out
    # ("ranked third with 5 wins" is not a board score). This only
    # MINTS the revised figure — it does not mark it as a revision:
    # 1,333 aligns with 1,336 inside the large-count tolerance, so
    # the revision itself still needs a variant/revision marker
    # (a design decision, not an extraction fix).
    if num.get("unit") == "count" and float(num["value"]) == int(float(num["value"])) \
            and float(num["value"]) >= 100:
        # A number that already carries its own count noun ("4,200
        # agent sessions", "3,885 votes") is that count, not a board
        # score — the board rule only fires for a bare figure ("at
        # 1,333, three points behind"). "points" is deliberately not
        # in the list: a board score stated in points ("a score of
        # 1,333 points") is still a score.
        _near = sentence[max(0, num["start"] - 25):num["end"] + 25]
        if not re.search(r"\b(?:sessions?|votes?|tasks?|agents?|attempts?|questions?|tests?|runs?|examples?|items?|cases?|evals?)\b",
                         _near, re.I) and \
                re.search(r"\b(?:arena|board|leaderboard)\b", sentence, re.I) and \
                re.search(r"\b(?:overall|ranked?|third|second|first|place[ds]?)\b", sentence, re.I):
            return "score"
    # v1.8.7 decimal guard: a non-integer bare decimal (0.838, 61.82)
    # is a measurement, never a count of a count-noun family — a
    # version fragment or a score must not read as a task count.
    # "sessions" is count-only: a percentage never takes it.
    raw = num.get("raw", "")
    is_bare_decimal = (num.get("unit") == "count" and "." in str(raw)
                       and float(num["value"]) != int(num["value"]))
    is_percent = num.get("unit") == "percent"
    # v1.8.8 Region 0.5 (benchmark score): a percentage whose own
    # clause names a benchmark is a score on that benchmark — the
    # nearest noun is the denominator, not the family ("identified
    # 89.6% of vulnerabilities in its internal CVE Rediscovery
    # evaluation" is a CVE Rediscovery score, not a vulnerabilities
    # count). A rate or derived cue adjacent to the number ("30%
    # faster", "a 51% relative improvement", "16 percentage points")
    # still wins: that is a speed claim, a relative improvement, or
    # a gap — not a score. Clause-scoped, not sentence-scoped: a benchmark named in another clause of the
    # same sentence is a different result ("about 55% fewer blocks
    # ... its ProgramBench result" — the 55% is not a ProgramBench
    # score). Percentages only: a bare count in a benchmark sentence
    # is still a count.
    if is_percent:
        clause_a, clause_b = _generic_clause_span(sentence, num["start"], num["end"])
        clause_text = sentence[clause_a:clause_b]
        if _generic_benchmark_phrase(clause_text) or _generic_benchmark(clause_text, num["start"] - clause_a):
            near = sentence[max(0, num["start"] - 20):num["end"] + 20]
            # The guard checks cue stems, not full phrases: the
            # 20-char window truncates "relative improvement" to
            # "relative improvemen", so the full phrase never matches.
            # A stem is enough — the guard only vetoes the override,
            # the ordinary regions still decide the family.
            # v1.8.9 (run-20 gap 8): a cost word vetoes it too — a
            # "32% lower cost per task" in a benchmark-naming clause
            # is a cost delta, not a score on that benchmark.
            if not re.search(r"\b(?:faster|slower|recall|relative|percentage|cost|price|cheaper|complet\w*|refus\w*)\b", near, re.I):
                return "score"
        # v1.8.8 r2: a benchmark lead-in ("- On Last Secure Code
        # Bench, function-level pass rate rose from 22.0% to 31.8%",
        # "...; on Terminal-Bench 2.0, 59.3 against 52.5") owns the
        # percentages in the clause it introduces — the benchmark
        # sits in the preceding clause, not this one, so the
        # clause-scoped check above misses it. Only a lead-in counts
        # (On/For/In + benchmark, no verb): a complete earlier
        # statement ("HLE was 51.7%") never owns a later percentage.
        # The check is on the IMMEDIATELY preceding clause, not the
        # sentence start — a sentence can open on one benchmark and
        # hand the next clause to another ("On SWE-bench Verified,
        # ...; on Terminal-Bench 2.0, 59.3").
        prev_a, prev_b = _generic_prev_clause_span(sentence, clause_a)
        prev = sentence[prev_a:prev_b] if prev_a is not None else ""
        if prev and re.match(r"\s*[-•*]?\s*(?:On|For|In)\b", prev, re.I) and \
                _generic_benchmark_phrase(prev):
            near = sentence[max(0, num["start"] - 20):num["end"] + 20]
            if not re.search(r"\b(?:faster|slower|recall|relative|percentage|cost|price|cheaper|complet\w*|refus\w*)\b", near, re.I):
                return "score"
        # v1.8.9 (run-20 gap 4): the harness pair — "On SWE-sweep,
        # it already leads on the mini-SWE harness (12.8%) and more
        # than triples to a state-of-the-art 39.0% in Google
        # Antigravity": the 39.0% sits two clauses past the
        # benchmark lead-in, so the clause-scoped check and the
        # immediately-preceding-clause lead-in both miss it, and the
        # number never mints at all. When the sentence names exactly
        # one benchmark before the number and the number's own clause
        # names none, the number is a score on that benchmark — the
        # same exactly-one rule the benchmark-field fallback uses
        # below. The derived-claim guard still vetoes it ("55%
        # fewer blocks ... ProgramBench" keeps its own family; a
        # cost delta in a benchmark sentence — "costs fell 15%" —
        # keeps its cost family), and two named benchmarks stay
        # ambiguous. One more guard (v1.8.8 r2 pinned design): the
        # number's own clause must not contain a fresh "from X% to
        # Y%" range — that clause states its own metric ("the rate
        # of passing all three granularities rose from 8% to 14.7%"),
        # and a lead-in that owned it would also own unrelated
        # later metrics. The harness case has no from-to pair
        # ("more than triples to a state-of-the-art 39.0%"), so it
        # still fires there.
        if len(_span_bench_slugs(sentence[:num["start"]])) == 1 and \
                not re.search(r"\bfrom\s+[\d.,]+\s*%\s+to\s+[\d.,]+\s*%",
                              clause_text, re.I):
            near = sentence[max(0, num["start"] - 20):num["end"] + 20]
            if not re.search(r"\b(?:faster|slower|recall|relative|percentage|cost|price|cheaper|complet\w*|refus\w*)\b",
                            near, re.I):
                return "score"
    # v1.8.9 (run-20 gap 12): a number immediately followed by a
    # plural count noun names a count OF that noun ("Neon's four
    # levels" is a count of levels, not a token figure — the
    # sentence's "token spend" cue used to claim it as tokens, so it
    # pooled with "512K tokens of context" and "1,048,576 tokens"
    # into one false alignment). Only when the noun is not itself a
    # known metric family ("two tasks" is already tasks by the
    # ordinary cues); gap/scale nouns ("points", "times", "percent")
    # and hyphenated modifiers ("five-hour" — no space + noun after
    # the word) never fire it. WORD numbers only: a digit followed by
    # a plural verb ("31.4% matches", "52.6% differ") is not a count
    # of anything — the rule as first written fired on those and
    # minted family "matches", breaking the bare-fragment row test.
    _after_noun = re.match(r"\s+([A-Za-z]+)", sentence[num["end"]:num["end"] + 20])
    if _after_noun and not re.search(r"\d", str(num.get("raw", ""))):
        _noun = _after_noun.group(1).lower()
        # A plural ends in a single "s" — "pass"/"class"/"stress"
        # end in "ss" and are not counts of anything.
        if _noun.endswith("s") and not _noun.endswith("ss") \
                and _noun not in ("points", "times", "percent"):
            _known = {fam for _, fam in _GENERIC_METRIC_CUES} | _GENERIC_COUNT_NOUN_FAMILIES
            if _noun not in _known and _noun.rstrip("s") not in _known:
                return _noun
    # v1.8.9 (run-20 gap 8): a relative cost-per-unit delta ("32%
    # lower cost per task", "36%+ less per task", "25% cheaper per
    # token") keys on metric=cost regardless of other cues in the
    # sentence — the "per task"/"per token" cue used to claim it as
    # tasks/tokens, so the same 32% minted three ways and never
    # compared. A direction word right after the number (allowing a
    # "+" suffix, "36%+ less") plus a cost word and a per-unit phrase
    # in the clause.
    if is_percent:
        _rel_after = sentence[num["end"]:num["end"] + 12]
        if re.match(r"\s*[+−-]?\s*(?:less|more|fewer|higher|lower|above|below|cheaper|costlier|greater|smaller|faster|slower|better|worse)\b",
                    _rel_after, re.I):
            _ca, _cb = _generic_clause_span(sentence, num["start"], num["end"])
            _clause = sentence[_ca:_cb]
            if re.search(r"\b(?:costs?|costing|cheaper|price)\b", _clause, re.I) and \
                    re.search(r"\bper\s+(?:\w+\s+){0,2}(?:task|token|run|test|million)\b", _clause, re.I):
                return "cost"
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


def _generic_prev_clause_span(sentence: str, clause_a: int) -> Tuple[Optional[int], Optional[int]]:
    """The (start, end) span of the comma/and clause before the clause
    starting at clause_a, or (None, None) when there is none."""
    parts = []
    last = 0
    for m in _GENERIC_CLAUSE_SPLIT_RE.finditer(sentence):
        parts.append((last, m.start()))
        last = m.end()
    parts.append((last, len(sentence)))
    for i, (a, b) in enumerate(parts):
        if a == clause_a and i > 0:
            return parts[i - 1]
    return (None, None)


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
    """The benchmark name nearest the number, excluding org names —
    the benchmark the claim belongs to, when the sentence names one.
    Recorded on the claim and (v1.8.7) part of the identity: two
    scores on different benchmarks are different claims, even for the
    same model. A benchmark named in one doc and unnamed in another
    is the same claim — the compare loop's benchmark merge attaches
    the unnamed side when the values match. The nearest name wins,
    not the first in the sentence: in a long sentence that mentions
    several benchmarks, the first one is arbitrary (\"about 55% fewer
    blocks ... its ProgramBench result\" — the 55% is not a
    ProgramBench score).
    v1.8.9 (run-20 gaps 2b/4/6): hyphenated and all-caps benchmark
    names (\"Terminal-Bench\", \"SWE-sweep\") never match the CamelCase
    pattern, so the resolver fell through to whatever CamelCase token
    sat elsewhere in the sentence — 65.2% on Terminal-Bench 4.1 was
    labeled FrontierCode, SWE-sweep scores were bench-less, and
    clarification labels pooled adjacent benchmarks. Phrase-list
    names now compete on the same distance rule as CamelCase tokens.
    Distance alone is not enough, though: in \"65.2% on
    Terminal-Bench 4.1 and 46.2% at high on FrontierCode 2.0\" the
    nearer name to 46.2% is Terminal-Bench (behind it), while the
    number's own phrase says FrontierCode — so attachment beats
    distance. A name the number's own prepositional phrase points
    to (\"N% on Benchmark\") wins first; then a name the number sits
    in parentheses after (\"Benchmark (N%)\"); then the nearest
    benchmark named BEFORE the number, longest wins a tie
    (\"Terminal-Bench-Science\" over \"Terminal-Bench\") — a benchmark
    named after the number never owns it (v1.8.8 r2 pinned rule)."""
    def _name_spans(s: str):
        spans = []
        for m in _GENERIC_CAMEL_RE.finditer(s):
            if m.group(0).lower() in _GENERIC_BENCH_ORG_STOPLIST:
                continue
            spans.append((m.start(), m.end(), m.group(0)))
        for m in _GENERIC_BENCHMARK_PHRASE_RE.finditer(s):
            hit = m.group(0)
            canonical = ""
            for p in _GENERIC_BENCHMARK_PHRASES:
                if p.lower() == hit.lower():
                    canonical = p
                    break
            spans.append((m.start(), m.end(), canonical or hit))
        # Position order: the attachment scan below breaks at 40 chars.
        return sorted(spans)

    # (1) Attachment: the number's own "on/in/at/for <Name>" phrase.
    # The phrase must sit in the number's own clause — a ";" ends
    # the clause, so "scores 77.2 against 76.2; on Terminal-Bench
    # 2.0" never reads Terminal-Bench as the 76.2's benchmark
    # (v1.8.8 r2 pinned rule).
    after = sentence[num_start:num_start + 60].split(";")[0]
    for start, end, name in _name_spans(after):
        if start > 40:
            break
        before_name = after[max(0, start - 15):start]
        if re.search(r"\b(?:on|in|at|for)\s+(?:the\s+)?$", before_name, re.I):
            return name
    # (1b) Immediately after: "the 84.0% RiemannBench result" — the
    # benchmark name directly follows the number in the same clause.
    # A comma, semicolon, or any other token between the number and
    # the name means the name belongs elsewhere, and a model token
    # ("84.0% Fable 5.2") is a measured system, not a benchmark —
    # both fall through to the later tiers.
    m_num = re.match(r"[+-]?\$?[\d,]+(?:\.\d+)?\s*(?:%|bps)?", sentence[num_start:])
    if m_num:
        rest = sentence[num_start + m_num.end():]
        ws = re.match(r"\s+", rest)
        rest2 = rest[ws.end():] if ws else rest
        if rest2 and not _MODEL_TOKEN_RE.match(rest2):
            for start, end, name in _name_spans(rest2):
                if start == 0:
                    return name
    # (2) Parenthesis: "Benchmark (N%)" — the name right before "(".
    before = sentence[max(0, num_start - 60):num_start]
    paren_best = ""
    paren_end = -1
    for start, end, name in _name_spans(before):
        if re.match(r"\s*\(\s*$", before[end:]):
            if end > paren_end or (end == paren_end and len(name) > len(paren_best)):
                paren_end = end
                paren_best = name
    if paren_best:
        return paren_best
    # (3) Nearest by distance, longest wins a tie — but ONLY among
    # benchmarks named BEFORE the number (v1.8.8 r2 pinned rule: a
    # benchmark named after a number never owns it — "scores 77.2
    # against 76.2; on Terminal-Bench 2.0" must read the 76.2 as
    # SWE-bench, not Terminal-Bench, even though Terminal-Bench is
    # nearer). The after-the-number route is attachment (tier 1:
    # "N% on Benchmark") and parenthesis (tier 2) only.
    best = ""
    best_dist = 10 ** 9
    best_len = 0
    for start, end, name in _name_spans(sentence):
        if end > num_start:
            continue
        dist = num_start - end
        if dist < best_dist or (dist == best_dist and len(name) > best_len):
            best_dist = dist
            best = name
            best_len = len(name)
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


# v1.8.9 (run-20 gap 14): a stated uncertainty is not a gap — "plus
# or minus 0.94 percentage points" is the published margin on a
# score, not a percentage-point gap between two figures. The
# uncertainty value is the number right after "plus or minus" / "±"
# / "margin of error", or the number in an "uncertainty" sentence
# followed by "percentage point(s)".
_GENERIC_UNCERTAINTY_BEFORE_RE = re.compile(
    r"(?:plus\s+or\s+minus|±|margin\s+of\s+error(?:\s+of)?)\s*$", re.I)
_GENERIC_UNCERTAINTY_SENT_RE = re.compile(r"\buncertainty\b", re.I)


def _generic_rank(sentence: str, num_start: int) -> Optional[int]:
    """A stated rank for the number, from a rank cue just before it
    in the same sentence ("debuts #2 globally with a score of 1336",
    "second at 70.87%", "moved Neon High to third overall at
    1,333"). The nearest cue wins; None when the sentence states no
    rank for this figure. Only the cue, never a derivation — a rank
    the sentence does not state stays unstated."""
    window = sentence[max(0, num_start - 45):num_start]
    best: Tuple[int, int] = (-1, 0)
    for m in re.finditer(r"#(\d+)\b", window):
        if m.start() > best[0]:
            best = (m.start(), int(m.group(1)))
    for m in re.finditer(
            r"\brank(?:ed|s|ing)?\s+#?(\d+)\b", window, re.I):
        if m.start() > best[0]:
            best = (m.start(), int(m.group(1)))
    for m in re.finditer(
            r"\brank(?:ed|s)?\s+(first|second|third|fourth|fifth)\b",
            window, re.I):
        if m.start() > best[0]:
            best = (m.start(), _GENERIC_RANK_WORDS[m.group(1).lower()])
    for m in re.finditer(
            r"\b(first|second|third|fourth|fifth)\b", window, re.I):
        if m.start() > best[0]:
            best = (m.start(), _GENERIC_RANK_WORDS[m.group(1).lower()])
    return best[1] if best[0] >= 0 else None


_GENERIC_RANK_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def _generic_is_results_continuation(sentence: str) -> bool:
    """True when the sentence reads as a continuation of a results
    listing: a scoring verb ("scored", "ranked", "placed", "topped",
    "led", "put") with a model token in the sentence. The benchmark
    carryover only fires for such sentences (or for uncertainty
    claims) — a sentence that merely sits after a benchmark section
    ("Their subtask-weighted rewards were 39.0%...") is about its
    own measure, not the previous benchmark's, and must not inherit
    it."""
    if not _MODEL_TOKEN_RE.search(sentence):
        return False
    return bool(re.search(
        r"\b(scored|achieved|reached|ranked|placed|topped|led|put|puts)\b",
        sentence, re.I))


def _generic_prior_benchmark(sentences: List[str], sent_idx: int,
                             window: int = 5) -> str:
    """The benchmark named in the sentences just before this one, or
    "" when none is named. A results continuation ("Fable 5.2 scored
    71.31%, Sonnet 5.5 scored 70.85%") inherits the benchmark of the
    sentence that introduced the results ("... 70.87% on Vals Index
    v2.2") — the scores are one ranking, split across sentences. The
    nearest benchmark-naming sentence decides: scan backwards and
    stop at the first sentence that names any benchmark — exactly
    one names it, two or more is a genuine ambiguity and nothing is
    inherited. The window is five sentences because the sentence
    splitter fragments abbreviations ("U.S. GDP." becomes two
    fragments), so the stated uncertainty two real sentences on can
    sit four fragments back. Only the curated phrase list counts (a
    CamelCase token is not enough here), and only when the current
    sentence names no benchmark of its own — a doc that moves on to
    another benchmark names it, and the scan stops there instead of
    reaching past it."""
    for j in range(sent_idx - 1, max(-1, sent_idx - 1 - window), -1):
        named = set()
        for p in _GENERIC_BENCHMARK_PHRASES:
            if re.search(r"\b" + re.escape(p) + r"\b", sentences[j], re.I):
                named.add(p)
        if named:
            return next(iter(named)) if len(named) == 1 else ""
    return ""


_GENERIC_EFFORT_RE = re.compile(r"^\s*(extra high|high|medium|low)\b", re.I)


def _generic_effort(sentence: str) -> str:
    """Thinking-level tier for a duration claim in a model table row
    ("Gemini 4 Neon Extra High: 61, $6.44 per task, 21.5 minutes" —
    the tier sits right after the model token). Returns the tier slug
    ("extra-high", "high", "medium", "low") or "" when no tier word
    follows a model token in the sentence."""
    for m in _MODEL_TOKEN_RE.finditer(sentence):
        after = sentence[m.end():m.end() + 20]
        em = _GENERIC_EFFORT_RE.match(after)
        if em:
            return em.group(1).lower().replace(" ", "-")
    return ""


# v1.8.9 (run-20 gap 11): a multiplier / speed-ratio claim — "three
# times as quickly as Carbon", "3x faster than Carbon", "twice as
# fast". _generic_numbers_in deliberately drops multipliers ("five
# times" is context, not a claim), so this pass scans sentences
# directly and mints its own claim kind instead of going through
# the number loop. A multiplier is a headline exaggeration metric
# by construction: it carries info_weight "low" and signal_role
# "bias" and never enters factual alignment — it only enters the
# cross-unit speed-ratio check, where measured durations either
# confirm it or catch it.
_GENERIC_MULTIPLIER_WORDS = {
    "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "twice": 2.0,
}
_GENERIC_MULTIPLIER_RE = re.compile(
    r"\b(?:(about|approximately|roughly|nearly|around)\s+)?"
    r"(one|two|three|four|five|six|seven|eight|nine|ten|twice|\d+(?:\.\d+)?)"
    r"\s+times\s+(?:as\s+)?(fast|faster|quickly|quick|slow|slower)\b",
    re.I)
_GENERIC_MULTIPLIER_X_RE = re.compile(
    r"\b(?:(about|approximately|roughly|nearly|around)\s+)?"
    r"(\d+(?:\.\d+)?)\s*x\s+(fast|faster|quickly|quick|slow|slower)\b",
    re.I)


def _generic_multiplier_claims(sentences: List[str]) -> List[Dict]:
    """Multiplier claims in the sentences: value, direction, measured
    subject (nearest model token before the phrase, else the
    sentence's exactly-one model), and comparator baseline (the model
    after "than"/"as" following the speed word). A multiplier with no
    stated baseline still mints — the speed-ratio check then reports
    it as uncheckable rather than silently dropping a headline
    exaggeration."""
    claims: List[Dict] = []
    seen = set()
    for si, sentence in enumerate(sentences):
        for rx in (_GENERIC_MULTIPLIER_RE, _GENERIC_MULTIPLIER_X_RE):
            for m in rx.finditer(sentence):
                qualifier = (m.group(1) or "").lower()
                word = m.group(2).lower()
                speed = m.group(3).lower()
                if word in _GENERIC_MULTIPLIER_WORDS:
                    value = _GENERIC_MULTIPLIER_WORDS[word]
                else:
                    try:
                        value = float(word)
                    except ValueError:
                        continue
                direction = ("faster" if speed in ("fast", "faster", "quickly", "quick")
                             else "slower")
                before = sentence[:m.start()]
                model_before = list(_MODEL_TOKEN_RE.finditer(before))
                subject_slug = ""
                subject_display = ""
                if model_before:
                    subject_display = model_before[-1].group(1)
                    subject_slug = _model_slug(subject_display)
                if not subject_slug:
                    sent_models = {_model_slug(mm.group(1)): mm.group(1)
                                   for mm in _MODEL_TOKEN_RE.finditer(sentence)}
                    if len(sent_models) == 1:
                        subject_slug = next(iter(sent_models))
                        subject_display = sent_models[subject_slug]
                after = sentence[m.end():m.end() + 40]
                baseline_slug = ""
                baseline_display = ""
                bm = re.search(
                    r"^\s*(?:than|as)\s+(" + _MODEL_TOKEN_RE.pattern + r")",
                    after)
                if bm:
                    baseline_display = bm.group(1)
                    baseline_slug = _model_slug(baseline_display)
                clause = re.split(r"[;,]|\bbut\b", before, flags=re.I)[-1]
                if sentence.rstrip().endswith("?"):
                    stance = "question"
                elif re.search(r"\b(?:not|never|isn['’]t|wasn['’]t|cannot|can['’]t)\b", clause, re.I):
                    stance = "negated_or_qualified"
                elif re.search(r"\b(?:if|could|might|would|whether|hypothetically)\b", clause, re.I):
                    stance = "conditional"
                else:
                    stance = "asserted"
                dedup = (si, m.start(), value, subject_slug, baseline_slug, direction, stance)
                if dedup in seen:
                    continue
                seen.add(dedup)
                raw = m.group(0).strip()
                label = (f"{value:g}x {direction} than "
                         f"{baseline_display or 'unstated baseline'}")
                if subject_display:
                    label = f"{label} — {subject_display}"
                claims.append({
                    "entity_key": f"generic|multiplier|speed|||{subject_slug or 'unstated'}",
                    "entity_label": label,
                    "numeric_value": float(value),
                    "unit": "x",
                    "raw_value": raw,
                    "version": "",
                    "shot_count": None,
                    "variant": "",
                    "claim_kind": "generic_quantitative",
                    "subject": subject_slug,
                    "subject_display": subject_display,
                    "metric": "speed",
                    "benchmark": "",
                    "context": "",
                    "context_numbers": "",
                    "denominator": "",
                    "rel_direction": "",
                    "qualifier": qualifier,
                    "time": "",
                    "attribution": "",
                    "span": sentence.strip()[:200],
                    "mention_span": sentence,
                    "multiplier": float(value),
                    "assertion_status": stance,
                    "sentence_index": si,
                    "expression_offset_in_sentence": m.start(),
                    "multiplier_direction": direction,
                    "multiplier_baseline": baseline_slug,
                    "multiplier_baseline_display": baseline_display,
                    "info_weight": "low",
                    "signal_role": "bias",
                })
    return claims


# ── Headline position (v1.8.9, run-20 follow-on 1) ─────────────────────
# Taylor's ruling: headlines are weighted LOW for information quality
# — most news runs on exaggeration metrics through headlines — and are
# better read as bias detection + target-audience signal. A headline
# figure is therefore a claim with info_weight "low", signal_role
# "bias", and a from_headline marker, routed out of factual alignment
# into headline_weighting (compare), never counted as agreement,
# conflict, or a unique.
#
# The gap-8 conflict, resolved: the v1.8.8 r2 gap-8 rule ("Source
# lines are labels, never claims") stays INTACT — the Source line is
# still dropped as a sentence source, so its own words (publisher
# names, dates, "Published"/"Fetched" metadata) never mint claims.
# What is "un-dropped" is only the quoted TITLE span inside the Source
# line (or the title half of a "<Publisher> — <Title>" source_name),
# and only as position metadata: figures are minted from the title
# text alone, by two targeted passes (multiplier phrases; numbers with
# a benchmark cue nearby). A headline figure can never be a factual
# claim — it is marked from_headline and routed to bias — so gap-8's
# intent (a Source line is not a factual claim) holds by
# construction. What the conflict costs: headline figures OUTSIDE the
# two patterns (e.g. a headlined price) are not captured; catching
# them would mean running the generic extractor over title text,
# which re-opens gap-8's exact failure mode (header numbers minting
# as claims) and needs an Arch decision.
_HEADLINE_BENCH_CUE_RE = re.compile(
    r"\b(?:Intelligence\s+(?:Index|Score)|Index|Score|Benchmark|Exam)\b",
    re.I)


def _headline_title(text: str, source_name: str = "") -> str:
    """The doc's headline (article title), or "" when the doc carries
    none. First the quoted title in the first-line Source line
    ('Source: SiliconANGLE — "Google's latest ..." by ...'), else the
    title half of a "<Publisher> — <Title>" source_name. A bare
    publisher name (or "doc_N") yields no title — never guessed."""
    first_line = (text or "").split("\n", 1)[0]
    if first_line.strip().lower().startswith("source:"):
        m = re.search(r'"([^"]{4,200})"', first_line)
        if m:
            return m.group(1).strip()
    if source_name:
        # Separator must carry whitespace on both sides (" — ", " - "),
        # so a hyphenated publisher name ("AI-News") never splits.
        parts = re.split(r"\s+[—–]\s+|\s+-\s+", source_name, maxsplit=1)
        if len(parts) == 2 and len(parts[1].strip()) >= 4:
            return parts[1].strip()
    return ""


def _without_metadata(text: str) -> str:
    return "\n".join("" if _GENERIC_FEED_HEADER_LINE_RE.match(line.strip()) else line
                     for line in text.splitlines())


def headline_diagnostic(text: str, source_name: str) -> List[Dict]:
    title = _headline_title(text, source_name)
    entries = [{"doc_idx": 0, "claim": c} for c in _headline_figures(title)] if title else []
    return _headline_weighting(entries, lambda _: source_name)


def _era_suggestions(text: str, source: str, doc_idx: int) -> List[Dict]:
    """Accept only a complete price sentence and an adjacent explicit same-model rate reference."""
    money = r"\$\d+(?:\.\d+)?"
    month = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    date = month + r" (?:[12]\d|3[01]|[1-9])(?:, \d{4})?"
    price = re.compile(r"(?P<model>" + _MODEL_TOKEN_RE.pattern + r") (?:costs|is priced at) "
                       + money + r" per (?:task|million (?:input|output) tokens)\.", re.I)
    candidates = []
    for paragraph in re.split(r"\n\s*\n", text):
        sentences = [v.strip() for v in _GENERIC_SENT_SPLIT_RE.split(paragraph) if v.strip()]
        for statement, context in zip(sentences, sentences[1:]):
            match = price.fullmatch(statement)
            if not match:
                continue
            name = match.group("model")
            linked = re.fullmatch(r"That " + re.escape(name) + r" rate (?:will take|takes) effect on ("
                                  + date + r")\.", context, re.I)
            if linked:
                candidates.append(dict(document_index=doc_idx, source=source, subject=name,
                                       era="unknown", suggested_era="future", effective_date=linked[1],
                                       price_span=statement, context_span=context, review_required=True,
                                       basis="One price and one explicit same-model rate reference"))
    return candidates


def _span_gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Character gap between two spans (0 when adjacent/overlapping)."""
    if b_end <= a_start:
        return a_start - b_end
    if a_end <= b_start:
        return b_start - a_end
    return 0


def _headline_score_claims(title: str) -> List[Dict]:
    """Benchmark-score figures in the headline: a number with a
    benchmark cue nearby ("61 Intelligence Score", "62 on Artificial
    Analysis"). The cue is a listed benchmark phrase or a bare
    Intelligence Index/Score/Index/Benchmark/Exam word; the NEAREST
    cue to the number wins, so "SWE-bench 76.2, Terminal-Bench 59.3"
    keys each score to its own benchmark. Skipped: 4-digit years,
    version tokens ("Gemini 4"), numbers glued to a preceding letter
    ("Work360"), and magnitudes other than "%" ("$3", "2 million" are
    not benchmark scores)."""
    claims: List[Dict] = []
    seen = set()
    for m in _GENERIC_DIGIT_RE.finditer(title):
        digits = m.group(2)
        if re.fullmatch(r"\d{4}", digits) and 1900 <= int(digits) <= 2099:
            continue
        if _generic_is_version(title, m.start(), m.end()):
            continue
        if m.start() > 0 and title[m.start() - 1].isalpha():
            continue
        magnitude = (m.group(3) or "").strip()
        if magnitude and magnitude != "%":
            continue
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue
        start, end = m.start(), m.end()
        win_start = max(0, start - 40)
        window = title[win_start:end + 40]
        best = None  # (distance, cue_text, canonical_phrase_or_"")
        for p in _GENERIC_BENCHMARK_PHRASES:
            for pm in re.finditer(r"\b" + re.escape(p) + r"\b", window, re.I):
                dist = _span_gap(start, end,
                                 win_start + pm.start(), win_start + pm.end())
                if best is None or dist < best[0]:
                    best = (dist, pm.group(0), p)
        for cm in _HEADLINE_BENCH_CUE_RE.finditer(window):
            dist = _span_gap(start, end,
                             win_start + cm.start(), win_start + cm.end())
            if best is None or dist < best[0]:
                best = (dist, cm.group(0), "")
        if best is None:
            continue
        bench_display = best[2] or best[1]
        bench_slug = re.sub(r"[^a-z0-9]+", "-", bench_display.lower()).strip("-")
        before = title[:start]
        model_before = list(_MODEL_TOKEN_RE.finditer(before))
        subject_slug = ""
        subject_display = ""
        if model_before:
            subject_display = model_before[-1].group(1)
            subject_slug = _model_slug(subject_display)
        if not subject_slug:
            sent_models = {_model_slug(mm.group(1)): mm.group(1)
                           for mm in _MODEL_TOKEN_RE.finditer(title)}
            if len(sent_models) == 1:
                subject_slug = next(iter(sent_models))
                subject_display = sent_models[subject_slug]
        dedup = (value, bench_slug, subject_slug)
        if dedup in seen:
            continue
        seen.add(dedup)
        label = bench_display
        if not re.search(r"\b(score|index|benchmark|exam)$", bench_display, re.I):
            label = f"{bench_display} score"
        if subject_display:
            label = f"{label} — {subject_display}"
        claims.append({
            "entity_key": f"generic|headline|score|bench={bench_slug}||{subject_slug or 'unstated'}",
            "entity_label": label,
            "numeric_value": float(value),
            "unit": "percent" if magnitude == "%" else "",
            "raw_value": m.group(0).strip(),
            "version": "",
            "shot_count": None,
            "variant": "",
            "claim_kind": "generic_quantitative",
            "subject": subject_slug,
            "subject_display": subject_display,
            "metric": "score",
            "benchmark": bench_slug,
            "context": bench_display,
            "context_numbers": f"bench={bench_slug}",
            "denominator": "",
            "rel_direction": "",
            "qualifier": "",
            "time": "",
            "attribution": "",
            "span": title.strip()[:200],
            "headline": title,
            "from_headline": True,
            "info_weight": "low",
            "signal_role": "bias",
        })
    return claims


def _headline_figures(title: str) -> List[Dict]:
    """Figures stated in the headline itself: multiplier phrases
    (reusing _generic_multiplier_claims on the title alone) and
    benchmark scores (_headline_score_claims). Every figure carries
    from_headline + info_weight low + signal_role bias. The generic
    number loop never sees the title (gap-8: the Source line is
    dropped as a sentence source), so these two targeted passes are
    the only way headline figures surface — and they surface as bias
    signal, never as facts."""
    figures: List[Dict] = []
    if not title or not title.strip():
        return figures
    for c in _generic_multiplier_claims([title]):
        figures.append({**c, "from_headline": True, "headline": title})
    figures.extend(_headline_score_claims(title))
    return figures


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
    sentences = [s for s in _GENERIC_SENT_SPLIT_RE.split(_without_metadata(body)) if s.strip()]
    # v1.8.8 r2: drop feed-header label lines (gap 8) before the
    # sentence loop — a "Source: ..." line is a label, never a claim.
    sentences = [s for s in sentences
                 if not _GENERIC_FEED_HEADER_LINE_RE.match(s.strip())]
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
    # v1.8.8: the table rows, parsed once and reused by the table-row
    # pass below, plus the line label each sentence sits on — a bare
    # "Model N%" fragment ("Fable 5.1 31.4% matches") on a benchmark
    # row line is that row's score (item 3 — the run-11 medium
    # coverage gap: the row parser only reads a row's first clause,
    # so later-clause fragments never mint).
    table_rows = _parse_table_rows(text)
    sent_line_bench: List[str] = []
    for s in sentences:
        s_clean = s.strip().lstrip("-•* ")
        bench = ""
        for row in table_rows:
            if s_clean and (s_clean in row["line"] or row["line"].startswith(s_clean[:120])):
                bench = row["benchmark_name"]
                break
        sent_line_bench.append(bench)
    claims: List[Dict] = []
    _sent_pair_tiers: Dict[int, Dict[int, str]] = {}  # per-sentence pricing-pair tiers (v1.8.9 gap 2c)
    for si, sentence in enumerate(sentences):
        numbers = _generic_numbers_in(sentence)
        prev_subject: Tuple[str, str] = ("", "")
        prev_num: Optional[Dict] = None
        for num in numbers:
            if (si, num["value"]) in existing_pairs:
                continue
            if num["start"] > 0 and sentence[num["start"] - 1] == "#":
                # v1.8.9 (run-20 gap 12): "#2" is a rank, not a
                # count — "debuts #2 globally" and "ranks #2 on
                # steerability" minted as task counts and pooled
                # with real scores. A hash immediately before the
                # digits marks a ranking, never a measurement.
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
            line_bench = ""
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
                    # v1.8.8 line-label fallback: a bare "Model N%"
                    # fragment on a table-row line is that row's
                    # benchmark score ("Fable 5.1 31.4% matches" on
                    # the AutomationBench line). Only when the
                    # sentence names no benchmark of its own — the
                    # AA-Briefcase guard: a summary line's own
                    # benchmark ("RoboDojo Sim: Astra 29.8%") wins
                    # over the row label, and Region 0.5 already
                    # reads that one as a score.
                    if (sent_line_bench[si] and num.get("unit") == "percent"
                            and _MODEL_TOKEN_RE.search(before20)
                            and not _generic_benchmark_phrase(sentence)
                            and not _generic_benchmark(sentence, num["start"])):
                        family = "score"
                        line_bench = sent_line_bench[si]
                if not family:
                    continue
            subject_slug, subject_display = _generic_subject(sentence, num, numbers)
            if not subject_slug:
                # v1.8.9 (run-20 gap 2a follow-on): a sentence naming
                # exactly one model states that model's figures —
                # "Gemini 4 Carbon, introductory: $2.00 in, $10.00
                # out, best index score 60." is Carbon's 60, but the
                # clause-scoped subject check never sees the model
                # (it sits before an earlier number in the sentence),
                # so the 60 pooled bench-less with Neon's 55. Two
                # named models stay ambiguous, as before.
                _sent_models: Dict[str, str] = {}
                # Worker B (v1.8.9 correction): a model token after
                # the next number is skipped only when it is that
                # later number's "for <Model>" owner ("refrained ...
                # in 99.93% of cases, compared to 94.49% for GPT-5.6
                # Cyber" — GPT-5.6 Cyber is the 94.49%'s model, and
                # giving it to the 99.93% misattributes the figure;
                # "reached 40.3% recall versus 32.0% for GPT-6
                # Astra" — Astra is the 32.0%'s baseline, not the
                # 40.3%'s subject). A model named after a cost +
                # effort label ("45.5% at $1.26 per test for Neon
                # Low") labels the whole pair and still counts — the
                # blanket "after the next number" skip wrongly
                # unstated Neon Low's 45.5%. The comparator
                # exclusion below also covers the "compared to
                # <num>% for <Model>" shape; this check is the
                # belt-and-braces for it.
                _next = min(
                    [n for n in numbers if n["start"] > num["end"]],
                    key=lambda n: n["start"], default=None)
                for _mm in _MODEL_TOKEN_RE.finditer(sentence):
                    # Worker B (v1.8.9 correction, Nov 7 48.5 — ported
                    # from repo-work; the RC3 port dropped it): a model
                    # token after the next number is that later
                    # number's "for <Model>" owner ("the new model
                    # achieved 48.5%, up from 32.1% for GPT-5.6
                    # Cyber" — GPT-5.6 Cyber owns the 32.1%, not the
                    # 48.5%), so it never counts toward the
                    # exactly-one set for this number. The figure
                    # mints unstated and the compare merge attaches it
                    # to the one value-matching stated group — or
                    # leaves it unstated when no group matches, never
                    # guessed. Only the bare "% for " / "for " gap
                    # fires: a cost + effort label in between ("45.5%
                    # at $1.26 per test for Neon Low") still counts.
                    if _next is not None and _mm.start() >= _next["start"]:
                        between = sentence[_next["end"]:_mm.start()]
                        if re.fullmatch(r"\s*(?:%|percent)?\s*for\s+",
                                        between, re.I):
                            continue
                    # A model named only as a comparator ("32% lower
                    # cost per task than Carbon", "faster than Opus",
                    # "9.6 points above Argon") is the thing measured
                    # AGAINST, not the measured subject — it never
                    # counts toward the exactly-one set, or Neon's 32%
                    # would read as Carbon's. Likewise a model named
                    # only as a parenthetical price gloss ("(Argon:
                    # $9.95)") labels that price's owner, not the
                    # sentence's subject for other numbers — but a
                    # "Model: <figures>" header ("Gemini 4 Hydrogen:
                    # $1.50 in, $7.50 out, best index score 54.")
                    # DOES own the figures that follow it, so the
                    # colon exclusion only fires inside parentheses.
                    # Worker B: the comparator may sit a number and a
                    # "for" away ("compared to 94.49% for GPT-5.6
                    # Cyber") — that model is still the baseline.
                    _before = sentence[max(0, _mm.start() - 40):_mm.start()]
                    if re.search(r"\b(?:than|vs\.?|against|versus|above|below|compared\s+(?:with|to))\s*$",
                                 _before, re.I):
                        continue
                    if re.search(r"\b(?:than|vs\.?|against|versus|above|below|compared\s+(?:with|to))\b(?:[\d\s.,%]*\bfor\b)?\s*$",
                                 _before, re.I):
                        continue
                    if re.match(r"\s*:", sentence[_mm.end():_mm.end() + 4]) and \
                            sentence.rfind("(", 0, _mm.start()) > sentence.rfind(")", 0, _mm.start()):
                        continue
                    _sent_models[_model_slug(_mm.group(1))] = _mm.group(1)
                if len(_sent_models) == 1:
                    subject_slug = next(iter(_sent_models))
                    subject_display = _sent_models[subject_slug]
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
            if family == "cost" and not denominator and num.get("currency"):
                # v1.8.9 (run-20 gap 2c): a $ amount's pricing tier
                # is its denominator. "$2.00 in" is Carbon's intro
                # INPUT price and "$2.12" is an AutomationBench cost
                # per run; both minted as bare generic cost with
                # denom "", so they falsely aligned. The tier comes
                # from the positional pair rule on the sentence, then
                # the in/out abbreviation right after the amount
                # ("$2.00 in, $10.00 out" — the full-word pair rule
                # never sees "in"/"out"). The abbreviation must be
                # followed by a list separator, not prose ("$2.00 in
                # the first week" is not an input price).
                if si not in _sent_pair_tiers:
                    _sent_pair_tiers[si] = _pricing_pair_tiers(sentence)
                tier = _sent_pair_tiers[si].get(num["start"] + 1, "")
                if not tier:
                    after = sentence[num["end"]:num["end"] + 12]
                    ab = re.match(r"\s*(in|out|cached|cache|batch)\b\s*(?:[,/]|and\b|$)",
                                  after, re.I)
                    if ab:
                        tier = _DOT_LIST_TIER_WORDS[ab.group(1).lower()]
                if tier:
                    denominator = tier
            qualifier = _generic_qualifier(sentence, num["start"])
            benchmark = _generic_benchmark(sentence, num["start"])
            if family == "score":
                # v1.8.8: a score carries the benchmark its clause
                # names when the CamelCase detector misses it ("CVE
                # Rediscovery", "Terminal-Bench-Science"), so it keys
                # exactly onto the table-row group instead of
                # attaching through the merge — and the claim's
                # benchmark field says what it is a score on. A
                # phrase that EXTENDS the CamelCase token ("RoboDojo
                # Sim" over "RoboDojo") is the fuller name and wins.
                clause_a, clause_b = _generic_clause_span(sentence, num["start"], num["end"])
                clause_bench = _generic_benchmark_phrase(sentence[clause_a:clause_b])
                if clause_bench and (not benchmark or clause_bench.startswith(benchmark)):
                    benchmark = clause_bench
                if not benchmark:
                    # The benchmark may sit in another clause of the
                    # same sentence ("... in its internal CVE
                    # Rediscovery evaluation, compared with 78.3% for
                    # Mythos 5.1" — the 78.3% is a figure in that
                    # evaluation). Use the sentence's benchmark only
                    # when it names exactly one — two named benchmarks
                    # is a genuine ambiguity, left bench-less for the
                    # merge. The merge's ambiguity guard refuses to
                    # guess when a near value sits on another
                    # benchmark (78.3 vs protein 79.1), so a score
                    # left bench-less here never aligns.
                    # v1.8.8 r2: only benchmarks named BEFORE the
                    # number count — a benchmark named in a LATER
                    # clause belongs to a later figure, and using it
                    # mislabels this one ("... scores 77.2 against
                    # 76.2; on Terminal-Bench 2.0, 59.3 ..." gave the
                    # SWE-bench 76.2 a Terminal-Bench key, a false
                    # conflict; "out of 100 ... OSWorld-Verified ..."
                    # gave a scale marker an OSWorld key).
                    before = sentence[:num["start"]]
                    named = set()
                    for m in _GENERIC_CAMEL_RE.finditer(before):
                        if m.group(0).lower() not in _GENERIC_BENCH_ORG_STOPLIST:
                            named.add(m.group(0))
                    for p in _GENERIC_BENCHMARK_PHRASES:
                        if re.search(r"\b" + re.escape(p) + r"\b", before, re.I):
                            named.add(p)
                    if len(named) == 1:
                        benchmark = next(iter(named))
            if line_bench:
                # v1.8.8 line-label fallback: the row's benchmark is
                # the claim's benchmark.
                benchmark = line_bench
            if not benchmark and family in ("score", "uncertainty"):
                # v1.8.9 (run-20 gap 14): the benchmark may sit in the
                # sentence that introduced the results, not this one —
                # "Fable 5.2 scored 71.31%, Sonnet 5.5 scored 70.85%"
                # is the continuation of the Vals ranking the previous
                # sentence opened ("... 70.87% on Vals Index v2.2"),
                # and the uncertainty sentence two lines on ("Its
                # published uncertainty for Neon is plus or minus
                # 0.94...") belongs to the same ranking. Two gates, so
                # a nearby benchmark never owns an unrelated figure:
                # the current sentence must name NO benchmark phrase
                # of its own (a sentence naming two — "Artificial
                # Analysis's separate Coding Agent Index" — is
                # ambiguous in itself, and reaching back past it to
                # an older benchmark is wrong), and it must be a
                # results continuation (scoring verb + model token)
                # or an uncertainty claim — a sentence that merely
                # sits after a benchmark section ("Their subtask-
                # weighted rewards were 39.0%...") is about its own
                # measure and inherits nothing.
                # v1.8.9: never carry a benchmark into a sentence that
                # names its own benchmark in words the phrase list does
                # not know ("this Gray Swan benchmark") — the carried
                # benchmark would be factually wrong AND it breaks the
                # subject-merge chain (the bench-less Gray Swan 0.0 is
                # the one value-match that routes the 0.09% safety
                # figure into the Terminal-Bench Science mismatch
                # group; benching it left the 0.09 stranded in the
                # bench-less pool, where the modal paired it with
                # Kingy's 10.1% as a false conflict).
                if not _generic_benchmark_phrase(sentence) and \
                        not re.search(r"\bbenchmark\b", sentence, re.I) and \
                        (family == "uncertainty"
                         or _generic_is_results_continuation(sentence)):
                    benchmark = _generic_prior_benchmark(sentences, si)
            attribution = _generic_attribution(sentence, doc_attribution)
            # v1.8.9 (run-20 gap 11): a duration claim in a model
            # table row carries its thinking-level tier (Neon Extra
            # High 21.5 min vs Neon High 14.8 min are different
            # durations, not one claim) — the tier rides in the ctx
            # slot as effort=, the same way bench= does.
            effort = _generic_effort(sentence) if family == "duration" else ""
            # v1.8.9 (run-20 gap 14): a stated rank rides on the
            # claim so the separability check can stack score +
            # uncertainty + rank + source direction instead of
            # collapsing to one verdict.
            rank = _generic_rank(sentence, num["start"]) if family == "score" else None
            # v1.8.9 (run-20 gap 10 residual): a board score whose
            # sentence says the board MOVED ("Arena's November 18
            # board moved Neon High to third overall at 1,333") is a
            # revision of the earlier board value, not a second
            # measurement that agrees with it. The marker rides on
            # the claim; compare_documents surfaces the revision
            # instead of aligning it.
            revision = False
            revision_cue = ""
            revision_date = ""
            if family == "score" and re.search(
                    r"\b(?:arena|board|leaderboard)\b", sentence, re.I):
                _rev_m = re.search(
                    r"\b(board\s+moved|updated|revised|revisions?|now\s+at|"
                    r"dropped\s+to|fell\s+to|rose\s+to|climbed\s+to|"
                    r"slipped\s+to|moved\s+(?:\w+\s+)?to)\b",
                    sentence, re.I)
                if _rev_m:
                    revision = True
                    revision_cue = _rev_m.group(1).lower()
                    _date_m = re.search(
                        r"\b(January|February|March|April|May|June|July|August|"
                        r"September|October|November|December)\s+(\d{1,2})\b",
                        sentence)
                    if _date_m:
                        revision_date = f"{_date_m.group(1)} {_date_m.group(2)}"
            # v1.8.7: the benchmark is part of the identity — two
            # scores on different benchmarks are different claims,
            # even for the same model (the run-11 table rows all share
            # the key "generic|percentage|score|||fable-5-2" and
            # collapse into one benchmark-mismatch group). The
            # benchmark slug rides in the ctx slot, ahead of any
            # context numbers.
            bench_slug = re.sub(r"[^a-z0-9]+", "-", benchmark.lower()).strip("-") if benchmark else ""
            # v1.8.8, Fix 1: two more identity dimensions ride in the
            # ctx slot. "rel" — the number is a RELATIVE delta ("22%
            # less per task"), not an absolute rate ("92% of tasks"):
            # a delta and a rate are different claims even when they
            # share a metric. "plan" — a currency amount that IS a
            # subscription plan's price ("the $19.99 plan") is a
            # different claim role from a per-unit price ("$8 input").
            # Both markers read the words immediately after the number
            # (within 12 chars), so a later sentence's wording never
            # re-qualifies this one.
            # v1.8.9 (run-20 gap 12): the direction rides with the
            # rel marker as a separate claim field, NOT in the key —
            # the pinned test contract (v1.8.8) keys relative deltas
            # as bare "rel" ("generic|percentage|speed|nodes=30;
            # turns=200;rel||unstated"). "25% cheaper per token" and
            # "50% more per token" are opposite directions of the
            # same metric and pooled into one false conflict as bare
            # "rel", so the compare partitions mixed-direction rel
            # groups by rel_direction instead (see _part_key below).
            # Synonyms normalize: less/lower/cheaper/fewer/smaller/
            # slower/worse/below -> down; more/higher/costlier/
            # greater/faster/better/above -> up. Same-direction
            # deltas ("32% lower" vs "17% lower") still share a key
            # and compare, which is what gap 8 needs.
            rel_marker = ""
            rel_direction = ""
            if num.get("unit") == "percent":
                after_num = sentence[num["end"]:num["end"] + 12]
                rel_m = re.match(
                    r"\s*[+−-]?\s*(less|more|fewer|higher|lower|above|below|cheaper|costlier|greater|smaller|faster|slower|better|worse)\b",
                    after_num, re.I)
                if rel_m:
                    rel_marker = "rel"
                    rel_direction = ("down" if rel_m.group(1).lower() in
                                     ("less", "fewer", "lower", "below", "cheaper",
                                      "smaller", "slower", "worse")
                                     else "up")
            plan_marker = ""
            if num.get("currency"):
                after_num = sentence[num["end"]:num["end"] + 12]
                if re.match(r"\s*plan\b", after_num, re.I):
                    plan_marker = "plan"
            ctx_parts = ([f"bench={bench_slug}"] if bench_slug else []) + \
                ([f"effort={effort}"] if effort else []) + \
                [p for p in (ctx_key, rel_marker, plan_marker) if p]
            ctx_key = ";".join(ctx_parts)
            denom_key = ""
            if family == "cost" and denominator:
                dlow = denominator.lower()
                if dlow.startswith("per"):
                    # v1.8.9 (run-20 gap 8): "per task than", "per
                    # completed task" and "per task" are the same
                    # denominator — the unit is the last known
                    # per-unit word in the phrase ("task", "token",
                    # "run", "test", "million"); adjectives
                    # ("completed") and trailing comparators ("than")
                    # drop out, or the same 32% mints under three
                    # different keys and never compares. No known
                    # unit ("per LPE exploit") keeps the full slug.
                    _words = re.findall(r"[a-z]+", dlow)
                    _unit = next((w for w in reversed(_words)
                                  if w in ("task", "token", "run", "test", "million")), "")
                    denom_key = (f"per-{_unit}" if _unit else
                                 re.sub(r"[^a-z0-9]+", "-", dlow).strip("-"))
                elif dlow in ("input", "output", "cached", "batch",
                              "batch-input", "batch-output",
                              "fast-input", "fast-output"):
                    denom_key = re.sub(r"[^a-z0-9]+", "-", dlow).strip("-")
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
                "rel_direction": rel_direction,
                "qualifier": qualifier,
                "time": year_m.group(0) if year_m else "",
                "attribution": attribution,
                "span": sentence.strip()[:200],
                "effort": effort,
                "rank": rank,
                "revision": revision,
                "revision_cue": revision_cue,
                "revision_date": revision_date,
            })
            prev_subject = (subject_slug, subject_display)
            prev_num = num
    # v1.8.9 (run-20 gap 11): multiplier / speed-ratio claims — minted
    # by their own pass (the number loop drops multipliers by design),
    # so a headline "three times faster" is a claim the speed-ratio
    # check can catch instead of invisible prose.
    claims.extend(_generic_multiplier_claims(sentences))
    # v1.8.9 (run-20 follow-on 1): headline figures — the doc's own
    # headline states figures the publisher chose to lead with. The
    # Source line stays dropped as a sentence source (gap-8); only the
    # quoted title span is read, and only by the two targeted passes
    # in _headline_figures, so a headline figure surfaces as bias
    # signal, never as a fact.
    _headline = _headline_title(text, source_name)
    if _headline:
        claims.extend(_headline_figures(_headline))
    # v1.8.7 table-row generic claims: a list-item benchmark row whose
    # benchmark is NOT in the mapped entity map (FrontierCode, CVE
    # Rediscovery, OSWorld, Terminal-Bench-Science) still states model
    # scores — mint them as generic score claims with the benchmark
    # name and the model subject, so the headline numbers surface
    # instead of vanishing. Mapped benchmarks are handled by the
    # mapped table-row pass; this pass only covers the unmapped ones.
    for row in table_rows:
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
            # v1.8.8: the sentence-loop span keeps the leading list
            # bullet ("- CVE Rediscovery: ...") while _parse_table_rows
            # returns the line without it — compare with the bullet
            # stripped on both sides, or the upgrade never fires and
            # the row mints a duplicate claim (the run-11 89.6%
            # double-mint: vulnerabilities AND score/bench).
            if c["numeric_value"] == float(row["value"]) and \
                    re.sub(r"^[-•*]\s+", "", c.get("span", "").strip())[:200] == row["line"][:200]:
                c["subject"] = subject
                c["subject_display"] = row["model_display"]
                c["benchmark"] = bench
                # v1.8.8: the upgrade changes the identity, so the
                # metric field must change with it — the sentence loop
                # may have minted this as "vulnerabilities" (the list
                # rule's first cue is the "CVE" in the row label)
                # while the key now says score.
                c["metric"] = "score"
                c["entity_key"] = (
                    f"generic|percentage|score|bench={bench_slug}||{subject or 'unstated'}")
                c["entity_label"] = f"{bench} score — {row['model_display']}"
                upgraded = True
                break
        if upgraded:
            continue
        # v1.8.8 r2: a bare row value ("- DeepSWE v1.1: 75.7") is a
        # measurement (bare decimal) or a task count (bare integer),
        # not a percentage — the role rides in the key so it never
        # lumps with a percentage claim of the same value.
        bare = not row["raw"].endswith("%")
        role = ("measurement" if "." in row["raw"] else "task_count") \
            if bare else "percentage"
        label = f"{bench} score — {row['model_display']}"
        claims.append({
            "entity_key": f"generic|{role}|score|bench={bench_slug}||{subject or 'unstated'}",
            "entity_label": label,
            "numeric_value": float(row["value"]),
            "unit": "count" if bare else "percent",
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
    if claim.get("effort"):
        entry["effort"] = claim["effort"]
    if claim.get("rank") is not None:
        entry["rank"] = claim["rank"]
    if claim.get("revision"):
        entry["revision"] = True
        entry["revision_cue"] = claim.get("revision_cue", "")
        entry["revision_date"] = claim.get("revision_date", "")
    if note:
        entry["note"] = note
    return entry


def _headline_weighting(headline_entries: List[Dict], doc_name_fn) -> List[Dict]:
    """Headline-position figures (v1.8.9, run-20 follow-on 1): per doc,
    the headline (title) and the figures it states, each marked
    from_headline + low info weight + bias role. A headlined figure is
    bias/audience signal — which number the publisher chose to lead
    with — not a factual claim: it never aligns, conflicts, or counts
    as a unique, and the body claim for the same figure is untouched
    (a doc is not punished for headlining its own lead fact)."""
    by_doc: Dict[int, List[Dict]] = {}
    for e in headline_entries:
        by_doc.setdefault(e["doc_idx"], []).append(e["claim"])
    out: List[Dict] = []
    for doc_idx in sorted(by_doc):
        claims = by_doc[doc_idx]
        figures = []
        for c in claims:
            fig: Dict = {
                "entity_key": c["entity_key"],
                "entity_label": c.get("entity_label", ""),
                "kind": ("multiplier" if c.get("multiplier") is not None
                         else "score"),
                "value": float(c["numeric_value"]),
                "unit": c.get("unit", ""),
                "subject": c.get("subject", ""),
                "subject_display": c.get("subject_display", ""),
                "benchmark": c.get("benchmark", ""),
                "from_headline": True,
                "info_weight": "low",
                "signal_role": "bias",
                "span": c.get("span", ""),
            }
            if c.get("multiplier") is not None:
                fig["multiplier_direction"] = c.get("multiplier_direction", "")
                fig["assertion_status"] = c.get("assertion_status", "asserted")
                fig["baseline"] = c.get("multiplier_baseline", "")
                fig["baseline_display"] = c.get("multiplier_baseline_display", "")
            figures.append(fig)
        out.append({
            "source": doc_name_fn(doc_idx),
            "headline": claims[0].get("headline", ""),
            "figures": figures,
            "note": ("Headline figures are bias/audience signal — which number the "
                     "publisher chose to lead with — weighted low for information "
                     "quality. They are never counted as factual agreement, and the "
                     "body claim for the same figure is unaffected."),
        })
    return out


def _speed_ratio_checks(multiplier_entries: List[Dict],
                        doc_claims: List[List[Dict]],
                        doc_name_fn) -> List[Dict]:
    """Group mentions by assertion and retain their sources. Compare declared
    effort tiers first. The provisional tolerance describes numerical distance
    from a claim; it cannot verify the claim or the underlying measurements.
    """
    ordered = sorted(multiplier_entries,
                     key=lambda e: bool(e["claim"].get("from_headline")))
    groups = {}
    for entry in ordered:
        claim = entry["claim"]
        key = (claim.get("subject", ""), claim.get("multiplier_baseline", ""),
               float(claim["numeric_value"]), claim.get("multiplier_direction", ""), claim.get("assertion_status"))
        groups.setdefault(key, []).append(entry)
    durations: Dict[Tuple[str, str], List[Dict]] = {}
    for doc_idx, claims in enumerate(doc_claims):
        for c in claims:
            if c.get("metric") != "duration" or not c.get("subject"):
                continue
            unit = (c.get("unit") or "").lower()
            if unit in ("minutes", "minute"):
                minutes = float(c["numeric_value"])
            elif unit in ("hours", "hour"):
                minutes = float(c["numeric_value"]) * 60.0
            elif unit in ("seconds", "second"):
                minutes = float(c["numeric_value"]) / 60.0
            else:
                continue
            durations.setdefault((c["subject"], c.get("effort", "")), []).append({
                "minutes": minutes, "document_index": doc_idx,
                "source": doc_name_fn(doc_idx),
                "span": c.get("span", ""),
            })

    def _fmt_ratio(r: float) -> str:
        return f"{r:.1f}x" if r < 10 else f"{r:.0f}x"

    checks: List[Dict] = []
    for mentions in groups.values():
        e = mentions[0]
        claim = e["claim"]
        subject = claim.get("subject", "")
        baseline = claim.get("multiplier_baseline", "")
        claimed = float(claim["numeric_value"])
        direction = claim.get("multiplier_direction", "faster")
        entry: Dict = {
            "entity_key": claim["entity_key"],
            "entity_label": claim.get("entity_label", ""),
            "multiplier": claimed,
            "multiplier_direction": direction,
            "subject": subject,
            "subject_display": claim.get("subject_display", ""),
            "baseline": baseline,
            "baseline_display": claim.get("multiplier_baseline_display", ""),
            "qualifier": claim.get("qualifier", ""),
            "source": doc_name_fn(e["doc_idx"]),
            "span": claim.get("span", ""),
            "from_headline": bool(claim.get("from_headline")),
            "info_weight": claim.get("info_weight", "low"),
            "signal_role": claim.get("signal_role", "bias"),
            "mentions": [{"document_index": m["doc_idx"], "source": doc_name_fn(m["doc_idx"]),
                          "sentence_index": m["claim"].get("sentence_index"),
                          "expression_offset_in_sentence": m["claim"].get("expression_offset_in_sentence"),
                          "location": "headline" if m["claim"].get("from_headline") else "body",
                          "span": m["claim"].get("mention_span", m["claim"].get("span", "")), "expression": m["claim"].get("raw_value")}
                         for m in mentions],
            "tolerance_percent": 25,
            "pairings": [],
            "verdict": "uncheckable",
            "note": "",
        }
        subj_durs = {eff: ds for (s, eff), ds in durations.items() if s == subject}
        base_durs = {eff: ds for (s, eff), ds in durations.items() if s == baseline}
        entry["assertion_status"] = claim.get("assertion_status", "asserted")
        if entry["assertion_status"] != "asserted" or (subject and subject == baseline):
            entry["note"] = "No affirmative comparison between two distinct models to assess."
            checks.append(entry)
            continue
        if not subject or not baseline:
            entry["note"] = (
                "Multiplier states no measured baseline to check against — "
                "a headline figure with nothing behind it. Uncheckable, "
                "not confirmed."
            )
            checks.append(entry)
            continue
        if not subj_durs or not base_durs:
            missing = []
            if not subj_durs:
                missing.append(claim.get("subject_display") or subject)
            if not base_durs:
                missing.append(claim.get("multiplier_baseline_display") or baseline)
            entry["note"] = (
                f"No measured task durations found for {', '.join(missing)} — "
                "the multiplier stands unchecked, not confirmed."
            )
            checks.append(entry)
            continue
        for s_eff, s_list in sorted(subj_durs.items()):
            for b_eff, b_list in sorted(base_durs.items()):
                samples = {"subject_samples": s_list, "baseline_samples": b_list}
                scope = ("effort_unspecified" if not s_eff or not b_eff else
                         "same_effort" if s_eff == b_eff else "cross_effort")
                pair = dict(subject_tier=s_eff or "unstated", baseline_tier=b_eff or "unstated",
                            comparison_scope=scope, claimed_ratio=claimed, **samples)
                # Without a workload identifier it is unsafe to choose among distinct durations.
                sv = {d["minutes"] for d in s_list}
                bv = {d["minutes"] for d in b_list}
                if len(sv) != 1 or len(bv) != 1 or min(sv | bv) <= 0 or claimed <= 0:
                    pair["verdict"] = "insufficient_or_ambiguous_measurements"
                else:
                    sm, bm = next(iter(sv)), next(iter(bv))
                    ratio = bm / sm if direction == "faster" else sm / bm
                    deviation = 100 * abs(ratio - claimed) / claimed
                    pair.update(subject_minutes=sm, baseline_minutes=bm, measured_ratio=round(ratio, 6),
                                deviation_percent=round(deviation, 4),
                                within_tolerance={str(t): deviation <= t + 1e-10 for t in (10, 15, 25, 30)})
                    pair["verdict"] = ("within_tolerance" if pair["within_tolerance"]["25"]
                                       else "outside_tolerance")
                entry["pairings"].append(pair)
        entry["pairings"].sort(key=lambda p: p["comparison_scope"] != "same_effort")
        comparable = [p for p in entry["pairings"] if p["comparison_scope"] == "same_effort"]
        verdicts = {p["verdict"] for p in comparable}
        entry["verdict"] = ("no_same_effort_pair" if not comparable else
                            next(iter(verdicts)) if len(verdicts) == 1 else "mixed")
        entry["note"] = ("The 25% tolerance is a provisional arithmetic heuristic. The summary uses "
                         "only matched effort tiers. Workload equivalence and source independence "
                         "require review. Cross-effort measurements are context only.")
        checks.append(entry)
    return checks


def _separability_report(doc_claims: List[List[Dict]],
                         doc_name_fn) -> List[Dict]:
    """Statistical separability (v1.8.9, run-20 gap 14): a stated
    uncertainty against ranked scores on the same benchmark. When a
    source states a margin (Vals ±0.94pp) and two ranked scores sit
    inside each other's margin (Neon 70.87 #2 vs Fable 71.31 #1, gap
    0.44; vs Sonnet 70.85 #3, gap 0.02), the rank is not separable —
    the order is inside the noise. Only STATED margins are ever used;
    no margin is invented. The layers are stacked, not collapsed:
    each score carries its value, stated rank (or null when the
    source states none), and source; each pair carries its gap and
    its own verdict; the entry verdict is per-pair, so a group can
    be mixed. The Sieve offers the options — the human chooses which
    source direction to trust."""
    scores_by_bench: Dict[str, List[Dict]] = {}
    unc_by_bench: Dict[str, List[Dict]] = {}
    for doc_idx, claims in enumerate(doc_claims):
        for c in claims:
            # v1.8.9 (follow-on 1): a headline figure is bias signal,
            # not a body score — it never joins a separability group.
            if c.get("from_headline"):
                continue
            bench = c.get("benchmark", "")
            if not bench:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", bench.lower()).strip("-")
            if c.get("metric") == "score" and c.get("unit") == "percent":
                scores_by_bench.setdefault(slug, []).append({
                    "subject": c.get("subject", ""),
                    "subject_display": c.get("subject_display", ""),
                    "value": float(c["numeric_value"]),
                    "rank": c.get("rank"),
                    "source": doc_name_fn(doc_idx),
                    "span": c.get("span", ""),
                    "benchmark": bench,
                })
            elif c.get("metric") == "uncertainty":
                unc_by_bench.setdefault(slug, []).append({
                    "value": float(c["numeric_value"]),
                    "subject": c.get("subject", ""),
                    "subject_display": c.get("subject_display", ""),
                    "source": doc_name_fn(doc_idx),
                    "span": c.get("span", ""),
                    "benchmark": bench,
                })
    report: List[Dict] = []
    for slug, scores in scores_by_bench.items():
        subjects = {s["subject"] for s in scores if s["subject"]}
        if len(subjects) < 2:
            continue
        uncs = unc_by_bench.get(slug, [])
        if not uncs:
            continue
        margins = sorted({u["value"] for u in uncs})
        margin = margins[-1]
        bench_display = scores[0]["benchmark"]
        pairs = []
        ordered = sorted(scores, key=lambda s: -s["value"])
        # The same figure reported by several sources (Neon 70.9 in
        # three docs) would otherwise repeat the same pair once per
        # reporting source — dedupe by (subject, value) on each side,
        # first occurrence wins; the scores list above still stacks
        # every source's report.
        seen_pairs = set()
        for i, hi in enumerate(ordered):
            for lo in ordered[i + 1:]:
                # Same model at different precision (Neon 70.87 vs
                # Neon 70.9, four sources) is one figure reported
                # four ways, not two ranks — a pair is only a rank
                # comparison between DIFFERENT subjects.
                if hi["subject"] and hi["subject"] == lo["subject"]:
                    continue
                pair_key = (hi["subject"], hi["value"],
                            lo["subject"], lo["value"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                gap = round(hi["value"] - lo["value"], 5)
                separable = gap >= margin
                pairs.append({
                    "higher": {"subject": hi["subject"],
                               "subject_display": hi["subject_display"],
                               "value": hi["value"], "rank": hi["rank"],
                               "source": hi["source"]},
                    "lower": {"subject": lo["subject"],
                              "subject_display": lo["subject_display"],
                              "value": lo["value"], "rank": lo["rank"],
                              "source": lo["source"]},
                    "gap": gap,
                    "separable": separable,
                    "note": (
                        f"Gap {gap}pp vs stated margin ±{margin}pp — "
                        f"{'separable' if separable else 'NOT separable: the order is inside the margin'}."
                    ),
                })
        verdicts = {p["separable"] for p in pairs}
        verdict = ("separable" if verdicts == {True}
                   else "not_separable" if verdicts == {False} else "mixed")
        not_sep = [p for p in pairs if not p["separable"]]
        note = (
            f"{bench_display}: stated uncertainty ±{margin}pp"
            + (f" (multiple stated margins {margins} — widest used)"
               if len(margins) > 1 else "")
            + f". {len(not_sep)} of {len(pairs)} ranked pair(s) sit inside "
            "the margin — those ranks are not separable. The layers are "
            "stacked below (score + stated rank + uncertainty + source); "
            "choose which source direction to trust. The Sieve does not "
            "collapse them to one verdict."
        )
        report.append({
            "benchmark": slug,
            "benchmark_display": bench_display,
            "uncertainty": margin,
            "uncertainty_sources": [
                {"source": u["source"], "value": u["value"],
                 "span": u["span"]} for u in uncs],
            "scores": ordered,
            "pairs": pairs,
            "verdict": verdict,
            "note": note,
        })
    return report


# ── Temporal before→after pairs (v1.8.9 correction, Fix 1) ────────────────
# Two values that are the endpoints of ONE before→after trajectory
# described in a source sentence are not two measurements of the same
# state, so they must never conflict with each other — Nov 9: Sonnet
# 4.6 "from 0.330 at baseline to 0.641 with memory plus refined
# instruction" and "barely moved (0.330 to 0.364) until prompted ...
# which took it to 0.641" are the same pair (baseline 0.330, prompted
# 0.641), not a 0.311 discrepancy. The detector is deliberately
# narrow: a compared entry's span must contain BOTH compared values
# as numbers AND a transition cue. A span that merely mentions one
# endpoint, or a plain disagreement ("scored 0.641" vs "scored
# 0.400"), never fires. A "ranged from X to Y" span is a range, not a
# trajectory, and is excluded.
_TEMPORAL_CUE_RES = (
    re.compile(r"\bbaseline\b", re.I),
    re.compile(r"\btook (?:it|them|him|her) to\b", re.I),
    re.compile(r"\buntil\b", re.I),
    re.compile(r"\b(?:rose|increased|went|moved|jumped|climbed|grew|improved|"
               r"doubled|nearly doubled)\b[^.!?\n]{0,120}?\bfrom\b[^.!?\n]{0,120}?\bto\b",
               re.I),
    re.compile(r"\bfrom\b[^.!?\n]{0,120}?\bat baseline\b[^.!?\n]{0,60}?\bto\b", re.I),
    re.compile(r"\bbefore\b[^.!?\n]{0,80}?\bafter\b", re.I),
)
_TEMPORAL_RANGE_EXCLUSION_RE = re.compile(
    r"\brang(?:e|ed|ing)\b[^.!?\n]{0,40}?\bfrom\b", re.I)
_TEMPORAL_NUMBER_RE = re.compile(r"(?<!\d)\d[\d,]*\.?\d*")


def _temporal_comparison(key: str, entries: List[Dict], val_objs: List[Dict],
                         variant: str, doc_name_fn) -> Optional[Dict]:
    """Return a temporal-comparison entry when the compared values are
    the two endpoints of one before→after trajectory, else None.

    *entries* are the compared claims (one value per doc after modal
    selection for generic groups); *val_objs* are their value objects
    with sources. Requires exactly two distinct compared values, and
    at least one compared entry's span to contain both values plus a
    transition cue. Before/after order follows first occurrence in
    the earliest satisfying span.
    """
    values = sorted({e["claim"]["numeric_value"] for e in entries})
    if len(values) != 2:
        return None
    low, high = values
    evidence: List[Dict] = []
    for e, vo in zip(entries, val_objs):
        span = e["claim"].get("span", "") or vo.get("span", "") \
            or e["claim"].get("context", "")
        if not span:
            continue
        nums = set()
        for m in _TEMPORAL_NUMBER_RE.finditer(span):
            try:
                nums.add(float(m.group(0).replace(",", "")))
            except ValueError:
                continue
        if low not in nums or high not in nums:
            continue
        if _TEMPORAL_RANGE_EXCLUSION_RE.search(span):
            continue
        if not any(cue.search(span) for cue in _TEMPORAL_CUE_RES):
            continue
        evidence.append({"source": vo["source"], "span": span})
    if not evidence:
        return None
    first_span = evidence[0]["span"]
    positions = {}
    for v in (low, high):
        best = None
        for m in _TEMPORAL_NUMBER_RE.finditer(first_span):
            try:
                if float(m.group(0).replace(",", "")) == v:
                    best = m.start()
                    break
            except ValueError:
                continue
        positions[v] = best if best is not None else 1 << 30
    before_val = low if positions[low] <= positions[high] else high
    after_val = high if before_val == low else low
    by_value = {}
    for vo in val_objs:
        by_value.setdefault(vo["value"], vo)
    entry: Dict = {
        "entity_key": key,
        "entity_label": entries[0]["claim"]["entity_label"],
        "before": {"value": before_val,
                   "source": by_value[before_val]["source"],
                   "raw": by_value[before_val]["raw"]},
        "after": {"value": after_val,
                  "source": by_value[after_val]["source"],
                  "raw": by_value[after_val]["raw"]},
        "values": val_objs,
        "variant": variant,
        "evidence_spans": evidence,
        "note": (
            f"Before→after pair: {before_val} at baseline → {after_val} "
            "after the stated change — the two values are the endpoints "
            "of one trajectory described in the source span(s), not two "
            "measurements of the same state. Not a conflict and not an "
            "alignment of equal values."
        ),
    }
    if entries and entries[0]["claim"].get("claim_kind") == "generic_quantitative":
        entry["claim_kind"] = "generic_quantitative"
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
          "not_comparable":     [...],   # one pricing key, several qualified identities (v1.8.8)
          "temporal_comparisons": [...], # before→after pairs, not conflicts (v1.8.9 correction)
          "needs_source_clarification": [...],  # values differ, identity incomplete — never a conflict (v1.8.8 r3, restored v1.8.9)
          "has_unresolved_comparisons": bool,  # needs_source_clarification or not_comparable non-empty (v1.8.8 r4, restored v1.8.9)
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
            "statistical_documents": [],
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
            "not_comparable":      [],
            "temporal_comparisons": [],
            "needs_source_clarification": [],
            "speed_ratio_checks":  [], "era_suggestions": [],
            "separability":        [],
            "revision_notes":      [],
            "headline_weighting":  [],
            "shared_dates":      [],
            "shared_years":      [],
            "unique_dates":      [],
            "timeline_reconciliation": [],
            "agreement_status":  "not_established",
            "has_unresolved_comparisons": False,
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
        if abs(a) <= 100 and abs(b) <= 100:
            # v1.8.9 (run-20 gap 2a): index scores live in 0-100
            # and adjacent ranks differ by ~1 point, so the old 2%
            # relative tolerance over-merged — a bare "Neon" AA
            # score of 61 attached to BOTH Carbon's 60 and Astra's
            # 62. Absolute 0.5 still covers rounding attachments
            # (58.6 <-> 59) while blocking adjacent-index confusion.
            # Large counts keep the relative rule (4,200 <-> 4,234
            # sessions are the same figure at different precision).
            return abs(a - b) <= 0.5
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
                # v1.8.8 r2: value proximity alone must not attach a
                # bench-less score to a benchmark its own sentence
                # never names — "On SWE-bench Verified, ... 76.2"
                # attached to DeepSWE 75.7 by value alone (a false
                # alignment; it had attached to Terminal-Bench the
                # same way). When the sentence names benchmarks and
                # the target is not among them, the entry stands
                # alone. A sentence naming no benchmark attaches as
                # before.
                target_bench = _ctx_bench(matches[0])[0]
                span_slugs = _span_bench_slugs(e["claim"].get("span", ""))
                if span_slugs and target_bench not in span_slugs:
                    continue
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
            eff = e["claim"].get("effort", "")
            era = e["claim"].get("era", "")
            # v1.8.8: a stated group matches only when its values are
            # close AND its thinking level is compatible with this
            # entry's — both stated and different means different
            # qualified identities, never an attachment. (Either side
            # unstated is compatible: level-less is provisional, not a
            # different level.)
            # Worker B (v1.8.9 correction): the era must match too —
            # an unstated-era price never stands in for a stated one,
            # and a prior-era price must never attach to a current-
            # era group (or vice versa) on value alone. A prior $10
            # attaches to the prior fable-5-1 group; an unknown-era
            # $8 attaches to the unknown-era fable-5-2 group. (For
            # benchmark keys the era is "" on every side, so the
            # check is a no-op there.)
            matches = [s for s, es in stated_subjects.items()
                       if any(_generic_close(x["claim"].get("numeric_value", 0), val)
                              and (not eff or not x["claim"].get("effort", "")
                                   or x["claim"].get("effort", "") == eff)
                              and x["claim"].get("era", "") == era
                              for x in es)]
            if len(matches) == 1:
                # Worker B (v1.8.9 correction): the same doc restating
                # the same price twice ("The standard price is $8 ...
                # down from $10" after "cutting Fable's API price to
                # $8") is one claim, not two — when the target group
                # already holds this doc's identical value, the
                # unstated restatement is dropped instead of attached
                # as a duplicate entry (live lists one $8 per doc).
                if any(x["doc_idx"] == e["doc_idx"]
                       and _generic_close(x["claim"].get("numeric_value", 0), val)
                       and (not eff or not x["claim"].get("effort", "")
                            or x["claim"].get("effort", "") == eff)
                       for x in stated_subjects[matches[0]]):
                    entity_map[key] = [x for x in entity_map[key] if x is not e]
                    continue
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
    # Identity splits (v1.8.8, Fix 1): one pricing key that held
    # several qualified identities (era / model / thinking level) —
    # visible NOT_COMPARABLE records, never a silent non-match.
    not_comparable: List[Dict] = []
    # Temporal before→after pairs (v1.8.9 correction, Fix 1): the
    # endpoints of one trajectory, recognised in the group loop below
    # — neither a conflict nor an alignment of equal values.
    temporal_comparisons: List[Dict] = []
    # Conflict integrity (v1.8.8 r3, restored v1.8.9 correction,
    # Fix 3): values that differ beyond tolerance under an INCOMPLETE
    # claim identity — subject, or measurement condition unstated on
    # some side — are neither a conflict nor an alignment. They are
    # listed here, with every value's identity and source span, for a
    # reader to resolve against the sources.
    needs_source_clarification: List[Dict] = []
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
        # v1.8.8, Fix 1: pricing partitions also carry effort, so a
        # per-task price at high thinking never compares against the
        # same price at medium — and a price list that spans a
        # sequenced change ("$2 through Nov. 30, then $4 from Dec. 1")
        # partitions into its current and future identities instead of
        # reading as a dispute. Everything else partitions by variant
        # alone, as before.
        stated_variants = {e["claim"].get("variant", "")
                           for e in entries if e["claim"].get("variant")}
        split_variant = len(stated_variants) > 1
        def _part_key(e):
            c = e["claim"]
            if key in _BENCHMARK_KEYS:
                v = c.get("variant", "") if split_variant else ""
                return (v, c.get("subject", ""), "")
            if key.startswith("pricing"):
                return (c.get("era", ""), c.get("subject", ""), c.get("effort", ""))
            if key.startswith("generic|"):
                # v1.8.9 (run-20 gap 12): a relative-delta group whose
                # sides point opposite directions ("25% cheaper" vs
                # "50% more") is two identities, not a dispute — the
                # direction lives on the claim, not in the key (the
                # pinned "rel" key is unchanged), so the partition
                # splits mixed-direction groups here. Same-direction
                # groups stay one partition and compare as before.
                return (c.get("variant", ""), c.get("rel_direction", ""), "")
            return (c.get("variant", ""), "", "")
        partitions: Dict[Tuple[str, str, str], List[Dict]] = {}
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
            if key.startswith("pricing"):
                # v1.8.8, Fix 1: the identity split is a visible
                # NOT_COMPARABLE record, not a silent non-match — the
                # key held several qualified identities (era / model /
                # thinking level), and no two of them may conflict.
                # (Generic keys split differently: a relative delta and
                # an absolute rate, or a plan price and a per-unit
                # price, mint under different keys by construction, so
                # their non-comparability is structural, not recorded
                # here. Benchmark variant splits already have
                # variant_mismatches above.)
                not_comparable.append({
                    "entity_key":   key,
                    "entity_label": entries[0]["claim"]["entity_label"],
                    "identities": [
                        {"era": pk[0], "subject": pk[1], "effort": pk[2],
                         "values": [{"source": _doc_name(e["doc_idx"]),
                                     "value": e["claim"]["numeric_value"],
                                     "unit": e["claim"]["unit"],
                                     "raw": e["claim"]["raw_value"],
                                     "context": e["claim"].get("context", "")}
                                    for e in part]}
                        for pk, part in partitions.items()
                    ],
                    "note": (
                        "Same pricing key reported under different "
                        "identities (era / model / thinking level) — "
                        "including an unstated era, which never stands "
                        "in for a stated one. Figures from different "
                        "identities are not directly comparable. Do not "
                        "read as a conflict or an alignment. Each value "
                        "carries its source span (context) so the "
                        "figure can be checked where it was written."
                    ),
                })
            for (variant, _subject, _effort), part in partitions.items():
                grouped.append((key, part, variant if split_variant else group_variant))
        else:
            grouped.append((key, entries, group_variant))

    # A presence/absence mismatch is recorded once per key, even when
    # the key was split into several variant partitions.
    _presence_mismatch_done: set = set()

    # ── Cross-unit / separability / revision outputs (v1.8.9) ─────────
    # Collected during the group loop and reported after it: multiplier
    # claims (never compared as facts), revision groups (never counted
    # as agreement), and the separability report (built from the raw
    # claims, not the groups).
    multiplier_entries: List[Dict] = []
    revision_notes: List[Dict] = []
    headline_entries: List[Dict] = []

    def _identity_gaps(key: str, group: List[Dict], generic: bool) -> List[str]:
        # v1.8.8 r3, restored v1.8.9 correction (Fix 3), with TWO
        # documented deviations. (1) The r3 "era" gap is NOT applied.
        # Era is partition-scoped in this tree — a group never mixes
        # eras, so the era gap could only fire when era is unstated
        # on EVERY side, and this work copy pins that case to compare
        # ("still compares two undated prices", v1.8.8 repair):
        # two undated prices that disagree conflict; mixed-era groups
        # never reach this gate because they split into
        # not_comparable records instead. (2) The r3 "subject" gap is
        # not applied to mapped claims when subject is unstated on
        # EVERY side (see the subject branch below): this tree's
        # mapped extraction leaves subject empty even when the
        # sentence names the model, so all-empty is an extraction gap
        # there, not an incomplete identity. Everything else is r3
        # verbatim: subject (mixed/differs, and all-unstated generic),
        # variant, version, shot count, generic benchmark/time, and
        # one source reporting several values.
        is_bench = key in _BENCHMARK_KEYS
        is_price = key.startswith("pricing")
        if not (is_bench or is_price or generic):
            return []
        gaps: List[str] = []

        def _partial(field: str) -> bool:
            stated = [bool(e["claim"].get(field)) for e in group]
            return any(stated) and not all(stated)

        subjects = [e["claim"].get("subject", "") for e in group]
        if any(subjects) and not all(subjects):
            gaps.append("subject")
        elif len(set(subjects)) > 1:
            gaps.append("subject (differs)")
        elif generic and not any(subjects):
            # Generic extraction reliably names the subject when the
            # sentence states one, so an all-unstated generic identity
            # is genuinely incomplete. Mapped claims are different:
            # this tree's mapped extraction often leaves subject empty
            # even when the sentence names the model ("On GPQA, Opus
            # 5.5 scored 87.0%"), so all-empty there is an extraction
            # gap, not evidence of an incomplete identity — and a
            # genuine same-model benchmark dispute must still
            # conflict. (v1.8.8 r3's verbatim rule would suppress it.)
            gaps.append("subject")
        if is_price:
            if key == "pricing-per-task" and not all(e["claim"].get("effort") for e in group):
                gaps.append("condition (thinking level)")
        if is_bench:
            if _partial("variant"):
                gaps.append("variant")
            if _partial("version"):
                gaps.append("version")
            shots = [e["claim"].get("shot_count") is not None for e in group]
            if any(shots) and not all(shots):
                gaps.append("shot count")
        if is_bench or is_price:
            # One source reporting two different values under one
            # identity is an extraction ambiguity inside that source,
            # not evidence that two sources disagree. (Generic groups
            # resolve this earlier by comparing value sets.)
            per_doc: Dict[int, set] = {}
            for e in group:
                per_doc.setdefault(e["doc_idx"], set()).add(e["claim"]["numeric_value"])
            if any(len(v) > 1 for v in per_doc.values()):
                gaps.append("one source reports several values")
        if generic:
            if _partial("benchmark"):
                gaps.append("benchmark")
            if _partial("time"):
                gaps.append("time")
            elif len({e["claim"].get("time", "") for e in group}) > 1:
                gaps.append("time (differs)")
        return gaps

    def _identity_values(group: List[Dict]) -> List[Dict]:
        out = []
        for e in group:
            c = e["claim"]
            vo = {"source": _doc_name(e["doc_idx"]), "value": c["numeric_value"],
                  "unit": c["unit"], "raw": c["raw_value"],
                  "subject": c.get("subject", ""),
                  "context": c.get("span") or c.get("context", "")}
            for f in ("era", "effort", "variant", "version", "qualifier", "time", "benchmark"):
                if c.get(f):
                    vo[f] = c[f]
            if c.get("shot_count") is not None:
                vo["shot_count"] = c["shot_count"]
            if c.get("subject_attached"):
                vo["subject_attached"] = True
            out.append(vo)
        return out

    for key, entries, variant in grouped:
        # v1.8.9 (run-20 gap 11): a multiplier claim is a headline
        # exaggeration metric, not a fact — it never aligns, conflicts,
        # or counts as a unique. It is routed to the speed-ratio check
        # (measured durations vs claimed multiplier) below.
        if key.startswith("generic|multiplier|"):
            multiplier_entries.extend(entries)
            # A headline multiplier is ALSO a headline figure — it
            # belongs in headline_weighting as the bias signal that
            # the publisher led with it (the speed-ratio check below
            # dedupes it against the body multiplier).
            for e in entries:
                if e["claim"].get("from_headline"):
                    headline_entries.append(e)
            continue
        # v1.8.9 (run-20 follow-on 1): headline score claims are bias
        # signal, never facts — routed out of the group loop into
        # headline_weighting, like multipliers.
        if key.startswith("generic|headline|"):
            headline_entries.extend(entries)
            continue
        # ── Generic-claim group shaping (v1.8.6, P2) ─────────────────────
        is_generic = bool(entries) and entries[0]["claim"].get("claim_kind") == "generic_quantitative"
        if is_generic:
            # (a) Revision guard (v1.8.9, run-20 gap 10 residual): a
            # group holding a revision-marked claim whose values
            # differ is a REVISION, not an agreement — "revised:
            # 1336 → 1333", not "aligned [1336, 1336, 1333]". This is
            # exactly the catch that matters: a revision that reads
            # as agreement is how a pattern of failure gets re-minted
            # as a pattern of success or consistency. The group is
            # surfaced as a revision (shared entry annotated, values
            # stacked with their sources) and never enters alignment.
            if any(e["claim"].get("revision") for e in entries):
                _rev_vals = {e["claim"]["numeric_value"] for e in entries}
                if len(_rev_vals) > 1:
                    _rev_vobjs = []
                    for e in entries:
                        _vo = {"source": _doc_name(e["doc_idx"]),
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
                               "attribution": e["claim"].get("attribution", ""),
                               "revision": bool(e["claim"].get("revision")),
                               "revision_cue": e["claim"].get("revision_cue", ""),
                               "revision_date": e["claim"].get("revision_date", "")}
                        if e["claim"].get("rank") is not None:
                            _vo["rank"] = e["claim"]["rank"]
                        _rev_vobjs.append(_vo)
                    _prev = [v for v, e in zip(_rev_vobjs, entries)
                             if not e["claim"].get("revision")]
                    _new = [v for v, e in zip(_rev_vobjs, entries)
                            if e["claim"].get("revision")]
                    _prev_vals = sorted({v["value"] for v in _prev})
                    _new_vals = sorted({v["value"] for v in _new})
                    _rev_label = entries[0]["claim"]["entity_label"]
                    _rev_note = (
                        f"Revised: {', '.join(str(v) for v in _prev_vals)} → "
                        f"{', '.join(str(v) for v in _new_vals)}"
                        + (f" ({_new[0].get('revision_date')})"
                           if _new and _new[0].get("revision_date") else "")
                        + " — the board moved; do not read as agreement. "
                        "A revision that reads as agreement is how a "
                        "pattern of failure gets re-minted as success."
                    )
                    revision_notes.append({
                        "entity_key": key,
                        "entity_label": _rev_label,
                        "previous": _prev,
                        "revised": _new,
                        "note": _rev_note,
                    })
                    shared_claims.append({
                        "entity_key": key,
                        "entity_label": _rev_label,
                        "found_in_docs": list(dict.fromkeys(
                            _doc_name(e["doc_idx"]) for e in entries)),
                        "values": _rev_vobjs,
                        "variant": variant,
                        "claim_kind": "generic_quantitative",
                        "revision": True,
                        "revision_note": _rev_note,
                    })
                    continue
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
                if e["claim"].get("effort"):
                    vo["effort"] = e["claim"]["effort"]
                if e["claim"].get("rank") is not None:
                    vo["rank"] = e["claim"]["rank"]
                if e["claim"].get("revision"):
                    vo["revision"] = True
                    vo["revision_cue"] = e["claim"].get("revision_cue", "")
                    vo["revision_date"] = e["claim"].get("revision_date", "")
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

        if key.startswith("share-close|"):
            is_conflict = spread != 0
            for value, entry in zip(val_objs, entries):
                value["source_span"] = entry["claim"]["context"]

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

        # Temporal before→after guard (v1.8.9 correction, Fix 1):
        # endpoints of one trajectory are a temporal comparison, not
        # a conflict — checked before the identity gate, because it
        # is the more specific diagnosis.
        if is_conflict:
            temporal = _temporal_comparison(key, entries, val_objs, variant, _doc_name)
            if temporal is not None:
                temporal_comparisons.append(temporal)
                continue

        # Conflict-integrity gate (v1.8.8 r3, restored v1.8.9
        # correction, Fix 3): a discrepancy under an incomplete
        # identity is never a conflict — and never an alignment
        # either. It is listed for source clarification.
        if is_conflict:
            gaps = _identity_gaps(key, entries, is_generic)
            if gaps:
                needs_source_clarification.append({
                    "entity_key":   key,
                    "entity_label": label,
                    "spread":       round(spread, 5),
                    "variant":      variant,
                    "missing":      gaps,
                    "values":       _identity_values(entries),
                    "note": (
                        f"Values differ by {spread:.3f}, but the claim identity is "
                        f"incomplete ({', '.join(gaps)} not stated, or not the same, on "
                        "every side) — the figures may describe different models, "
                        "dates or conditions. Not a conflict and not an alignment: "
                        "check each value's source span before forwarding."
                    ),
                })
                continue

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
        # v1.8.8 r3: the comparison identity, so a reader can see what
        # the values were compared AS ("" = unstated on some side).
        if key in _BENCHMARK_KEYS or key.startswith("pricing") or is_generic:
            subjects = {e["claim"].get("subject", "") for e in entries}
            if is_generic and key.count("|") == 5:
                subjects = {"" if key.split("|")[5] == "unstated" else key.split("|")[5]}
            ident = {"subject": next(iter(subjects)) if len(subjects) == 1 else ""}
            if key.startswith("pricing"):
                eras = {e["claim"].get("era", "") for e in entries}
                ident["era"] = next(iter(eras)) if len(eras) == 1 else ""
                efforts = {e["claim"].get("effort", "") for e in entries}
                if efforts != {""}:
                    ident["effort"] = next(iter(efforts)) if len(efforts) == 1 else ""
            entry["identity"] = ident

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

    # ── Temporal double-count note (v1.8.9 correction, Fix 1) ──────────
    # The baseline endpoint of a temporal pair may ALSO be aligned
    # under another metric (Nov 9: 0.330 aligns as `score` from the
    # Figure-3 alt text while the pair mints it under `memory`). Name
    # that overlap on the temporal entry so the figure is not silently
    # double-counted under two metrics. The figure still appears in
    # both groups' shared claims — removing it from either group is an
    # extraction-level dedup, not a compare-verdict call — but the
    # verdict no longer treats it as a conflict, and the overlap is
    # visible here.
    for _t in temporal_comparisons:
        _subj = _t["entity_key"].split("|")[-1] if "|" in _t["entity_key"] else ""
        _t["also_aligned_as"] = sorted({
            _a["entity_key"] for _a in alignment
            if _subj and _a["entity_key"] != _t["entity_key"]
            and _a["entity_key"].split("|")[-1] == _subj
            and any(_v["value"] == _t["before"]["value"]
                    for _v in _a.get("values", []))
        }) if _subj else []

    # ── Bare-cost cross-subject unit check (Worker B, v1.8.9 correction)
    # The per-key unit guard above only fires inside one key's group,
    # and generic cost keys carry the subject — so a bare cost parsed
    # as a count ("about $75 billion" mints unit "count") and a bare
    # cost parsed as dollars ("$14.83", unit "$") sit under different
    # subject keys and the $/count mismatch live v1.8.8 reported is
    # lost, even though both claims are the same bare-cost identity
    # (generic|cost|cost, no benchmark, no per-unit denominator).
    # This pass restores that classification WITHOUT de-attributing
    # either side: it pools the bare-cost claims across subject
    # variants of the one identity and, when 2+ docs report it in
    # different units, records one unit_mismatches entry under the
    # unstated key (the live v1.8.8 key). The claims themselves stay
    # where the group loop put them (uniques, never aligned, never a
    # conflict) — only the mismatch classification is restored.
    # Scoped to bare generic costs on purpose: a per-task or
    # bench-attached cost is a different identity and must never be
    # pooled here.
    _bare_cost: List[Dict] = []
    for _di, _claims in enumerate(doc_claims):
        for _c in _claims:
            if _c.get("claim_kind") != "generic_quantitative":
                continue
            _parts = _c.get("entity_key", "").split("|")
            if len(_parts) == 6 and _parts[1] == "cost" and _parts[2] == "cost" \
                    and _parts[3] == "" and _parts[4] == "":
                _bare_cost.append({"doc_idx": _di, "claim": _c})
    # One representative per doc (the first bare-cost claim in doc
    # order) — the mismatch is "same entity, different units across
    # sources", so one value per source is the entry, not every bare
    # cost in the packet. This mirrors live v1.8.8, whose entry
    # carried exactly the two compared values ($75 billion vs $14.83).
    _bc_by_doc: Dict[int, Dict] = {}
    for _e in _bare_cost:
        _bc_by_doc.setdefault(_e["doc_idx"], _e)
    _bc_reps = list(_bc_by_doc.values())
    _bare_units = {_e["claim"].get("unit", "") for _e in _bc_reps} - {""}
    if len(_bare_units) > 1 and len(_bc_reps) >= 2:
        _bc_vals = []
        for _e in _bc_reps:
            _c = _e["claim"]
            _vo = {"source": _doc_name(_e["doc_idx"]), "value": _c["numeric_value"],
                   "unit": _c["unit"], "raw": _c["raw_value"],
                   "version": _c.get("version", ""),
                   "shot_count": _c.get("shot_count"),
                   "variant": _c.get("variant", ""),
                   "span": _c.get("span", ""),
                   "metric": _c.get("metric", ""),
                   "benchmark": _c.get("benchmark", ""),
                   "context": _c.get("context", ""),
                   "qualifier": _c.get("qualifier", ""),
                   "time": _c.get("time", ""),
                   "attribution": _c.get("attribution", "")}
            _bc_vals.append(_vo)
        unit_mismatches.append({
            "entity_key":   "generic|cost|cost|||unstated",
            "entity_label": "cost",
            "units":        sorted(_bare_units),
            "values":       _bc_vals,
            "variant":      "",
            "note": (
                "Same entity reported in different units across sources "
                "(e.g. a count vs a percentage) — values are not directly "
                "comparable. Do not read as a conflict or an alignment."
            ),
        })

    # ── Cross-unit speed-ratio checks + separability (v1.8.9) ─────────
    # Built from the raw claims, not the groups: multipliers were
    # routed out of the group loop above, and separability groups
    # ranked scores by benchmark with their stated margins.
    speed_ratio_checks = _speed_ratio_checks(
        multiplier_entries, doc_claims, _doc_name)
    separability = _separability_report(doc_claims, _doc_name)
    headline_weighting = _headline_weighting(headline_entries, _doc_name)

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
    n_clarify = len(needs_source_clarification)
    if n_conflicts > 0:
        triage_summary = (
            f"{n_conflicts} numeric conflict(s) detected across "
            f"{len(analyzed_docs)} documents with {n_shared} shared claim(s). "
            "Verify publication timestamps and benchmark definitions before forwarding."
        )
    elif n_clarify > 0:
        triage_summary = (
            f"No confirmed conflicts across {len(analyzed_docs)} documents: "
            f"{len(alignment)} claim group(s) align within tolerance."
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
    # v1.8.8 r4 (restored v1.8.9 correction, Fix 3): agreement_status
    # "aligned" must never read as a clean bill of health while
    # comparisons sit unresolved. The flag is additive (the enum is
    # unchanged) and is stated here, immediately next to the headline,
    # so a skimming reader cannot mistake "aligned" for "nothing left
    # open".
    n_notcmp = len(not_comparable)
    has_unresolved_comparisons = bool(n_clarify or n_notcmp)
    if has_unresolved_comparisons:
        triage_summary += (
            f" UNRESOLVED: {n_clarify} discrepancy(ies) need source clarification, "
            f"{n_notcmp} claim key(s) are not comparable — agreement_status "
            f"'{agreement_status}' covers only the claims that were actually compared, "
            "not these. See needs_source_clarification and not_comparable."
        )
    if n_clarify > 0:
        missing_fields = sorted({g for c in needs_source_clarification for g in c["missing"]})
        triage_summary += (
            f" {n_clarify} discrepancy(ies) need source clarification: the values "
            f"differ, but the claim identity is incomplete ({', '.join(missing_fields)}) "
            "— not a conflict, not agreement. See needs_source_clarification."
        )
    if not_comparable:
        triage_summary += (
            f" {len(not_comparable)} claim key(s) held several identities (model / era / "
            "thinking level) — compared only within each identity. See not_comparable."
        )
    if temporal_comparisons:
        triage_summary += (
            f" {len(temporal_comparisons)} temporal comparison(s): a before→after pair "
            "(baseline → later state) was recognised as one trajectory, not a conflict "
            "and not an alignment of equal values. See temporal_comparisons."
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
    if speed_ratio_checks:
        _src_bad = sum(1 for c in speed_ratio_checks
                       if c["verdict"] in ("outside_tolerance", "mixed"))
        triage_summary += (
            f" {len(speed_ratio_checks)} speed-ratio check(s): a claimed "
            "multiplier was checked against measured task durations — "
            f"{_src_bad} outside tolerance or mixed (headline figure vs measured "
            "time). See speed_ratio_checks; a multiplier is never counted "
            "as factual agreement."
        )
    if separability:
        _src_notsep = sum(1 for s in separability
                          if s["verdict"] in ("not_separable", "mixed"))
        triage_summary += (
            f" {len(separability)} separability report(s): ranked scores "
            "checked against a stated uncertainty — "
            f"{_src_notsep} with ranks inside the margin (not separable). "
            "See separability; the layers are stacked for the reader to "
            "choose a source direction."
        )
    if revision_notes:
        triage_summary += (
            f" {len(revision_notes)} revision(s): a board value was revised "
            "(see revision_notes) — a revision is not agreement, and is "
            "not counted as one."
        )
    if headline_weighting:
        _n_figs = sum(len(h["figures"]) for h in headline_weighting)
        triage_summary += (
            f" {len(headline_weighting)} document(s) headline {_n_figs} figure(s) "
            "(see headline_weighting): a headlined figure is bias/audience "
            "signal — weighted low for information quality, never counted as "
            "factual agreement."
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

    from statistical_evidence import statistics, passage_evidence
    statistical_documents = []
    for idx, doc in enumerate(analyzed_docs):
        body = doc.get("text", "")
        rows = statistics(body)
        receipt = passage_evidence(body, doc.get("source_url", ""), doc.get("source_type", "auto"), rows, _doc_name(idx))
        if receipt is None and not rows and doc.get("source_type") in ("primary", "official") and doc.get("source_url"):
            from signal_sieve import analyze
            receipt = analyze(body, source_type=doc["source_type"], source_name=_doc_name(idx),
                              source_url=doc["source_url"]).get("passage_evidence")
        if rows or receipt:
            statistical_documents.append(dict(document_index=idx, source=_doc_name(idx),
                                              statistical_claims=rows, passage_evidence=receipt))
    return {
        "statistical_documents": statistical_documents,
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
        "not_comparable":      not_comparable,
        "temporal_comparisons": temporal_comparisons,
        "needs_source_clarification": needs_source_clarification,
        "era_suggestions": [hint for idx, doc in enumerate(analyzed_docs)
                            for hint in _era_suggestions(doc.get("text", ""), _doc_name(idx), idx)],
        "speed_ratio_checks": speed_ratio_checks,
        "separability":      separability,
        "revision_notes":    revision_notes,
        "headline_weighting": headline_weighting,
        "shared_dates":      shared_dates,
        "shared_years":      shared_years,
        "unique_dates":      unique_dates,
        "timeline_reconciliation": timeline_notes,
        "agreement_status":  agreement_status,
        "has_unresolved_comparisons": has_unresolved_comparisons,
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
