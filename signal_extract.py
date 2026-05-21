"""
signal_extract.py — genre detection, signal brief, custody breakdown, evidence shape.

Zero external dependencies. Drop alongside signal_sieve.py.
Imported by signal_sieve.analyze() when available (try/except import).

Phase 1.2 additions:
  detect_genre()          -> genre string
  build_custody_breakdown() -> layered custody dict
  build_evidence_shape()  -> evidence shape dict
  extract_signal_brief()  -> structured signal brief dict
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Genre detection patterns
# ---------------------------------------------------------------------------

_FIN_TERMS = re.compile(
    r"\b(ticker|tickers|shares?|stocks?|ETFs?|NASDAQ|NYSE|TSX|CSE|OTC|"
    r"previous close|market session|basis points|trading volume|IPO|"
    r"market cap|breadth|rally|sell.?off|earnings|revenue|dividend|yield|"
    r"hedge fund|futures|options|portfolio|short interest|float|"
    r"sector|equit(?:y|ies)|index|indices)\b",
    re.I,
)
_FIN_NUMERIC = re.compile(
    r"(?<![A-Za-z])[A-Z]{2,5}(?!\w)\s*(?:/|-|—)\s*[A-Za-z]|"  # TICKER / Name
    r"\$\d+(?:\.\d+)?\b|"                                        # $0.53
    r"[-+]\d+(?:\.\d+)?%|"                                       # -3.5%
    r"\d+(?:\.\d+)?\s*bps\b",                                    # 25 bps
)
_PRESS_RELEASE = re.compile(
    r"\b(FOR IMMEDIATE RELEASE|press release|we are pleased to (?:announce|report|inform)|"
    r"the company (?:announces|is pleased)|(?:NASDAQ|NYSE|TSX|CSE):\s*[A-Z]+|"
    r"forward.?looking statements?|safe harbor|"
    r"for more information.{0,40}(?:contact|visit))\b",
    # NOTE: bare "investor relations" is intentionally excluded — too broad.
    # URL-based IR page detection is handled in detect_genre() via _PRIMARY_URL_HINTS.
    re.I,
)
_OPINION_HINTS = re.compile(
    r"\b(in my (?:view|opinion)|i (?:believe|think|argue|contend)|my take|"
    r"the (?:real|true|actual) (?:issue|problem|story)|"
    r"what (?:nobody|everyone)|we should|they should|"
    r"the fact is|let me be clear|make no mistake|in my judgment)\b",
    re.I,
)
_SOCIAL_HINTS = re.compile(
    r"(?:^|\s)(?:RT\s+@|#[A-Za-z]\w{1,}|@[A-Za-z]\w{1,}|\bthread\b|\bdm me\b|\bfollow\b)",
    re.I | re.MULTILINE,
)
_REGULATORY_URL = re.compile(
    r"\b(sec\.gov|sedarplus\.ca|sedar\.com|cftc\.gov|finra\.org|edgar)\b",
    re.I,
)
_PRIMARY_URL_HINTS = (
    "sec.gov", "sedarplus", "sedar.com", "edgar",
    "investor.", "/investors/", "/ir/",
    "press-release", "press_release", "news-release", "newswire",
    ".csv", ".json", "arxiv.org", "pubmed", "doi.org", "clinicaltrials.gov",
)
_NEWSLIKE_URL_HINTS = (
    "news", "article", "story", "blog", "substack", "medium.com",
    "forbes.com", "reuters.com", "apnews.com", "bloomberg.com",
    "cnbc.com", "marketwatch.com", "thestreet.com", "benzinga.com",
    "seekingalpha.com", "motleyfool.com", "markets", "finance",
)


def detect_genre(text: str, url: str = "") -> str:
    """Classify article genre.

    Returns one of:
      regulatory_filing | official_release | financial_market_snapshot |
      news_article | opinion_article | social_post | generic_article
    """
    lower_url = url.lower()

    if _REGULATORY_URL.search(lower_url):
        return "regulatory_filing"

    if any(h in lower_url for h in (
        "investor.", "/investors/", "/ir/",
        "newswire", "press-release", "press_release", "news-release",
    )):
        return "official_release"

    # Require at least 2 distinct press-release signals to avoid false positives
    # when text merely *mentions* a press release in passing.
    if len(_PRESS_RELEASE.findall(text)) >= 2:
        return "official_release"

    fin_term_hits = len(_FIN_TERMS.findall(text))
    fin_num_hits  = len(_FIN_NUMERIC.findall(text))
    if fin_term_hits >= 4 and fin_num_hits >= 3:
        return "financial_market_snapshot"

    if any(h in lower_url for h in _NEWSLIKE_URL_HINTS):
        return "news_article"

    if _SOCIAL_HINTS.search(text):
        return "social_post"

    if _OPINION_HINTS.search(text):
        return "opinion_article"

    return "generic_article"


# ---------------------------------------------------------------------------
# Named source detection — split into data providers vs market venues
# ---------------------------------------------------------------------------

# Data providers / research services / government statistics agencies
_DATA_PROVIDER_RE = re.compile(
    r"\b(Finnhub|Bloomberg|Reuters|FactSet|Refinitiv|S&P Global|S&P 500|"
    r"Moody'?s|Fitch|EDGAR|SEC|SEDAR|"
    r"Yahoo Finance|Google Finance|Morningstar|CapIQ|PitchBook|Crunchbase|"
    r"CoinMarketCap|CoinGecko|Glassnode|Dune Analytics|"
    r"Alpha Vantage|Quandl|FRED|World Bank|IMF|BLS|BEA|Census Bureau|"
    r"Federal Reserve|The Fed)\b",
)

# Market venues / exchanges / trading platforms
_MARKET_VENUE_RE = re.compile(
    r"\b(NASDAQ|NYSE|TSX|CSE|OTC(?:\s+Markets)?|LSE|ASX|HKEX|SGX|Euronext|"
    r"CBOE|CME|NYMEX|ICE|Cboe)\b",
)

# Combined — keeps backward-compat for code that calls _find_named_sources()
_DATA_SOURCES = re.compile(
    r"\b(Finnhub|Bloomberg|Reuters|FactSet|Refinitiv|S&P Global|S&P 500|"
    r"Moody'?s|Fitch|EDGAR|SEC|SEDAR|NYSE|NASDAQ|TSX|CSE|OTC(?:\s+Markets)?|"
    r"Yahoo Finance|Google Finance|Morningstar|CapIQ|PitchBook|Crunchbase|"
    r"CoinMarketCap|CoinGecko|Glassnode|Dune Analytics|"
    r"Alpha Vantage|Quandl|FRED|World Bank|IMF|BLS|BEA|Census Bureau|"
    r"Federal Reserve|The Fed|LSE|ASX|HKEX|SGX|Euronext|CBOE|CME|NYMEX|ICE)\b",
)


def _unique_ordered(hits: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for h in hits:
        if h.lower() not in seen:
            seen.add(h.lower())
            out.append(h)
    return out


def _find_named_sources(text: str) -> List[str]:
    """All named data sources and market venues (combined)."""
    return _unique_ordered(_DATA_SOURCES.findall(text))


def _find_data_providers(text: str) -> List[str]:
    """Data providers / research services / government agencies only."""
    return _unique_ordered(_DATA_PROVIDER_RE.findall(text))


def _find_market_venues(text: str) -> List[str]:
    """Market exchanges and trading venues only."""
    return _unique_ordered(_MARKET_VENUE_RE.findall(text))


# ---------------------------------------------------------------------------
# Bullet compression
# ---------------------------------------------------------------------------

def _compress(text: str, max_chars: int = 200) -> str:
    """Truncate bullet text to max_chars at a word boundary.

    Strips dangling open punctuation (unclosed parentheses, brackets, etc.)
    so cut-off items don't look like 'Company (TICK…'.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    # Strip trailing noise: punctuation, open brackets, partial words
    truncated = re.sub(r"[\s.,;:\-–—(\[{]+$", "", truncated)
    return truncated + "…"


# ---------------------------------------------------------------------------
# TOP MOVERS parsing
# ---------------------------------------------------------------------------

_TOP_MOVER_ROW = re.compile(
    r"\b([A-Z]{2,5})\b"           # ticker
    r"(?:[^\n.!?]{0,40})"         # optional label / company name snippet
    r"(?:C?\$\s*\d+(?:\.\d+)?)"   # price
    r"[^\n.!?]{0,40}"             # gap
    r"([-+]\d+(?:\.\d+)?%)",      # change
)

# Known words that look like tickers but aren't
_FAKE_TICKERS = frozenset({
    "OTC", "THE", "FOR", "AND", "NOT", "ARE", "HAS", "HAD", "WAS", "ITS",
    "ALL", "BUT", "CAN", "DID", "GET", "GOT", "LET", "MAY", "NEW", "NOW",
    "OUT", "OWN", "PUT", "RUN", "SAY", "SET", "USE", "WAY", "WHO", "WHY",
    "YET", "YOU", "CMS", "SEC", "FDA", "TSX", "CSE", "ETF", "IPO", "CEO",
    "CFO", "COO", "CBD", "THC", "CEO", "EBITDA",
})


_PRICE_RE   = re.compile(r"C?\$\s*\d+(?:\.\d+)?")
_CHANGE_RE  = re.compile(r"[-+]\d+(?:\.\d+)?%")
_COMPANY_BEFORE_RE = re.compile(
    r"([A-Za-z][A-Za-z\s&',.\-]{2,35})\s*\(\s*$"
)


def _parse_top_movers(text: str) -> List[str]:
    """Extract ticker / price / change rows from market snapshot text.

    Returns a list of compact display strings like:
      'VFF  $2.53  +1.20%  (Village Farms International)'
    Handles the common 'CompanyName (TICKER) at $price (±change%)' pattern
    and strips stray closing parentheses that follow the ticker.
    """
    results: List[str] = []
    seen: set = set()

    for m in _TOP_MOVER_ROW.finditer(text):
        ticker = m.group(1)
        if ticker in _FAKE_TICKERS or ticker in seen:
            continue
        seen.add(ticker)

        span = m.group(0)
        price_m  = _PRICE_RE.search(span)
        change_m = _CHANGE_RE.search(span)
        if not price_m or not change_m:
            continue

        price  = price_m.group(0).replace(" ", "")
        change = change_m.group(0)

        # Look back up to 50 chars for a "CompanyName (" pattern
        ctx_start = max(0, m.start() - 50)
        ctx = text[ctx_start: m.start()]
        comp_m = _COMPANY_BEFORE_RE.search(ctx)
        company = comp_m.group(1).strip() if comp_m else ""

        if company:
            display = f"{ticker}  {price}  {change}  ({company})"
        else:
            display = f"{ticker}  {price}  {change}"

        results.append(display)

    return results[:12]


# ---------------------------------------------------------------------------
# Custody breakdown
# ---------------------------------------------------------------------------

_PUBLISHER_WEIGHTS: Dict[str, float] = {
    "primary": 0.88,
    "regulatory_filing": 0.92,
    "official": 0.78,
    "official_release": 0.78,
    "expert": 0.72,
    "first_hand": 0.84,
    "secondary": 0.65,
    "secondary_market_article": 0.65,
    "news_article": 0.65,
    "tertiary": 0.42,
    "anonymous": 0.28,
    "social": 0.28,
    "social_post": 0.28,
    "opinion": 0.40,
    "opinion_article": 0.40,
    "unknown": 0.25,
    "generic_article": 0.50,
}


def build_custody_breakdown(
    text: str,
    genre: str,
    source_type: str,
    source_url: str,
    st_confidence: float,
) -> Dict:
    """Return layered custody breakdown."""
    data_providers = _find_data_providers(text)
    market_venues  = _find_market_venues(text)
    named_srcs     = _unique_ordered(data_providers + market_venues)
    primary_linked = bool(re.search(
        r"https?://(?:sec\.gov|sedarplus|sedar\.com|edgar|doi\.org|arxiv\.org|pubmed)",
        text.lower(),
    ))
    attribution_hits = len(re.findall(
        r"\b(said|reported|according to|stated|confirmed|per|found|cited)\b",
        text, re.I,
    ))
    publisher_custody = _PUBLISHER_WEIGHTS.get(source_type, 0.40)
    data_custody = 0.85 if primary_linked else (0.60 if data_providers else 0.30)
    claim_custody = min(0.30 + attribution_hits * 0.07, 0.85)
    return {
        "publisher_custody":      round(publisher_custody, 2),
        "data_custody":           round(data_custody, 2),
        "claim_custody":          round(claim_custody, 2),
        "source_type_confidence": round(st_confidence, 2),
        "primary_source_present": primary_linked,
        "primary_source_linked":  primary_linked,
        "data_source_named":      bool(data_providers),
        "data_source_name":       data_providers[0] if data_providers else None,
        "named_data_providers":   data_providers,
        "markets_venues_mentioned": market_venues,
    }


# ---------------------------------------------------------------------------
# Evidence shape
# ---------------------------------------------------------------------------

_INTERP_CUES = re.compile(
    r"\b(tone|constructive|cautious|investors? appear|suggests?|signals?|"
    r"could|may indicate|likely|pressure|momentum|optimis\w*|sentiment|"
    r"poised|positioned|seems|appears? to|heading|trending|"
    r"outperform|underperform|bullish|bearish)\b",
    re.I,
)
_CAVEAT_CUES = re.compile(
    r"\b(reflects? prior|not live|as of|OTC price|prior (?:close|trading)|"
    r"no filing|no announcement|data provided by|source:|via\s|caveat|"
    r"note:|disclaimer|subject to change|may be delayed|estimated|approximate|"
    r"for informational purposes)\b",
    re.I,
)
_MISSING_CUES = re.compile(
    r"\b(no filing|no announcement|cause is unknown|no (?:clear )?driver|"
    r"not identified|unconfirmed|unverified|cannot confirm|unclear why|"
    r"reason for the \w+ (?:move|drop|jump|spike|decline|rally))\b",
    re.I,
)


def build_evidence_shape(
    text: str,
    sentences: List[str],
    anchored: int,
    unanchored: int,
) -> Dict:
    """Return evidence shape dict."""
    num_hits = len(re.findall(
        r"\$\d+(?:\.\d+)?|\b\d+(?:\.\d+)?(?:\s*(?:%|bps|million|billion|M\b|B\b|K\b))",
        text,
    ))
    local_numeric = "high" if num_hits >= 10 else ("medium" if num_hits >= 4 else "low")

    primary_links = bool(re.search(
        r"https?://(?:sec\.gov|sedarplus|edgar|doi\.org|arxiv\.org|pubmed)",
        text.lower(),
    ))
    all_links = len(re.findall(r"https?://\S+", text))
    external_receipts = (
        "strong" if primary_links else
        "medium" if all_links >= 2 else
        "weak"   if all_links >= 1 else
        "none"
    )

    data_providers = _find_data_providers(text)
    market_venues  = _find_market_venues(text)
    named_srcs     = _unique_ordered(data_providers + market_venues)
    interp_count   = len(_INTERP_CUES.findall(text))
    missing_count  = len(_MISSING_CUES.findall(text))

    return {
        "local_numeric_anchors":      local_numeric,
        "named_sources":              named_srcs,         # kept for backward compat
        "named_data_providers":       data_providers,
        "markets_venues_mentioned":   market_venues,
        "external_receipts":          external_receipts,
        "primary_links_present":      primary_links,
        "source_caveats_present":     bool(_CAVEAT_CUES.search(text)),
        "unverified_claims_count":    missing_count,
        "interpretive_claims_count":  interp_count,
    }


# ---------------------------------------------------------------------------
# Signal brief extraction
# ---------------------------------------------------------------------------

_TICKER_PRICE_PAT = re.compile(
    r"\b[A-Z]{2,5}\b[^.!?\n]{0,80}(?:\$\d+(?:\.\d+)?|[-+]\d+(?:\.\d+)?%)",
)
_BREADTH_PAT = re.compile(
    r"\b\d+\s*(?:stocks?|tickers?|names?|issues?)?\s*(?:up|down|advanc\w*|declin\w*|flat|unchanged)\b",
    re.I,
)
_SESSION_PAT = re.compile(
    r"\b(?:\d{1,2}:\d{2}\s*(?:am|pm|ET|EST|PT|UTC)|today|this morning|"
    r"this session|as of (?:this )?\w+|market open|market close|"
    r"premarket|at open|at close|Wednesday|Monday|Tuesday|Thursday|Friday)\b",
    re.I,
)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _is_key_signal_generic(sent: str) -> bool:
    """Sentence has specific numeric data and some attribution/specificity."""
    has_number = bool(re.search(
        r"\$\d+|\b\d+(?:\.\d+)?(?:\s*(?:%|million|billion|thousand|bps))",
        sent, re.I,
    ))
    has_attr = bool(re.search(
        r"\b(?:said|reported|found|shows?|data|according to|per|filed|published|recorded)\b",
        sent, re.I,
    ))
    return has_number and (has_attr or len(sent.split()) >= 12)


def _extract_financial(text: str, source_type: str, source_url: str) -> Dict:
    sentences      = _split_sentences(text)
    data_providers = _find_data_providers(text)
    market_venues  = _find_market_venues(text)
    named_srcs     = _unique_ordered(data_providers + market_venues)
    primary_linked = bool(re.search(
        r"https?://(?:sec\.gov|sedarplus|edgar|doi\.org|arxiv)", text.lower(),
    ))

    # ── Extract TOP MOVERS table rows first ──────────────────────────────────
    top_movers = _parse_top_movers(text)
    # Build a set of ticker strings to skip during sentence classification
    _mover_tickers = {m.split()[0] for m in top_movers if m}

    key_signals: List[str] = []
    source_caveats: List[str] = []
    interpretation: List[str] = []
    missing_receipts: List[str] = []

    for sent in sentences:
        if not sent or len(sent.split()) < 4:
            continue
        # Skip table-blob sentences (very long lines = extracted table rows already captured)
        if len(sent) > 350:
            continue
        # Priority: caveat > missing > key signal > interpretation
        if _CAVEAT_CUES.search(sent):
            source_caveats.append(_compress(sent))
        elif _MISSING_CUES.search(sent):
            missing_receipts.append(_compress(sent))
        elif (
            _TICKER_PRICE_PAT.search(sent)
            or _BREADTH_PAT.search(sent)
            or _SESSION_PAT.search(sent)
            or _DATA_SOURCES.search(sent)
        ):
            key_signals.append(_compress(sent))
        elif _INTERP_CUES.search(sent):
            interpretation.append(_compress(sent))

    # Auto-add: named data provider but not linked
    if data_providers and not primary_linked:
        missing_receipts.append(
            f"{', '.join(data_providers[:2])} data is named but not directly verified by this run."
        )
    # Auto-add: secondary article → no primary endpoint checked
    if source_type in ("secondary", "secondary_market_article", "news_article"):
        missing_receipts.append(
            "No linked filing, exchange quote, company release, or raw data endpoint "
            "was checked by this run."
        )

    # Do not pass forward as
    dnpf: List[str] = []
    if source_type in ("secondary", "secondary_market_article", "news_article"):
        dnpf.append("Primary financial data — this is a secondary article.")
    has_otc_caveat = (
        any("OTC" in s.upper() or "prior close" in s.lower() for s in source_caveats)
        or "OTC" in " ".join(market_venues).upper()
    )
    if has_otc_caveat:
        dnpf.append("Live OTC market data (prices reflect prior close, not real-time).")
    # Unexplained large moves
    big_moves = re.findall(r"\b([A-Z]{2,5})\b[^.!?\n]{0,60}[-+]\d{2,}(?:\.\d+)?%", text)
    for ticker in list(dict.fromkeys(big_moves))[:2]:
        if ticker in _FAKE_TICKERS:
            continue
        if not re.search(
            rf"\b{re.escape(ticker)}\b.{{0,120}}"
            r"\b(because|due to|following|after|amid|on news|on volume)\b",
            text, re.I,
        ):
            dnpf.append(
                f"Proof of why {ticker} moved — cause is not established in this text."
            )
            break

    # Follow-up sources
    follow_ups: List[str] = []
    if market_venues and any(v.upper() in ("OTC", "OTC MARKETS") for v in market_venues):
        follow_ups.append("OTC Markets (otcmarkets.com) for real-time OTC quotes.")
    if data_providers:
        follow_ups.append(
            f"Direct {data_providers[0]} API or dashboard for raw data verification."
        )
    follow_ups.append(
        "Company investor-relations pages or SEC/SEDAR filings for any named move drivers."
    )

    return {
        "genre":                      "financial_market_snapshot",
        "source_type":                source_type,
        "primary_data_named":         named_srcs,
        "named_data_providers":       data_providers,
        "markets_venues_mentioned":   market_venues,
        "primary_data_verified":      primary_linked,
        "top_movers":                 top_movers,
        "key_signals":                key_signals[:8],
        "source_caveats":             source_caveats[:5],
        "interpretation_or_framing":  interpretation[:6],
        "missing_receipts":           missing_receipts[:6],
        "do_not_pass_forward_as":     dnpf[:5],
        "follow_up_sources":          follow_ups[:5],
    }


def _extract_generic(text: str, genre: str, source_type: str, source_url: str) -> Dict:
    sentences      = _split_sentences(text)
    data_providers = _find_data_providers(text)
    market_venues  = _find_market_venues(text)
    named_srcs     = _unique_ordered(data_providers + market_venues)
    primary_linked = bool(re.search(
        r"https?://(?:sec\.gov|sedarplus|edgar|doi\.org|arxiv\.org|pubmed)",
        text.lower(),
    ))

    key_signals: List[str] = []
    source_caveats: List[str] = []
    interpretation: List[str] = []
    missing_receipts: List[str] = []

    for sent in sentences:
        if not sent or len(sent.split()) < 5:
            continue
        if _CAVEAT_CUES.search(sent):
            source_caveats.append(_compress(sent))
        elif _MISSING_CUES.search(sent):
            missing_receipts.append(_compress(sent))
        elif _is_key_signal_generic(sent):
            key_signals.append(_compress(sent))
        elif _INTERP_CUES.search(sent):
            interpretation.append(_compress(sent))

    dnpf: List[str] = []
    if source_type in ("secondary", "news_article", "tertiary", "secondary_market_article"):
        dnpf.append("Primary source material — verify key claims at the original source.")
    if data_providers and not primary_linked:
        dnpf.append(
            f"Direct output from {data_providers[0]} — named but not linked or verified by this run."
        )

    follow_ups: List[str] = ["Primary source for each main claim."]
    if named_srcs:
        follow_ups.append(f"Direct source: {', '.join(named_srcs[:2])}.")

    return {
        "genre":                      genre,
        "source_type":                source_type,
        "primary_data_named":         named_srcs,
        "named_data_providers":       data_providers,
        "markets_venues_mentioned":   market_venues,
        "primary_data_verified":      primary_linked,
        "key_signals":                key_signals[:8],
        "source_caveats":             source_caveats[:5],
        "interpretation_or_framing":  interpretation[:6],
        "missing_receipts":           missing_receipts[:4],
        "do_not_pass_forward_as":     dnpf[:4],
        "follow_up_sources":          follow_ups[:4],
    }


def extract_signal_brief(
    text: str,
    genre: str,
    source_type: str,
    source_url: str = "",
) -> Dict:
    """Return structured signal brief. Dispatches to genre-specific extractor."""
    if genre == "financial_market_snapshot":
        return _extract_financial(text, source_type, source_url)
    return _extract_generic(text, genre, source_type, source_url)
