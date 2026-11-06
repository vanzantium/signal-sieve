#!/usr/bin/env python3
"""
signal_sieve.py — heuristic signal-to-noise and source-custody assessor.

Zero-dependency pre-belief filter. It does NOT fact-check the world; it audits
whether text behaves like signal/noise, well/weakly sourced, calibrated/over-
certain, neutral/pressured, or incentive-shaped.

QUICK START
-----------
Command line:
  python signal_sieve.py --demo --pretty
  python signal_sieve.py --text "claim here" --pretty
  python signal_sieve.py --file article.txt --source-type secondary
  cat article.txt | python signal_sieve.py --stdin --json

Library:
  from signal_sieve import analyze
  result = analyze(text, source_type="auto", source_name="Reuters",
                   source_url="https://reuters.com/...")
  # result is a JSON-serializable dict

PIPELINE PATTERN
----------------
  result = analyze(text, source_type="auto")
  if result["recommended_action"] in {"drop", "reject"}:
      # refuse, re-prompt, or warn
  else:
      # feed result["questions_to_ask"] back into the model as a refinement step
      # weight confidence by result["confidence_band"] not single overall_confidence

OUTPUT SCHEMA (stable keys)
---------------------------
  recommended_action  : use_as_anchor | verify_primary | treat_as_lead | seek_receipts | drop | reject
  triage_summary      : one-line summary for fast routing / downstream AI
  verdict             : human-readable summary
  scores              : dict of 0-1 floats (gap is -1..1)
  confidence_band     : {"low": float, "high": float}
  custody_label       : human-readable source custody description
  custody_warning_level: clean | watch | weak | uncertain | mismatch
                        ('mismatch' only when the caller declared a source type
                         the text contradicts; 'uncertain' when undeclared/auto
                         and inference is weak)
  evidence_breakdown  : {"anchored": int, "unanchored": int}
  flags               : [str, ...]
  questions_to_ask    : [str, ...]
  ai_instruction      : str
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# signal_extract is optional (Phase 1.2); signal_sieve stays zero-dependency standalone
try:
    from signal_extract import (
        detect_genre,
        build_custody_breakdown,
        build_evidence_shape,
        extract_signal_brief,
    )
    _EXTRACT_AVAILABLE = True
except ImportError:
    _EXTRACT_AVAILABLE = False

try:
    from signal_sieve_i18n import analyze as analyze_language_layer
    _I18N_AVAILABLE = True
except ImportError:
    _I18N_AVAILABLE = False

# ---------------------------------------------------------------------------
# Regex patterns (English-only heuristics)
# ---------------------------------------------------------------------------

CERTAINTY = re.compile(
    r"\b(always|never|everyone|no one|undeniable|irrefutable|proven|proves|proof|"
    r"definitely|certainly|clearly|obviously|guaranteed|without doubt|no doubt|"
    r"settled science|the science is settled|impossible|cannot be wrong|100%|zero chance)\b",
    re.I,
)
HEDGES = re.compile(
    r"\b(may|might|could|appears|seems|suggests|indicates|roughly|approximately|"
    r"preliminary|uncertain|unresolved|likely|unlikely|plausible|possible|"
    r"limited evidence|early evidence|according to|we do not know|not enough data)\b",
    re.I,
)
HYPE = re.compile(
    r"\b(breakthrough|revolutionary|game[- ]changing|groundbreaking|shocking|"
    r"miracle|secret|one weird trick|they don't want you to know|exclusive|"
    r"mind[- ]blowing|incredible|unbelievable|explosive|bombshell|wake up|"
    r"mainstream media won't tell you|experts hate|hidden truth)\b",
    re.I,
)
URGENCY = re.compile(
    r"\b(act now|hurry|limited time|before it'?s too late|share this|"
    r"must watch|must read|do your own research|open your eyes|don't wait|"
    r"they are coming|final warning|last chance|gets deleted|banned)\b",
    re.I,
)
EVIDENCE_MARKERS = re.compile(
    r"\b(dataset|raw data|methodology|sample size|n\s*=|confidence interval|"
    r"p[- ]value|appendix|supplementary|replication|trial|control group|"
    r"filing|court record|transcript|audit|minutes|invoice|receipt|photo metadata|"
    r"source code|commit|log file|primary source|full report|downloaded data|"
    r"direct quote|interview transcript|official record|SEC|10-Q|EDGAR|limitations section)\b",
    re.I,
)
# Specificity: numbers-with-units, ISO dates, named dates, times, proper-noun pairs, URLs, n=
SPECIFICITY = re.compile(
    r"(https?://\S+|\b\d{4}-\d{2}-\d{2}\b|\b\w+ \d{1,2}, \d{4}\b|\b\d{1,2}:\d{2}\b|"
    r"\b\d+(?:\.\d+)?\s?(?:%|percent|bps|basis points|million|billion|kg|km|miles|hours|days|weeks|months|years|usd|cad|\$|B|M)\b|"
    r"\$\d+(?:\.\d+)?[BMK]?|\bn\s*=\s*\d+|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b)"
)
# Wider window for anchoring evidence markers — any number, URL, date, or proper-noun pair
SPEC_WINDOW = re.compile(
    r"(https?://\S+|\b\d{4}-\d{2}-\d{2}\b|\b\w+ \d{1,2}, \d{4}\b|\b\d+(?:[.,]\d+)?\b|\$\d|\bn\s*=|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b)"
)
LOGIC = re.compile(
    r"\b(because|therefore|however|although|unless|if|then|so that|whereas|"
    r"compared with|relative to|caused by|correlated with|mechanism|tradeoff|"
    r"limitation|counterpoint|alternative explanation|alternative explanations|bias|excluded|though)\b",
    re.I,
)
BIAS_INCENTIVE = re.compile(
    r"\b(our product|our platform|investors?|shareholders?|sponsored|affiliate|"
    r"paid partnership|press release|partnered with|we are excited|"
    r"political action|donate|subscribe|buy now|use code|limited offer|"
    r"my course|my newsletter|my book|my token|financial interest)\b",
    re.I,
)
ATTACK_LANGUAGE = re.compile(
    r"\b("
    r"idiots?|morons?|traitors?|evil|degenerates?|sheep|NPCs?|cultists?|shills?|"
    r"enem(?:y|ies)\s+of|vermin|parasites?|subhuman|"
    r"destroy(?:ed|ing|s)?\s+(?:them|the country)|"
    r"lock(?:ed)?\s+(?:them|him|her|people|those|us)\s*up|"
    r"must\s+be\s+(?:jailed|imprisoned|locked\s+up|purged|removed|eradicated)|"
    r"should\s+be\s+(?:jailed|imprisoned|locked\s+up|purged)|"
    r"purged?|eradicated?|exterminated?"
    r")\b",
    re.I,
)
CONNECTIVES = re.compile(
    r"\b(therefore|because|however|although|despite|since|thus|"
    r"consequently|furthermore|moreover|nevertheless|whereas|"
    r"as a result|in contrast|for this reason|by contrast|given that|"
    r"it follows that|on the other hand|in turn|this means|which means)\b",
    re.I,
)
FILLER = re.compile(
    r"\b(very|really|basically|actually|literally|sort of|kind of|"
    r"at the end of the day|needless to say|it is important to note|"
    r"as everyone knows|of course|trust me)\b",
    re.I,
)
QUOTE_PAT = re.compile(r"[\u201c\"][^\u201d\"]{20,}[\u201d\"]")
ATTRIBUTION_VERB = re.compile(
    r"\b(said|told|stated|wrote|noted|reported|added|explained|argued|"
    r"asked|replied|warned|testified|confirmed|denied|tweeted|posted)\b",
    re.I,
)
NAME_PAT = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# ---------------------------------------------------------------------------
# Source-type weights, labels, URL classes
# ---------------------------------------------------------------------------

SOURCE_TYPE_WEIGHTS = {
    "first_hand": 0.90,
    "primary": 0.88,
    "regulatory_filing": 0.92,
    "official": 0.78,
    "official_release": 0.78,
    "expert": 0.72,
    "secondary": 0.62,
    "secondary_market_article": 0.62,
    "news_article": 0.62,
    "tertiary": 0.42,
    "anonymous": 0.30,
    "social": 0.28,
    "social_post": 0.28,
    "opinion": 0.35,
    "opinion_article": 0.35,
    "unknown": 0.25,
}
SOURCE_LABELS = {
    "first_hand": "FIRST-HAND / direct witness",
    "primary": "PRIMARY / raw record or research",
    "regulatory_filing": "REGULATORY FILING / primary compliance record",
    "official": "OFFICIAL / institution statement, incentive-shaped",
    "official_release": "OFFICIAL RELEASE / company or institutional statement",
    "expert": "EXPERT / named interpretation",
    "secondary": "SECONDARY / reporting or analysis",
    "secondary_market_article": "SECONDARY MARKET ARTICLE / article-layer, not raw data",
    "news_article": "NEWS ARTICLE / secondary reporting",
    "tertiary": "TERTIARY / summary of summaries",
    "anonymous": "ANONYMOUS / weak custody",
    "social": "SOCIAL / high drift risk",
    "social_post": "SOCIAL POST / high drift risk",
    "opinion": "OPINION / argument, not evidence",
    "opinion_article": "OPINION ARTICLE / argument, not evidence",
    "unknown": "UNKNOWN / custody not established",
}

# Genre → effective source type (used by genre-aware resolution)
_GENRE_TO_SOURCE_TYPE = {
    "financial_market_snapshot":      "secondary_market_article",
    "global_macro_market_update":     "secondary_market_article",  # Phase 1.6
    "investment_strategy_commentary": "secondary_market_article",  # Phase 1.6
    "news_article":                   "secondary",
    "x_news_post":                    "social",
    "official_release":               "official",
    "regulatory_filing":              "primary",
    "opinion_article":                "opinion",
    "social_post":                    "social",
}
_HIGH_TRUST_TYPES = {"first_hand", "primary", "official", "expert", "regulatory_filing"}
_ARTICLE_GENRES = {
    # Only genres that clearly contradict a declared high-trust type.
    # "generic_article" is the fallback and does NOT trigger mismatch —
    # a genuine SEC filing or research paper may not match any positive signal.
    "financial_market_snapshot", "global_macro_market_update",  # Phase 1.6
    "investment_strategy_commentary",                            # Phase 1.6
    "news_article", "opinion_article", "social_post", "x_news_post",
}
# Form/role compatibility (Arch design note 2026-11-01): a declared document
# FORM ("article") vs an inferred evidentiary ROLE ("secondary") is not a
# custody conflict — an ordinary article being secondary is exactly what to
# expect. Format and role are different axes; only genuinely meaningful
# disagreements (e.g. declared PRIMARY, inferred SECONDARY) stay flagged.
_FORM_TYPES = {"article", "news_article", "generic_article"}
_ROLE_TYPES = {"secondary", "secondary_market_article", "news_article"}
# All financial secondary genres — shared logic (coherence flag, verify cap)
_FINANCIAL_SECONDARY_GENRES = {
    "financial_market_snapshot", "global_macro_market_update",
    "investment_strategy_commentary",
}
INSTITUTIONAL_DOMAINS = (".gov", ".edu", ".mil", ".int")
ACADEMIC_HOSTS = (
    "doi.org", "arxiv.org", "pubmed.ncbi", "ncbi.nlm.nih",
    "jstor.org", "nature.com", "science.org", "ssrn.com", "stanford.edu",
)
SOCIAL_HOSTS = (
    "x.com", "twitter.com", "reddit.com", "tiktok.com",
    "facebook.com", "youtube.com", "instagram.com", "t.me",
    "truthsocial.com", "bsky.app",
)
X_NEWS_RECEIPTS = re.compile(
    r"\b(official\s+(?:statement|record|filing|release)|press\s+conference|"
    r"court\s+record|police\s+statement|fire\s+department|agency\s+statement|"
    r"Reuters|Associated\s+Press|AP|AFP|BBC|source\s+link|video|raw\s+video|"
    r"transcript|document|filing|archive|full\s+clip|livestream)\b",
    re.I,
)
X_NEWS_WEAK_CHAIN = re.compile(
    r"\b(unconfirmed|rumou?r|alleged(?:ly)?|reportedly|sources?\s+say|"
    r"people\s+are\s+saying|hearing|if\s+true|developing|early\s+reports?|"
    r"screenshot|screen.?grab|viral\s+clip|no\s+source|trust\s+me)\b",
    re.I,
)
X_ENGAGEMENT_BAIT = re.compile(
    r"\b(share\s+this|retweet|repost|follow\s+for\s+updates|dm\s+me|"
    r"turn\s+on\s+notifications|like\s+and\s+share|they\s+will\s+delete\s+this)\b",
    re.I,
)
X_TIMESTAMP = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}\s?(?:AM|PM|UTC|EST|EDT|CST|CDT|MST|MDT|PST|PDT)?)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Scores:
    signal: float
    noise: float
    pressure: float
    source_custody: float
    source_custody_raw: float
    source_custody_adjusted: float
    certainty_bias: float
    specificity: float
    evidence: float
    attribution: float
    bias_risk: float
    manipulation_risk: float
    certainty_to_evidence_gap: float
    internal_coherence: float
    overall_confidence: float


@dataclass
class Assessment:
    kind: str = "heuristic"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_name: str = "unknown"
    source_type: str = "unknown"
    declared_source_type: str = "unknown"
    inferred_source_type: str = "unknown"
    source_type_confidence: float = 0.0
    source_url: str = ""
    word_count: int = 0
    language_note: str = ""
    language: Dict = field(default_factory=dict)
    language_assessment: Dict = field(default_factory=dict)
    verdict: str = ""
    recommended_action: str = ""
    triage_summary: str = ""
    scores: Scores | None = None
    confidence_band: Dict[str, float] = field(default_factory=dict)
    custody_label: str = ""
    custody_warning_level: str = "clean"
    evidence_breakdown: Dict[str, int] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    evidence_clues: List[str] = field(default_factory=list)
    pressure_clues: List[str] = field(default_factory=list)
    questions_to_ask: List[str] = field(default_factory=list)
    ai_instruction: str = ""
    # Phase 1.2 additions (present when signal_extract.py is available)
    source_genre: str = "generic_article"
    custody_breakdown: Dict = field(default_factory=dict)
    evidence_shape: Dict = field(default_factory=dict)
    signal_brief: Dict = field(default_factory=dict)
    x_post_assessment: Dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def count_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def snippets_for(pattern: re.Pattern, text: str, max_items: int = 5) -> List[str]:
    """Return short context windows around regex hits for human auditing."""
    out: List[str] = []
    for m in pattern.finditer(text):
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 60)
        s = re.sub(r"\s+", " ", text[start:end].replace("\n", " ")).strip()
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def looks_non_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return False
    ascii_ratio = sum(1 for c in letters if c.isascii()) / len(letters)
    if ascii_ratio >= 0.70:
        return False
    return not (LOGIC.search(text) or HEDGES.search(text) or CERTAINTY.search(text))

# ---------------------------------------------------------------------------
# Source-type inference and resolution
# ---------------------------------------------------------------------------

def _infer_source_type_from_signals(text: str, source_url: str = "") -> str:
    """Infer source type from text content and URL signals only."""
    t = text.lower()
    host = urlparse(source_url).netloc.lower() if source_url else ""
    if any(h in host for h in SOCIAL_HOSTS):
        return "social"
    if any(host.endswith(d) for d in INSTITUTIONAL_DOMAINS):
        return "official"
    if any(h in host for h in ACADEMIC_HOSTS):
        return "primary"
    if any(x in t for x in ["anonymous source", "source familiar", "unnamed source", "rumor", "leaked"]):
        return "anonymous"
    if any(x in t for x in ["press release", "we are excited to announce"]):
        return "official"
    # Hosted release artifact (Arch review 2026-11-05, v1.8.1): a
    # vendor's own release artifact hosted on a third-party page (a
    # model card on Hugging Face, a release on GitHub) is the
    # publisher's own document. Ownership is established by the
    # hierarchy in _hosted_release_artifact — an org-name match alone
    # never fires this. Checked here, after the rumor veto above but
    # before the weak secondary cues ("reported", "according to"),
    # which misfire on release tables ("results reported alongside...").
    if _hosted_release_artifact(text, source_url)[0]:
        return "primary"
    if any(x in t for x in ["abstract", "methodology", "sample size", "doi", "preprint", "sec 10-q", "edgar", "filing"]):
        return "primary"
    if any(x in t for x in ["i saw", "i recorded", "i measured", "my experience"]):
        return "first_hand"
    if any(x in t for x in ["opinion", "essay", "i think", "my take"]):
        return "opinion"
    if any(x in t for x in ["reported", "according to", "sources said"]):
        return "secondary"
    return "unknown"


def infer_source_type(text: str, source_type: str = "unknown", source_url: str = "") -> str:
    if source_type and source_type not in ("auto", "unknown"):
        return source_type.lower()
    return _infer_source_type_from_signals(text, source_url)


def resolve_source_type(declared: str, text: str, source_url: str = "") -> Tuple[str, str, str, float]:
    """Resolve effective source_type, returning (effective, declared_norm, inferred, confidence).

    A declared type that contradicts inferred signals drops confidence and flags
    a potential laundering attempt (e.g. passing source_type='primary' for a rumor).
    """
    declared_norm = (declared or "unknown").lower()
    inferred = _infer_source_type_from_signals(text, source_url)
    if declared_norm in ("auto", "unknown", ""):
        return inferred, declared_norm, inferred, 0.75 if inferred != "unknown" else 0.30
    if inferred == "unknown" or inferred == declared_norm:
        return declared_norm, declared_norm, inferred, 0.90
    if declared_norm in _FORM_TYPES and inferred in _ROLE_TYPES:
        # Declared form vs inferred role: compatible, not a mismatch.
        # Resolve to the role at match confidence so no low-confidence
        # "contradicts inferred" flag fires downstream.
        return inferred, declared_norm, inferred, 0.90
    return declared_norm, declared_norm, inferred, 0.40

# First-party announcement voice: the publisher speaking about its own
# action or policy ("we are announcing/introducing/supporting ...").
# Contractions count: "we're announcing" is the same voice as "we are
# announcing" — the v1.7.0 pattern required whitespace after "we" and
# missed every contracted announcement (Sieve gap 2026-11-04: Google's
# "Today, we're announcing our plans..."). This outranks a URL token
# ("blog") when deciding whether a declared high-trust type is really an
# article: a company blog post announcing its own policy IS a primary
# source. Secondary-reporting cues ("reported", "according to",
# "sources said") veto the exemption — a post reporting on others is
# not a first-party announcement, whatever its verbs.
_ANNOUNCEMENT_RE = re.compile(
    r"\bwe\s*(?:'re|are|will|have)?\s*"
    r"(?:announc|introduc|launch|publish|releas|roll\s*out|implement|support|offer|build|develop)"
    r"|"
    r"\btoday\s*,?\s*we\s*(?:'re|are|will)?\s*"
    r"(?:announc|introduc|launch|support)",
    re.I,
)
_SECONDARY_REPORTING_CUES = ("reported", "according to", "sources said", "sources say")
# Third-party attribution cues (Arch review 2026-11-04): the claim's
# ORIGIN is someone else, not the publisher. A first-party channel
# discussing somebody else's research is not primary evidence FOR that
# research, even when the channel itself is first-party. "Our study"
# (first-person possessive) is not a third-party cue; "a study by X"
# and "researchers found" are.
_THIRD_PARTY_ATTRIBUTION_CUES = (
    "according to", "announced by", "researchers found", "researchers at",
    "a study by", "study conducted by", "research conducted by",
)
_DOI_RE = re.compile(r"\bdoi\s*:\s*10\.\d{4,}/", re.I)


def _first_party_announcement(text: str) -> Tuple[bool, str]:
    """Return (fires, reason) for the first-party announcement exemption.

    Fires only when the publisher speaks in the first person about its
    OWN action or policy AND no cue attributes the main claim to another
    party. The reason names the deciding evidence so a downgrade flag
    can explain itself instead of citing vague "cues in text and URL".
    """
    low = text.lower()
    for cue in _THIRD_PARTY_ATTRIBUTION_CUES:
        if cue in low:
            return False, f"third-party attribution cue present ({cue!r})"
    for cue in _SECONDARY_REPORTING_CUES:
        if cue in low:
            return False, f"secondary-reporting cue present ({cue!r})"
    if _ANNOUNCEMENT_RE.search(text):
        return True, "first-person announcement voice about the publisher's own action"
    return False, "no first-person announcement voice found"


def _has_first_party_primary_evidence(text: str, source_url: str = "") -> bool:
    """Positive first-party evidence for an inferred PRIMARY type.

    An inferred primary is an upgrade: it must be earned, not assumed
    from passing evidence-marker mentions (Arch review 2026-11-04,
    Dealroom case). Evidence is one of: first-party announcement voice;
    an academic/institutional host; the document carrying its own
    primary identifiers (DOI, arXiv, SEC/EDGAR references); or a
    hosted release artifact whose ownership is independently
    established by hosting context (Arch objection 2026-11-05,
    v1.8.2: a candidate whose ownership is not established does NOT
    count as first-party evidence).
    """
    if _first_party_announcement(text)[0]:
        return True
    if _hosted_release_artifact(text, source_url)[0]:
        return True
    host = urlparse(source_url).netloc.lower() if source_url else ""
    if any(h in host for h in ACADEMIC_HOSTS):
        return True
    if any(host.endswith(d) for d in INSTITUTIONAL_DOMAINS):
        return True
    low = text.lower()
    if _DOI_RE.search(text):
        return True
    if "arxiv.org" in low or "sec.gov" in low or "edgar" in low:
        return True
    return False


# Hosted release artifacts — v1.8.2 (Arch objection 2026-11-05): TWO
# SEPARATE questions, because a document must not certify itself merely
# by looking sufficiently official. The v1.8.1 implementation mixed
# them: its authorship phrasing, release metadata, and in-text links
# all originate from the document being evaluated, so a convincing
# fake could establish its own ownership.
#
# Q1 — CANDIDATE: does the document BEHAVE like a first-party release
# artifact? Document-internal signals only: an authorship construction
# naming an org as releaser/author, plus release metadata and/or an
# official-looking link inside the text. The claimed org is read OUT
# OF THE TEXT, never taken from the URL — the URL answers Q2, not Q1.
# A candidate is NOT a verified primary; it is a first-party claim
# awaiting independent corroboration.
#
# Q2 — OWNERSHIP: can we INDEPENDENTLY establish that the claimed
# publisher controls the artifact? Only hosting-context signals count
# — signals the document cannot write for itself:
#   (a) platform namespace: the source URL is on a known artifact
#       host (huggingface.co, github.com, gitlab.com) and its first
#       path segment IS the claimed org slug. The platform assigned
#       that namespace; the document cannot choose where it is hosted.
#   (b) own-domain hosting: the source URL's host is the claimed
#       org's own domain (the org slug appears as a non-TLD label of
#       the host). The org controls its domain; the document does not
#       control where it was fetched from.
# A verified organization association (e.g. a platform's verified-org
# badge) would be a third Q2 signal, but no registry is available to
# this system — it is a documented future extension, never silently
# approximated from text. A namespace squatter (an attacker who
# controls an org namespace named after a real company) is likewise
# out of scope for this pass; the namespace check is the independent
# signal available here, and its limits are stated, not hidden.
#
# If Q1 holds and Q2 does not, the document is preserved as a
# candidate primary source with uncertain ownership — never promoted
# to verified primary, never allowed to anchor.
_ARTIFACT_HOSTS = ("huggingface.co", "github.com", "gitlab.com")
_RELEASE_METADATA_MARKERS = (
    "model card", "license", "release", "version", "parameters",
    "pre-train", "pretrain", "checkpoint", "weights",
)

_ORG_STOPWORDS = {
    "the", "our", "this", "that", "these", "those", "a", "an", "and",
    "by", "for", "with", "its", "their", "his", "her", "your", "my",
    "one", "two", "first", "last", "new", "all", "any",
}

# Authorship constructions that name a releaser/author, in priority
# order. The captured group is the claimed org. Patterns are
# case-sensitive on the captured name (it must start uppercase in the
# text) so lowercase prose ("the team", "by releasing") never mints
# an org. Tried in order: a "Team" construction outranks a looser
# release-verb mention, so "Baidu ERNIE Team" claims Baidu, not
# "Hugging" (which merely precedes "released" in the same header).
_CLAIMED_ORG_RES = [
    # "X Team" with up to two words between ("Baidu ERNIE Team").
    re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,30})(?:\s+[A-Za-z][A-Za-z0-9_-]{1,30}){0,2}\s+[Tt]eam\b"),
    # "X Labs / Research / AI".
    re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,30})\s+(?:[Ll]abs?|[Rr]esearch|[Aa][Ii])\b"),
    # "by X".
    re.compile(r"\b[Bb]y\s+([A-Z][A-Za-z0-9_-]{1,30})\b"),
    # "X's ...".
    re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,30})'s\b"),
    # "Copyright ... X".
    re.compile(r"[Cc]opyright[^.\n]{0,60}?\b([A-Z][A-Za-z0-9_-]{1,30})\b"),
    # "X ... released/announced/introduced/..." (loosest; last resort).
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]{1,30})\b[^.!?\n]{0,60}?"
        r"\b(?:[Rr]eleas|[Aa]nnounc|[Ii]ntroduc|[Pp]resent|[Pp]ublish|[Ll]aunch)\w*\b"
    ),
]


def _extract_claimed_org(text: str) -> str:
    """The org the document itself names as releaser/author, read out
    of the text's authorship constructions. Returns the lowercased slug
    or "" when no construction names one. Document-internal by
    definition: it says who the document CLAIMS to be from, never who
    actually controls the artifact."""
    for pat in _CLAIMED_ORG_RES:
        seen: Dict[str, int] = {}
        for m in pat.finditer(text):
            org = m.group(1)
            if len(org) < 2 or org.lower() in _ORG_STOPWORDS:
                continue
            seen[org.lower()] = seen.get(org.lower(), 0) + 1
        if seen:
            # Most-mentioned org from this construction wins; ties go
            # to the first mentioned (dict preserves insertion order).
            return max(seen.items(), key=lambda kv: kv[1])[0]
    return ""


def _hosted_artifact_candidate(text: str, source_url: str = "") -> Tuple[bool, str, str]:
    """Q1 — does the document behave like a first-party release artifact?

    Returns (candidate, claimed_org, reason). Document-internal signals
    only: the source URL must be on a known artifact host (the doc is a
    hosted artifact at all), the text must name an org as releaser/author
    (an authorship construction, not a bare name match), and the text
    must carry release metadata and/or an official-looking link. None of
    this establishes ownership — that is Q2's job.
    """
    if not source_url:
        return False, "", "no source URL: cannot be a hosted artifact"
    host = urlparse(source_url).netloc.lower()
    if not any(host == h or host.endswith("." + h) for h in _ARTIFACT_HOSTS):
        return False, "", "source URL is not on a known artifact host"
    claimed = _extract_claimed_org(text)
    if not claimed:
        return False, "", "no org named as releaser/author in the text"
    low = text.lower()
    # Official-looking link INSIDE the text (document-internal: a doc
    # can link anywhere, so this supports candidacy only).
    official_link = False
    for u in re.findall(r"https?://\S+", text):
        labels = urlparse(u).netloc.lower().split(".")
        if claimed in labels[:-1]:
            official_link = True
            break
    metadata_hits = [m for m in _RELEASE_METADATA_MARKERS if m in low]
    has_metadata = len(metadata_hits) >= 2
    if official_link:
        return True, claimed, (
            f"org {claimed!r} named as releaser/author + official link "
            "to the org's domain in the text"
        )
    if has_metadata:
        return True, claimed, (
            f"org {claimed!r} named as releaser/author + release metadata "
            f"({', '.join(metadata_hits[:4])})"
        )
    return False, claimed, (
        f"org {claimed!r} named as releaser/author but neither an official "
        "link nor release metadata in the text"
    )


def _hosted_ownership_verified(
    text: str, source_url: str = "", claimed_org: str = ""
) -> Tuple[bool, str]:
    """Q2 — can we INDEPENDENTLY establish that the claimed publisher
    controls the artifact?

    Only hosting-context signals count: the platform namespace (the
    source URL's first path segment on a known artifact host IS the
    claimed org slug) or own-domain hosting (the source URL's host is
    the claimed org's own domain). Both are facts about where the
    document was fetched from, not things the document says about
    itself. Document-internal signals — authorship phrasing, metadata,
    links inside the text — can NEVER establish this on their own.
    Returns (verified, reason).
    """
    org = (claimed_org or _extract_claimed_org(text)).lower()
    if not org:
        return False, "no claimed publisher named in the text"
    if not source_url:
        return False, "no source URL: ownership cannot be established independently"
    parts = [p for p in urlparse(source_url).path.split("/") if p]
    host = urlparse(source_url).netloc.lower()
    # (a) Platform namespace: the artifact host assigned this org slug.
    # This check decides for artifact hosts — own-domain hosting below
    # never runs for them, so a doc claiming "Hugging Face" under
    # somebody else's namespace cannot ride on the huggingface.co host.
    if any(host == h or host.endswith("." + h) for h in _ARTIFACT_HOSTS):
        if parts and parts[0].lower() == org:
            return True, (
                f"source URL namespace {parts[0]!r} on {host} matches "
                f"claimed publisher {org!r}"
            )
        slug = parts[0] if parts else "(no org path segment)"
        return False, (
            f"source URL namespace {slug!r} does not match claimed "
            f"publisher {org!r}"
        )
    # (b) Own-domain hosting: the claimed org's own domain.
    labels = host.split(".")
    if org in labels[:-1]:
        return True, (
            f"source URL host {host} is the claimed publisher's own domain"
        )
    return False, (
        f"source URL host {host} neither matches the claimed publisher's "
        "artifact namespace nor its own domain"
    )


def _hosted_release_artifact(text: str, source_url: str = "") -> Tuple[bool, str]:
    """Verified hosted first-party evidence: the document behaves like a
    first-party release artifact (Q1, candidate) AND its ownership is
    independently established by hosting context (Q2, verified).

    Returns (fires, reason). A candidate whose ownership is not
    established does NOT fire here — it is preserved as a candidate
    primary with uncertain ownership by the caller, never promoted.
    The reason names both stages, so a custody decision can explain
    itself instead of citing vague "cues".
    """
    candidate, org, cand_reason = _hosted_artifact_candidate(text, source_url)
    if not candidate:
        return False, f"not a hosted release artifact candidate: {cand_reason}"
    verified, ver_reason = _hosted_ownership_verified(text, source_url, org)
    if not verified:
        return False, (
            f"hosted artifact candidate (claimed publisher {org!r}: "
            f"{cand_reason}) but ownership not independently established: "
            f"{ver_reason}"
        )
    return True, (
        f"hosted release artifact: {cand_reason}; ownership independently "
        f"established: {ver_reason}"
    )


def _gate_inferred_primary(
    declared_norm: str, inferred: str, text: str, source_url: str
) -> Tuple[str, bool]:
    """Gate an inferred PRIMARY behind positive first-party evidence.

    Returns (inferred, gated). When inferred is "primary" but nothing
    establishes that this publisher originated the claim, the inference
    drops to "secondary" and the caller must flag it with Arch's
    custody-uncertain wording. Non-primary inferences pass through.
    """
    if inferred == "primary" and not _has_first_party_primary_evidence(text, source_url):
        return "secondary", True
    return inferred, False


def resolve_source_type_v2(
    declared: str, text: str, source_url: str, genre: str
) -> Tuple[str, str, str, float, bool, Dict]:
    """Genre-aware source resolution.

    Returns (effective, declared_norm, inferred, confidence,
    genre_mismatch, notes). notes carries:
      first_party_exemption : bool — the exemption held the declared type
      exemption_reason      : str  — the deciding evidence, for flag text
      inferred_primary_gated: bool — an inferred PRIMARY was gated to
        SECONDARY for lack of first-party evidence (Arch 2026-11-04)
    genre_mismatch is True when the caller declared a high-trust type but the
    text/URL signals an article or secondary source.
    """
    effective, declared_norm, inferred, confidence = resolve_source_type(declared, text, source_url)
    genre_st = _GENRE_TO_SOURCE_TYPE.get(genre)
    genre_mismatch = False

    # Inferred-primary gate (Arch review 2026-11-04, Dealroom case): an
    # inferred PRIMARY with no positive first-party evidence is not
    # reported as primary at all. It drops to secondary — the safe
    # default for an article-form document — and the caller flags it
    # with Arch's custody-uncertain wording.
    inferred, gated = _gate_inferred_primary(declared_norm, inferred, text, source_url)
    if gated and declared_norm in _FORM_TYPES:
        effective = "secondary"
        confidence = max(confidence, 0.70)

    # First-party announcement exemption (Sieve gap 2026-11-02, generalized
    # 2026-11-04 per Arch): a declared high-trust type on a first-person
    # announcement keeps its type even when the genre signals an article.
    # The trigger is ownership + claim origin (the publisher's own voice
    # about its own action, no third-party attribution), NOT the URL
    # token — a blog URL neither grants nor denies the exemption. This
    # does NOT grant anchor status — custody, manipulation, and rumor
    # gates still apply downstream; it only stops the genre downgrade.
    fires, reason = _first_party_announcement(text)
    if (
        declared_norm in _HIGH_TRUST_TYPES
        and genre in _ARTICLE_GENRES
        and fires
    ):
        return declared_norm, declared_norm, inferred, max(confidence, 0.70), False, {
            "first_party_exemption": True,
            "exemption_reason": reason,
            "inferred_primary_gated": gated,
        }

    # Declared high-trust but genre signals article/secondary → mismatch
    if declared_norm in _HIGH_TRUST_TYPES and genre in _ARTICLE_GENRES:
        confidence = min(confidence, 0.40)
        genre_mismatch = True
        if genre_st:
            effective = genre_st

    # Auto-inferred + genre gives a more specific type → use it
    elif declared_norm in ("auto", "unknown", "") and genre_st:
        effective = genre_st
        confidence = max(confidence, 0.70)

    return effective, declared_norm, inferred, confidence, genre_mismatch, {
        "first_party_exemption": False,
        "exemption_reason": reason,
        "inferred_primary_gated": gated,
    }


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------

def url_custody_bonus(text: str) -> Tuple[float, Dict[str, int]]:
    urls = re.findall(r"https?://\S+", text)
    counts = {"institutional": 0, "academic": 0, "social": 0, "web": 0}
    bonus = 0.0
    for u in urls:
        host = urlparse(u).netloc.lower()
        if any(host.endswith(d) for d in INSTITUTIONAL_DOMAINS):
            counts["institutional"] += 1
            bonus += 0.04
        elif any(h in host for h in ACADEMIC_HOSTS):
            counts["academic"] += 1
            bonus += 0.05
        elif any(h in host for h in SOCIAL_HOSTS):
            counts["social"] += 1
            bonus -= 0.01
        else:
            counts["web"] += 1
            bonus += 0.015
    return max(-0.10, min(0.18, bonus)), counts


def source_custody_score(text: str, source_type: str, source_url: str = "") -> Tuple[float, str, List[str]]:
    st = infer_source_type(text, source_type, source_url)
    base = SOURCE_TYPE_WEIGHTS.get(st, SOURCE_TYPE_WEIGHTS["unknown"])
    evidence_hits = count_matches(EVIDENCE_MARKERS, text)
    anon_hits = len(re.findall(r"\b(anonymous|unnamed|source familiar|people are saying|rumor)\b", text, re.I))
    quote_hits = count_matches(QUOTE_PAT, text)
    url_bonus, url_counts = url_custody_bonus(text)
    score = base + min(evidence_hits * 0.035, 0.18) + url_bonus + min(quote_hits * 0.02, 0.08) - min(anon_hits * 0.07, 0.25)
    clues: List[str] = []
    if evidence_hits:
        clues.append(f"evidence markers: {evidence_hits}")
    if anon_hits:
        clues.append(f"anonymous/rumor markers: {anon_hits}")
    if quote_hits:
        clues.append(f"long quoted passages: {quote_hits}")
    for k, label in [
        ("institutional", "institutional links (.gov/.edu/.mil)"),
        ("academic", "academic links (DOI/arXiv/PubMed)"),
        ("social", "social-platform links"),
        ("web", "other web links"),
    ]:
        if url_counts[k]:
            clues.append(f"{label}: {url_counts[k]}")
    return clamp(score), SOURCE_LABELS.get(st, SOURCE_LABELS["unknown"]), clues


def evidence_breakdown(text: str, sent_count: int) -> Tuple[float, int, int, List[str]]:
    """Count anchored vs unanchored evidence markers.

    'Anchored' means the marker has a nearby number, URL, date, or proper-noun
    pair inside the text. This does NOT mean externally verified — a manipulator
    can salt anchors near evidence words. Use alongside pressure and certainty_bias.
    """
    anchored = 0
    unanchored = 0
    snippets: List[str] = []
    for m in EVIDENCE_MARKERS.finditer(text):
        ws = max(0, m.start() - 80)
        we = min(len(text), m.end() + 80)
        window = text[ws:we]
        if SPEC_WINDOW.search(window):
            anchored += 1
        else:
            unanchored += 1
        if len(snippets) < 5:
            snippets.append(re.sub(r"\s+", " ", window).strip())
    weighted = anchored + unanchored * 0.3
    return clamp(weighted / max(sent_count, 1)), anchored, unanchored, snippets


def shouting_score(text: str) -> float:
    """Detect excessive all-caps without penalising legitimate acronyms (NASA, SEC, etc.)."""
    long_words = re.findall(r"\b[A-Za-z]{4,}\b", text)
    if len(long_words) < 8:
        return 0.0
    all_caps = sum(1 for w in long_words if w.isupper())
    if all_caps < 4:
        return 0.0
    ratio = all_caps / len(long_words)
    return 0.0 if ratio < 0.20 else clamp((ratio - 0.20) * 5.0)


def attribution_score(text: str) -> float:
    """Ratio of quoted passages with a nearby name + attribution verb."""
    attributed = 0
    bare = 0
    for m in QUOTE_PAT.finditer(text):
        window = text[max(0, m.start() - 120):min(len(text), m.end() + 120)]
        if ATTRIBUTION_VERB.search(window) and NAME_PAT.search(window):
            attributed += 1
        else:
            bare += 1
    return 0.0 if attributed + bare == 0 else clamp(attributed / (attributed + bare))


def assess_x_news_post(text: str, source_url: str = "") -> Dict:
    """Return a receipt-focused assessment for X/Twitter breaking-news posts."""
    urls = re.findall(r"https?://\S+", text)
    hosts = [urlparse(u).netloc.lower() for u in urls]
    non_social_links = [
        u for u, host in zip(urls, hosts)
        if not any(h in host for h in SOCIAL_HOSTS) and "t.co" not in host
    ]
    primary_links = [
        u for u, host in zip(urls, hosts)
        if any(host.endswith(d) for d in INSTITUTIONAL_DOMAINS)
        or any(h in host for h in ACADEMIC_HOSTS)
        or any(h in host for h in ("reuters.com", "apnews.com", "sec.gov", "courtlistener.com"))
    ]
    receipt_hits = count_matches(X_NEWS_RECEIPTS, text) + len(primary_links)
    weak_hits = count_matches(X_NEWS_WEAK_CHAIN, text)
    bait_hits = count_matches(X_ENGAGEMENT_BAIT, text)
    has_timestamp = bool(X_TIMESTAMP.search(text))
    has_non_social_link = bool(non_social_links or primary_links)
    receipt_score = clamp(
        receipt_hits * 0.22
        + (0.20 if has_timestamp else 0.0)
        + (0.22 if has_non_social_link else 0.0)
        - weak_hits * 0.16
        - bait_hits * 0.18
    )
    return {
        "kind": "x_news_post",
        "receipt_score": round(receipt_score, 3),
        "receipt_markers": receipt_hits,
        "weak_chain_markers": weak_hits,
        "engagement_bait_markers": bait_hits,
        "has_timestamp": has_timestamp,
        "has_non_social_link": has_non_social_link,
        "primary_links": primary_links[:3],
        "non_social_links": non_social_links[:3],
    }

# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

ACTION_IMPERATIVES = {
    "use_as_anchor": "Anchor-grade primary source — suitable to anchor the evidence chain for the claim it directly supports. Custody is proximity, not proof of truth.",
    "verify_primary": "Usable signal; verify key claims at the primary source.",
    "treat_as_lead": "Treat as lead only; do not pass forward as evidence.",
    "seek_receipts": "Request receipts before trusting.",
    "drop": "Do not pass forward without independent verification.",
    "reject": "Reject as evidence.",
}


def custody_warning_level(custody: float, st_confidence: float, declared_st: str = "") -> str:
    # "mismatch" means the CALLER declared a source type the text contradicts.
    # An undeclared/auto analysis with weak inference is "uncertain", not a
    # mismatch: the caller declared nothing, so nothing can be mismatched.
    # (Previously every auto-mode text with unknown inference reported
    # "Source-type declaration mismatch" in the triage summary.)
    caller_declared = (declared_st or "").lower() not in ("", "auto", "unknown")
    if st_confidence < 0.50:
        return "mismatch" if caller_declared else "uncertain"
    if custody < 0.45:
        return "weak"
    if custody < 0.70:
        return "watch"
    return "clean"


def build_triage_summary(
    *,
    custody_warning: str,
    certainty_bias: float,
    pressure: float,
    attack_hits: int,
    anchored_n: int,
    unanchored_n: int,
    action: str,
) -> str:
    parts = [{"weak": "Weak custody", "mismatch": "Source-type declaration mismatch", "uncertain": "Source type uncertain", "watch": "Mid custody", "clean": "Solid custody"}.get(custody_warning, "Unknown custody state")]
    if attack_hits >= 1:
        parts.append("attack language present")
    if pressure >= 0.55:
        parts.append("high pressure language")
    if certainty_bias >= 0.55:
        parts.append("high certainty bias")
    if unanchored_n > anchored_n and unanchored_n >= 2:
        parts.append("unanchored evidence terms")
    return f"{' + '.join(parts)}. {ACTION_IMPERATIVES.get(action, '')}".strip()

# ---------------------------------------------------------------------------
# CFT-inspired internal coherence scorer
# ---------------------------------------------------------------------------

def _score_internal_coherence(words: list, sentences: list) -> float:
    """Estimate internal logical coherence of the text (CFT-inspired).

    Measures three signals:
      1. Logical connective density — structured argument flow (therefore,
         because, however, although, etc.)
      2. Lexical diversity — repetitive text scores low; varied vocab scores high.
      3. Sentence-length variance — natural prose varies; mechanical repetition
         or copy-paste padding shows as very uniform sentence lengths.

    High score (>0.65): well-structured, non-repetitive argument.
    Mid score (0.35–0.65): typical web writing, mixed quality.
    Low score (<0.35): repetitive, circular, or logically unconnected.

    This does NOT assess truth — a coherent argument can still be wrong.
    """
    if len(words) < 10:
        return 0.50

    text_lower = " ".join(words).lower()

    # 1. Logical connective density
    connective_hits = len(CONNECTIVES.findall(text_lower))
    # 0.25 connectives/sentence is a healthy rate for structured writing
    connective_score = min(connective_hits / max(len(sentences), 1) / 0.25, 1.0)

    # 2. Lexical diversity  (unique / total word tokens)
    lex_div = len(set(w.lower() for w in words)) / len(words)
    # <0.25 = very repetitive, 0.65+ = highly diverse
    diversity_score = min(max((lex_div - 0.25) / 0.40, 0.0), 1.0)

    # 3. Sentence length variance (coefficient of variation)
    if len(sentences) >= 3:
        lens = [len(s.split()) for s in sentences if s.strip()]
        if len(lens) >= 3:
            mean_l = sum(lens) / len(lens)
            cv = (sum((l - mean_l) ** 2 for l in lens) / len(lens)) ** 0.5 / max(mean_l, 1)
            variance_score = min(cv / 0.5, 1.0)
        else:
            variance_score = 0.5
    else:
        variance_score = 0.5

    return round(clamp(0.45 * connective_score + 0.40 * diversity_score + 0.15 * variance_score), 3)


# ---------------------------------------------------------------------------
# Main analyze()
# ---------------------------------------------------------------------------

def analyze(
    text: str,
    *,
    source_type: str = "auto",
    source_name: str = "unknown",
    source_url: str = "",
    language: Optional[str] = "en",
) -> Dict:
    """Analyze text and return a JSON-serializable assessment dict.

    Parameters
    ----------
    text        : The text to analyze.
    source_type : Custody hint — auto|first_hand|primary|official|expert|
                  secondary|tertiary|anonymous|social|opinion|unknown.
                  'auto' runs shallow inference from the text and URL.
    source_name : Publication, person, or system that produced the text.
    source_url  : URL of the source if available (improves custody inference).
    language    : Declared language code. English is the backward-compatible
                  default; use Spanish ("es") for the starter Spanish pack,
                  or another code for honest structural-only fallback.
    """
    raw = text or ""
    normalized = re.sub(r"\s+", " ", raw).strip()
    words = re.findall(r"\b\w+\b", normalized)
    word_count = len(words)
    sentences = sentence_split(raw)
    sent_count = max(len(sentences), 1)
    language_assessment: Dict = {}
    language_routing: Dict = {}
    if _I18N_AVAILABLE:
        language_assessment = analyze_language_layer(raw, language=language)
        language_routing = language_assessment.get("language", {})

    if language_routing.get("mode") == "structural_only":
        language_note = language_routing.get(
            "note",
            "Lexical risk is unavailable for this language; structural-only guardrails apply.",
        )
    elif language_routing.get("calibration") == "starter_partial":
        language_note = (
            "Spanish starter lexicon active. Results include language-aware risk checks, "
            "but the pack is not yet corpus-calibrated to English parity."
        )
    elif language_routing.get("warning"):
        language_note = language_routing["warning"]
    else:
        language_note = (
            "Text appears non-English but no language-aware layer was available; "
            "scores will be conservative."
            if looks_non_english(raw) else ""
        )

    # Raw hit counts
    cert_hits = count_matches(CERTAINTY, raw)
    hedge_hits = count_matches(HEDGES, raw)
    hype_hits = count_matches(HYPE, raw)
    urgent_hits = count_matches(URGENCY, raw)
    spec_hits = count_matches(SPECIFICITY, raw)
    logic_hits = count_matches(LOGIC, raw)
    bias_hits = count_matches(BIAS_INCENTIVE, raw)
    attack_hits = count_matches(ATTACK_LANGUAGE, raw)
    filler_hits = count_matches(FILLER, raw)
    exclaims = raw.count("!")

    # Genre detection (Phase 1.2 — requires signal_extract.py)
    source_genre = "generic_article"
    if _EXTRACT_AVAILABLE:
        source_genre = detect_genre(normalized, source_url)
    x_post_assessment: Dict = {}
    if source_genre == "x_news_post":
        x_post_assessment = assess_x_news_post(raw, source_url)

    # Custody — genre-aware resolution prevents article pages from being classed as primary
    if _EXTRACT_AVAILABLE:
        effective_st, declared_st, inferred_st, st_confidence, genre_mismatch, resolve_notes = \
            resolve_source_type_v2(source_type, raw, source_url, source_genre)
    else:
        effective_st, declared_st, inferred_st, st_confidence = \
            resolve_source_type(source_type, raw, source_url)
        genre_mismatch = False
        inferred_st, gated_fallback = _gate_inferred_primary(declared_st, inferred_st, raw, source_url)
        if gated_fallback and declared_st in _FORM_TYPES:
            effective_st = "secondary"
            st_confidence = max(st_confidence, 0.70)
        resolve_notes = {
            "first_party_exemption": False,
            "exemption_reason": _first_party_announcement(raw)[1],
            "inferred_primary_gated": gated_fallback,
        }

    # Hosted-artifact ownership separation (Arch objection 2026-11-05,
    # v1.8.2): Q1 (behaves like a first-party release artifact) and Q2
    # (ownership independently established by hosting context) are
    # separate questions. A candidate whose ownership is NOT established
    # is preserved as a candidate primary with uncertain ownership: its
    # source-type confidence is capped below the verified-primary bar
    # (0.70) so it can never anchor and gets the custody question, and
    # it is flagged by name below. It is NOT silently demoted to
    # secondary — the candidacy itself is evidence worth keeping
    # visible. The cap does not apply when independent first-party
    # evidence exists (DOI, announcement voice, academic/institutional
    # host): then the primary standing does not rest on the hosted
    # claim at all.
    hosted_candidate, hosted_claimed_org, hosted_candidate_reason = \
        _hosted_artifact_candidate(raw, source_url)
    hosted_verified, hosted_verified_reason = \
        _hosted_ownership_verified(raw, source_url, hosted_claimed_org)
    hosted_ownership_uncertain = bool(
        hosted_candidate
        and not hosted_verified
        and not _has_first_party_primary_evidence(raw, source_url)
    )
    if hosted_ownership_uncertain:
        st_confidence = min(st_confidence, 0.60)

    custody_raw, custody_label, custody_clues = source_custody_score(raw, effective_st, source_url)
    # st_confidence scales custody: prevents laundering weak text through a declared 'primary' label
    custody = custody_raw * (0.65 + 0.35 * st_confidence)

    # Sub-scores
    specificity = clamp((spec_hits / sent_count) * 0.9)
    evidence, anchored_n, unanchored_n, evidence_snips = evidence_breakdown(raw, sent_count)
    attribution = attribution_score(raw)
    logic_density = clamp((logic_hits / sent_count) * 0.8)
    shouting = shouting_score(raw)

    # Density-normalized pressure components
    cert_density = cert_hits / sent_count
    hedge_density = hedge_hits / sent_count
    hype_density = hype_hits / sent_count
    urgent_density = urgent_hits / sent_count
    exclaim_density = exclaims / sent_count
    attack_density = attack_hits / sent_count
    filler_density = filler_hits / sent_count
    # Filler is less punitive for natural speech contexts
    filler_load = filler_density * (0.15 if effective_st in ("first_hand", "social") else 0.25)

    signal = clamp(0.25 + specificity * 0.25 + evidence * 0.25 + logic_density * 0.15 + custody * 0.20 + attribution * 0.05)
    pressure = clamp(hype_density * 1.2 + urgent_density * 1.5 + cert_density * 0.5 + exclaim_density * 0.4 + shouting * 0.6 + attack_density * 0.9)
    custody_gap = max(0.0, 0.55 - custody)
    evidence_gap = max(0.0, 0.40 - evidence)
    certainty_bias = clamp(cert_density * 1.3 - hedge_density * 0.5 + custody_gap * 0.4 + evidence_gap * 0.3)
    bias_risk = clamp(bias_hits * 0.07 + attack_hits * 0.05 + (0.12 if "sponsored" in raw.lower() else 0.0))
    manipulation_risk = clamp(pressure * 0.48 + certainty_bias * 0.32 + bias_risk * 0.20)
    noise = clamp(0.18 + filler_load + pressure * 0.32 + (1.0 - signal) * 0.30)
    overall = clamp(signal * 0.40 + custody * 0.35 + evidence * 0.15 + attribution * 0.05 - manipulation_risk * 0.30 - certainty_bias * 0.15)

    # Hard caps: high pressure or attack language puts a ceiling on overall confidence
    if pressure >= 0.65:
        overall = min(overall, 0.40)
    if attack_hits >= 1:
        overall = min(overall, 0.40)
    if attack_hits >= 2:
        overall = min(overall, 0.25)
    # Short-text cap: a dense but tiny passage can't earn high confidence on its own
    if word_count < 50:
        overall = min(overall, 0.55)
    if x_post_assessment:
        receipt_score = x_post_assessment.get("receipt_score", 0.0)
        weak_chain_hits = x_post_assessment.get("weak_chain_markers", 0)
        bait_hits = x_post_assessment.get("engagement_bait_markers", 0)
        # X news can be useful as a lead, but social custody is not receipt-grade
        # without off-platform or primary confirmation.
        overall = min(overall, 0.62 if receipt_score >= 0.55 else 0.42)
        if weak_chain_hits or bait_hits:
            manipulation_risk = clamp(manipulation_risk + weak_chain_hits * 0.04 + bait_hits * 0.06)
            certainty_bias = clamp(certainty_bias + weak_chain_hits * 0.04)

    internal_coherence = _score_internal_coherence(words, sentences)
    certainty_to_evidence_gap = certainty_bias - evidence
    band_width = 0.10 + (manipulation_risk + certainty_bias) * 0.12
    confidence_low = clamp(overall - band_width)
    confidence_high = clamp(overall + band_width * 0.4)

    # Flags
    flags: List[str] = []
    if custody < 0.45:
        flags.append("Weak source custody: treat as lead, not conclusion.")
    if certainty_bias > 0.55:
        flags.append("Certainty bias: claims sound more final than support appears to justify.")
    if pressure > 0.55:
        flags.append("High pressure language: possible manipulation, hype, panic, or sales framing.")
    if evidence < 0.20 and cert_hits >= 2:
        flags.append("Strong claims with low evidence markers: demand receipts.")
    if unanchored_n > anchored_n and (anchored_n + unanchored_n) >= 2:
        flags.append("Evidence terms appear without nearby numbers, links, or names: unanchored, not anchored.")
    if bias_risk > 0.35:
        flags.append("Possible incentive/bias language: inspect who benefits.")
    if attack_hits:
        flags.append("Attack/dehumanizing language detected: may be identity bait, not analysis.")
    if word_count < 50:
        flags.append("Short text: limited context; confidence capped.")
    if internal_coherence < 0.30 and word_count >= 50:
        if source_genre in ("financial_market_snapshot", "global_macro_market_update"):
            flags.append(
                "Table-heavy market snapshot: argument coherence scoring is not the primary "
                "signal mode for this genre — use evidence shape and signal brief instead."
            )
        else:
            flags.append("Low internal coherence: argument appears repetitive or logically unconnected.")
    # Phase 1.6: house-view and forward-looking flags for investment strategy commentary
    if source_genre == "investment_strategy_commentary":
        flags.append(
            "House view / institutional positioning: allocation claims reflect "
            "an asset manager's strategic thesis — not independent primary financial data."
        )
        flags.append(
            "Forward-looking allocation claims: overweight/underweight/tactical views "
            "are expressions of conviction, not receipt-grade assertions."
        )
    if x_post_assessment:
        flags.append(
            "X news post: treat as a time-sensitive lead until primary/off-platform receipts are checked."
        )
        if not x_post_assessment.get("has_timestamp"):
            flags.append("X news post lacks a clear timestamp/time zone for the reported event.")
        if not x_post_assessment.get("has_non_social_link"):
            flags.append("X news post lacks a non-social source link; watch for repost-chain drift.")
        if x_post_assessment.get("weak_chain_markers", 0) > 0:
            flags.append("X news post uses unconfirmed/rumor-chain language; seek independent receipts.")
        if x_post_assessment.get("engagement_bait_markers", 0) > 0:
            flags.append("X news post includes engagement-bait cues; separate reporting from spread incentives.")
    if language_note:
        flags.append(language_note)
    if resolve_notes.get("inferred_primary_gated"):
        # Arch review 2026-11-04: an inferred PRIMARY with no first-party
        # evidence is custody-uncertain, not "trusted at low confidence".
        flags.append(
            "Source custody uncertain: available evidence does not establish "
            "that this publisher originated the claim. Treat as a lead pending verification."
        )
    if hosted_ownership_uncertain:
        # Arch objection 2026-11-05 (v1.8.2): a document that behaves
        # like a first-party release artifact but whose ownership is
        # not independently established must not certify itself. It is
        # preserved as a candidate primary with uncertain ownership.
        flags.append(
            "Hosted release artifact claims first-party ownership "
            f"(candidate: {hosted_candidate_reason}) but ownership is not "
            f"independently established ({hosted_verified_reason}). "
            "Candidate primary source with uncertain ownership — treat as "
            "a lead pending verification."
        )
    if st_confidence < 0.50 and declared_st not in ("auto", "unknown", "") and not genre_mismatch:
        flags.append(
            f"Declared source_type='{declared_st}' contradicts inferred='{inferred_st}'. "
            f"Custody is discounted for the contradiction; treat as a lead pending verification."
        )
    if genre_mismatch:
        # The flag names the deciding evidence (genre + why the
        # first-party exemption did not apply) instead of vague "cues".
        flags.append(
            f"Declared high-trust source type ('{declared_st}') conflicts with "
            f"genre '{source_genre}' (secondary-reporting signals); first-party "
            f"exemption did not apply: {resolve_notes.get('exemption_reason', 'unknown')}. "
            f"Effective type downgraded to '{effective_st}'."
        )

    # Verdict and recommended_action
    if attack_hits >= 1 and pressure >= 0.55:
        verdict, recommended_action = "manipulation + attack language — reject as evidence", "reject"
    elif manipulation_risk >= 0.65:
        verdict, recommended_action = "high-pressure / low-trust — do not pass forward without receipts", "drop"
    elif (
        effective_st in ("primary", "first_hand", "regulatory_filing")
        and custody >= 0.70
        and not genre_mismatch
        and st_confidence >= 0.70
        and manipulation_risk < 0.35
        and not re.search(
            r"\b(anonymous|unnamed|source familiar|people are saying|rumor)\b",
            raw, re.I,
        )
    ):
        # Anchor-grade primary (Arch design note 2026-11-01): a clean
        # primary source earns "use_as_anchor" — suitable to anchor the
        # evidence chain for the claim it directly supports. Custody is
        # proximity, not proof of truth, so this is deliberately NOT an
        # overall-confidence gate: custody and epistemic confidence are
        # separate axes, and overall is not consulted here.
        # Operationalization of Arch's conditions:
        #   custody PRIMARY     -> effective type in the primary-grade family
        #   identity verified   -> st_confidence >= 0.70 (not a laundered label)
        #                          AND no anonymous/rumor sourcing language:
        #                          a "primary" that talks like a rumor has
        #                          not verified its provenance
        #   no material flags   -> no genre mismatch, no contradiction flag
        #                          (st_confidence >= 0.70 subsumes the latter)
        #   no contradiction    -> manipulation_risk < 0.35 (same bar as below)
        #   claim actually present + scope/version/date appropriate
        #                       -> the CALLER's check when anchoring a
        #                       specific claim; analyze() sees documents,
        #                       not claims, so it cannot verify them.
        verdict, recommended_action = (
            "primary source — suitable to anchor the evidence chain for the claim "
            "it directly supports; custody is proximity, not proof of truth",
            "use_as_anchor",
        )
    elif (overall >= 0.70 or (overall >= 0.62 and custody >= 0.85)) \
            and custody >= 0.70 and manipulation_risk < 0.35:
        # Calibration fix (2026-10-22): the flat 0.70 overall bar made
        # verify_primary unreachable in practice — density-normalized signal
        # scoring keeps even textbook-grade primary texts (trial abstracts with
        # n, CIs, p-values, registration, DOI) in the 0.60-0.68 band. The
        # 0.62 + custody>=0.85 path lets exceptional, cleanly-scored custody
        # compensate. Manipulation_risk and the custody floor still gate, so
        # confident scams and clean-wrapper hype stay out (see pinned issue #1).
        verdict, recommended_action = "usable signal — still verify key claims at the primary source", "verify_primary"
    elif overall >= 0.50:
        verdict, recommended_action = "mixed signal — read, verify, and separate facts from framing", "treat_as_lead"
    elif certainty_bias >= 0.50 or unanchored_n > anchored_n:
        verdict, recommended_action = "claims outpace support — request receipts", "seek_receipts"
    else:
        # Custody and confidence are separate axes (Arch design note
        # 2026-11-01): a thin-signal document with solid custody must never
        # be described as "weak custody" merely because it fell through to
        # this branch.
        if custody >= 0.70:
            verdict, recommended_action = "thin signal despite solid custody — useful as a lead, not evidence", "treat_as_lead"
        else:
            verdict, recommended_action = "weak custody or noisy frame — useful as a lead, not evidence", "treat_as_lead"

    # Cap for secondary market articles — cannot earn verify_primary
    if (
        source_genre in _FINANCIAL_SECONDARY_GENRES
        and effective_st in ("secondary", "secondary_market_article", "news_article")
        and recommended_action == "verify_primary"
    ):
        recommended_action = "treat_as_lead"
        verdict = "secondary market source — treat as lead, verify at primary data source"
    if x_post_assessment:
        receipt_score = x_post_assessment.get("receipt_score", 0.0)
        weak_chain_hits = x_post_assessment.get("weak_chain_markers", 0)
        bait_hits = x_post_assessment.get("engagement_bait_markers", 0)
        if recommended_action in ("verify_primary", "use_as_anchor"):
            # Social custody is not anchor-grade without off-platform
            # confirmation — same reason verify_primary is capped here.
            recommended_action = "treat_as_lead"
            verdict = "X news post with some receipts - treat as lead, verify off-platform before repeating"
        if receipt_score < 0.35 and (weak_chain_hits or bait_hits or certainty_bias >= 0.45):
            recommended_action = "seek_receipts"
            verdict = "X news post claims outpace receipts - verify before passing forward"

    # The language layer is a conservative guardrail. It may lower trust when
    # it sees lexical or paragraph-level risk, but it never upgrades the richer
    # custody-aware verdict produced above.
    language_guardrail_action = ""
    if language_assessment:
        # Guardrail-internal comparison ONLY (Arch design note 2026-11-01):
        # the six actions are different next moves, not points on one
        # confidence scale, and no numerical ordering is imposed on the
        # enum itself. The language layer's vocabulary tops out at
        # verify_primary, so that value means "no language objection" —
        # it must NOT displace an anchor-grade custody verdict. Only a
        # strictly more cautious language action (treat_as_lead or worse)
        # downgrades an anchor. The guardrail never upgrades anything TO
        # use_as_anchor: the language layer never returns it.
        action_rank = {
            "use_as_anchor": 0,
            "verify_primary": 0,
            "treat_as_lead": 1,
            "seek_receipts": 2,
            "drop": 3,
            "reject": 4,
        }
        language_action = language_assessment.get("recommended_action", "treat_as_lead")
        if action_rank.get(language_action, 1) > action_rank.get(recommended_action, 1):
            language_guardrail_action = language_action
            recommended_action = language_action
            verdict = (
                "language-aware guardrail lowered trust — inspect the language assessment "
                "and seek receipts before passing this text forward"
            )
            flags.append(
                f"Language-aware guardrail selected '{language_action}' based on "
                "declared-language lexical or paragraph-level risk."
            )

    # Dynamic follow-up questions. Two-stage custody-aware (Arch
    # review 2026-11-05, v1.8.1): a VERIFIED primary gets dates/scope/
    # versions questions; a doc that CLAIMS primary status but whose
    # custody is still uncertain gets a custody question — what evidence
    # would establish its authorship or authority. A document claiming
    # to be the original source is never asked where the original source
    # is. Declaration is never proof of custody: the verified bar is
    # the same st_confidence >= 0.70 the anchor predicate uses, with
    # no genre mismatch.
    primary_verified = (
        effective_st in ("primary", "first_hand", "regulatory_filing")
        and st_confidence >= 0.70
        and not genre_mismatch
    )
    questions = []
    if primary_verified:
        questions.append(
            "What implementation dates, scope limits, or superseding versions apply to this claim?"
        )
    elif declared_st in _HIGH_TRUST_TYPES:
        questions.append(
            "What evidence would establish that this publisher originated the claim — "
            "authorship, ownership, or an official link to the source?"
        )
    else:
        questions.append("What is the closest primary source for the main claim?")
    questions.extend([
        "Who benefits if this framing is believed?",
        "What would change my mind if this claim were wrong?",
        "Which claims are observed facts vs interpretation vs speculation?",
    ])
    if certainty_bias > 0.45:
        questions.append("Why is the author so certain, and what uncertainty is missing?")
    if custody < 0.50:
        questions.append("Can I trace this to a named witness, raw document, dataset, or recording?")
    if pressure > 0.50:
        questions.append("What emotion is this trying to produce before I can inspect it?")
    if unanchored_n > anchored_n and (anchored_n + unanchored_n) > 0:
        questions.append("Which evidence terms here are actually backed by a number, link, or named source?")
    if bias_risk > 0.35:
        questions.append("Who profits if this framing spreads?")
    if attack_hits:
        questions.append("Strip the attack language; does the underlying argument still stand?")
    if attribution < 0.5 and count_matches(QUOTE_PAT, raw) >= 1:
        questions.append("The quotes here — who actually said them, when, and on what record?")
    if x_post_assessment:
        questions.extend([
            "Can I find the same claim in an official statement, named wire report, court record, or raw recording?",
            "Is this the original post/source, or a repost/screenshot/quote-post chain?",
            "What exact time and time zone does the claimed event or confirmation refer to?",
        ])
        if not x_post_assessment.get("has_non_social_link"):
            questions.append("What non-social link would confirm this without relying on X engagement signals?")

    scores = Scores(
        signal=round(signal, 3),
        noise=round(noise, 3),
        pressure=round(pressure, 3),
        source_custody=round(custody, 3),
        source_custody_raw=round(custody_raw, 3),
        source_custody_adjusted=round(custody, 3),
        certainty_bias=round(certainty_bias, 3),
        specificity=round(specificity, 3),
        evidence=round(evidence, 3),
        attribution=round(attribution, 3),
        bias_risk=round(bias_risk, 3),
        manipulation_risk=round(manipulation_risk, 3),
        certainty_to_evidence_gap=round(certainty_to_evidence_gap, 3),
        internal_coherence=internal_coherence,
        overall_confidence=round(overall, 3),
    )
    warning_level = custody_warning_level(custody, st_confidence, declared_st)
    triage_summary = build_triage_summary(
        custody_warning=warning_level,
        certainty_bias=certainty_bias,
        pressure=pressure,
        attack_hits=attack_hits,
        anchored_n=anchored_n,
        unanchored_n=unanchored_n,
        action=recommended_action,
    )
    if language_guardrail_action:
        triage_summary = (
            f"Language-aware guardrail selected "
            f"{language_guardrail_action.replace('_', ' ')}; inspect lexical and "
            "paragraph-level risk before passing this text forward."
        )
    # Hosted-artifact first-party evidence (v1.8.1, gap 1; split into
    # candidate + verified ownership in v1.8.2), computed once above
    # and reused here: the custody breakdown needs it for
    # primary_source_present, and the signal brief needs to know
    # whether the doc was downgraded (see the no-cue sentinel).
    # Only the VERIFIED pair fires primary_source_present; a candidate
    # whose ownership is not established never certifies itself.
    hosted_first_party = hosted_verified
    if hosted_verified:
        hosted_reason = (
            f"hosted release artifact: {hosted_candidate_reason}; "
            "ownership independently established: "
            f"{hosted_verified_reason}"
        )
    elif hosted_candidate:
        hosted_reason = (
            f"hosted artifact candidate ({hosted_candidate_reason}) but "
            "ownership not independently established: "
            f"{hosted_verified_reason}"
        )
    else:
        hosted_reason = hosted_candidate_reason
    hosted_ownership_status = (
        "verified" if hosted_verified
        else "candidate_uncertain" if hosted_candidate
        else "none"
    )
    # A consequential custody change: a declared high-trust type that
    # was not kept at full confidence, or a genre mismatch. The signal
    # brief must not accept an empty key_signals list in that case
    # (Arch review 2026-11-05): if no supporting cue exists, it must
    # say so rather than manufacture certainty.
    custody_downgraded = bool(genre_mismatch) or (
        declared_st in _HIGH_TRUST_TYPES and st_confidence < 0.50
    )
    # Phase 1.2 enrichment (requires signal_extract.py)
    custody_breakdown: Dict = {}
    evidence_shape_data: Dict = {}
    signal_brief: Dict = {}
    if _EXTRACT_AVAILABLE:
        custody_breakdown = build_custody_breakdown(
            raw, source_genre, effective_st, source_url, st_confidence,
            hosted_first_party=hosted_first_party,
            hosted_reason=hosted_reason,
            hosted_candidate=hosted_candidate,
            hosted_ownership_status=hosted_ownership_status,
            hosted_ownership_reason=hosted_reason,
        )
        evidence_shape_data = build_evidence_shape(
            raw, sentences, anchored_n, unanchored_n
        )
        signal_brief = extract_signal_brief(
            raw, source_genre, effective_st, source_url,
            custody_downgraded=custody_downgraded,
        )

    assessment = Assessment(
        source_name=source_name,
        source_type=effective_st,
        declared_source_type=declared_st,
        inferred_source_type=inferred_st,
        source_type_confidence=round(st_confidence, 3),
        source_url=source_url,
        word_count=word_count,
        language_note=language_note,
        language=language_routing,
        language_assessment=language_assessment,
        verdict=verdict,
        recommended_action=recommended_action,
        triage_summary=triage_summary,
        scores=scores,
        confidence_band={"low": round(confidence_low, 3), "high": round(confidence_high, 3)},
        custody_label=custody_label,
        custody_warning_level=warning_level,
        evidence_breakdown={"anchored": anchored_n, "unanchored": unanchored_n},
        flags=flags,
        evidence_clues=evidence_snips + custody_clues,
        pressure_clues=snippets_for(HYPE, raw, 3) + snippets_for(URGENCY, raw, 3) + snippets_for(CERTAINTY, raw, 3),
        questions_to_ask=questions,
        ai_instruction=(
            "Use this as heuristic triage only. The verdict is not truth. "
            "For any consequential claim: seek primary sources, separate observation "
            "from interpretation, preserve uncertainty labels, and prefer the "
            "confidence_band over the single overall_confidence value."
            + (
                " Language layer: "
                + language_assessment.get("ai_instruction", "")
                if language_assessment else ""
            )
        ),
        source_genre=source_genre,
        custody_breakdown=custody_breakdown,
        evidence_shape=evidence_shape_data,
        signal_brief=signal_brief,
        x_post_assessment=x_post_assessment,
    )
    return asdict(assessment)

# ---------------------------------------------------------------------------
# Pretty report (CLI / human)
# ---------------------------------------------------------------------------

def pretty_report(result: Dict) -> str:
    s = result["scores"]
    band = result["confidence_band"]
    eb = result["evidence_breakdown"]
    sep = "=" * 66
    lines = [sep, "SIGNAL SIEVE -- heuristic receipt check", sep]
    lines.append(f"Triage         : {result['triage_summary']}")
    lines.append("")
    lines.append(f"Verdict        : {result['verdict']}")
    lines.append(f"Action         : {result['recommended_action']}")
    lines.append(f"Custody        : {result['custody_label']} [{result['custody_warning_level']}]")
    lines.append(
        f"Source type    : effective={result['source_type']} "
        f"declared={result['declared_source_type']} "
        f"inferred={result['inferred_source_type']} "
        f"confidence={result['source_type_confidence']:.2f}"
    )
    lines.append(f"Words          : {result['word_count']}")
    lines.append(f"Confidence band: low={band['low']:.2f}  high={band['high']:.2f}")
    lines.append(f"Evidence       : anchored={eb['anchored']}  unanchored={eb['unanchored']}")
    if result.get("language_note"):
        lines.append(f"Language note  : {result['language_note']}")
    lines.append("")
    lines.append("Scores 0-1 (gap is -1..1):")
    for k in [
        "signal", "noise", "pressure", "source_custody",
        "certainty_bias", "evidence", "attribution",
        "internal_coherence", "manipulation_risk",
        "certainty_to_evidence_gap", "overall_confidence",
    ]:
        lines.append(f"  {k:28s} {s[k]:.3f}")
    if result["flags"]:
        lines.append("\nFlags:")
        for f in result["flags"]:
            lines.append(f"  - {f}")
    if result["evidence_clues"]:
        lines.append("\nEvidence / source clues:")
        for e in result["evidence_clues"][:6]:
            lines.append(f"  - {e}")
    if result["pressure_clues"]:
        lines.append("\nPressure / certainty clues:")
        for p in result["pressure_clues"][:6]:
            lines.append(f"  - {p}")
    lines.append("\nQuestions to ask:")
    for q in result["questions_to_ask"]:
        lines.append(f"  - {q}")

    # Evidence shape (Phase 1.2)
    es = result.get("evidence_shape", {})
    if es:
        lines.append("\nEvidence shape:")
        named = es.get("named_sources", [])
        lines.append(f"  {'Local numbers':22s}: {es.get('local_numeric_anchors', '?')}")
        lines.append(f"  {'Named data source':22s}: {'yes — ' + ', '.join(named) if named else 'none'}")
        lines.append(f"  {'Primary data linked':22s}: {'yes' if es.get('primary_links_present') else 'no'}")
        lines.append(f"  {'External receipts':22s}: {es.get('external_receipts', '?')}")
        lines.append(f"  {'Caveats present':22s}: {'yes' if es.get('source_caveats_present') else 'no'}")

    # Signal brief (Phase 1.2)
    sb = result.get("signal_brief", {})
    if sb:
        lines.append(f"\nSIGNAL BRIEF")
        lines.append("-" * 50)
        genre_label = (result.get("source_genre") or sb.get("genre", "?")).replace("_", " ").title()
        st_label    = (sb.get("source_type") or result.get("source_type") or "?").replace("_", " ").title()
        lines.append(f"Genre        : {genre_label}")
        lines.append(f"Source type  : {st_label}")
        pdn = sb.get("primary_data_named", [])
        if pdn:
            verified = "verified" if sb.get("primary_data_verified") else "named, not verified by this run"
            lines.append(f"Primary data : {', '.join(pdn)} — {verified}")
        for section_key, label in (
            ("key_signals",              "KEY SIGNALS"),
            ("source_caveats",           "SOURCE CAVEATS"),
            ("interpretation_or_framing","INTERPRETATION / FRAMING"),
            ("missing_receipts",         "MISSING RECEIPTS"),
            ("do_not_pass_forward_as",   "DO NOT PASS FORWARD AS"),
            ("follow_up_sources",        "FOLLOW-UP SOURCES TO CHECK"),
        ):
            items = sb.get(section_key, [])
            if items:
                lines.append(f"\n{label}")
                for item in items:
                    lines.append(f"  • {item}")

    lines.append("\n" + result["ai_instruction"])
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Demo cases
# ---------------------------------------------------------------------------

DEMO_CASES = [
    (
        "clear hype / social",
        "BREAKTHROUGH miracle discovery proves doctors were wrong all along. "
        "Experts hate this one weird trick. They don't want you to know about this hidden truth. "
        "Act now before it is banned. Share this before it gets deleted! "
        "No dataset, no method, no named source -- but mainstream media won't tell you. "
        "Wake up. The science is settled. Final warning.",
        "social",
    ),
    (
        "evidence-word salad (adversarial)",
        "According to peer-reviewed research with n=10000 participants, "
        "sample size and methodology in the appendix, the dataset clearly "
        "proves that this miracle compound definitely cures everything. "
        "Trust me. The science is settled.",
        "primary",
    ),
    (
        "hedged analytical / opinion",
        "There is a reasonable case that AI development may proceed faster "
        "than current regulation can adapt. Some labs publish capabilities "
        "reports; others remain opaque. According to the Stanford AI Index 2024, "
        "training compute roughly doubles every six months, though the "
        "methodology notes selection bias. A counterpoint: economic gravity "
        "tends to slow disruptive technologies.",
        "opinion",
    ),
    (
        "strong primary source",
        "The Q3 2024 SEC 10-Q filing for Acme Corp (filed November 4, 2024) "
        "reports revenue of $1.2 billion, up 8% year-over-year. The methodology "
        "section in the supplementary appendix details the n=4892 customer cohort. "
        "Sample size and confidence intervals are provided. Direct quote from CFO "
        'Jane Doe said: \u201cWe see headwinds in Q4 due to FX exposure.\u201d '
        "See https://www.sec.gov/edgar for the filing.",
        "primary",
    ),
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Reconfigure stdout for UTF-8 on Windows to handle any non-ASCII in source text
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Signal/noise + source-custody heuristic assessor")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--text", help="Text to analyze")
    src.add_argument("--file", help="Path to text file")
    src.add_argument("--stdin", action="store_true", help="Read text from stdin")
    ap.add_argument(
        "--source-type", default="auto",
        help="auto|first_hand|primary|official|expert|secondary|tertiary|anonymous|social|opinion|unknown",
    )
    ap.add_argument("--source-name", default="unknown", help="Name of source / publication / person")
    ap.add_argument("--source-url", default="", help="URL if available")
    ap.add_argument(
        "--language", default="en",
        help="Declared language: en (reference), es (starter), or another code for structural-only mode",
    )
    ap.add_argument("--pretty", action="store_true", help="Print human-readable report")
    ap.add_argument("--json", action="store_true", help="Print JSON (default if --pretty not used)")
    ap.add_argument("--demo", action="store_true", help="Run demo samples spanning the verdict range")
    args = ap.parse_args()

    if args.demo:
        for label, text, st in DEMO_CASES:
            print(f"\n>>> DEMO: {label}  (source_type={st})")
            res = analyze(text, source_type=st, source_name=f"demo: {label}")
            if args.json and not args.pretty:
                print(json.dumps(res, indent=2, ensure_ascii=False))
            else:
                print(pretty_report(res))
        return

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif args.stdin:
        text = sys.stdin.read()
    else:
        text = args.text or ""

    if not text.strip():
        ap.error("Provide --text, --file, --stdin, or --demo")

    result = analyze(
        text,
        source_type=args.source_type,
        source_name=args.source_name,
        source_url=args.source_url,
        language=args.language,
    )
    if args.pretty:
        print(pretty_report(result))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
