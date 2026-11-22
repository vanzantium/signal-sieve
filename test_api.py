#!/usr/bin/env python3
"""
test_api.py — Flask test-client tests for the Signal Sieve API v1.

Tests:
  - /health                        -> enriched JSON
  - /openapi.json                  -> valid spec structure
  - POST /api/v1/analyze           -> full enveloped JSON
  - POST /api/v1/analyze.txt       -> stable plain-text report
  - missing text -> 400 bad_request
  - text too long -> 413 text_too_long
  - optional API key -> 401 if missing when key is set
  - response includes request_id and analysis_version
  - recommended_action is a valid enum value
  - source_type is a valid enum value
  - confidence_band is [low, high] array

Exit code: number of failures (0 = all pass).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import app as app_module
from app import (
    API_VERSION, ANALYSIS_VERSION, MAX_TEXT_CHARS, MAX_TEXT_CHARS_KEYED,
    _ACTION_ENUM, _SOURCE_TYPE_ENUM,
)
from signal_compare import extract_numeric_claims
from signal_compare import extract_generic_quantitative_claims, extract_date_claims
from signal_sieve import _hosted_release_artifact
from signal_sieve import _infer_source_type_from_signals
from signal_sieve import _hosted_artifact_candidate, _hosted_ownership_verified
from signal_sieve import _hosted_namespace_matched
from signal_sieve import _has_first_party_primary_evidence, _first_person_research
from signal_sieve import _host_is_or_subdomain

client = app_module.app.test_client()

GREEN  = "\033[32m" if sys.stdout.isatty() else ""
RED    = "\033[31m" if sys.stdout.isatty() else ""
DIM    = "\033[2m"  if sys.stdout.isatty() else ""
RESET  = "\033[0m"  if sys.stdout.isatty() else ""

_SAMPLE_TEXT = (
    "According to the Bureau of Labor Statistics, inflation fell to 2.4% in "
    "October 2024. The Federal Reserve kept rates at 5.25 to 5.50 percent. "
    "Higher rates contributed to a 23% drop in housing starts over twelve months. "
    "Some 5.2 million households are now in mortgage stress. "
    "The Fed should cut rates immediately. Policy makers appear to be fighting "
    "the last war. Tight money policy likely transfers wealth from workers to "
    "financial institutions."
)

TESTS: list[tuple[str, callable]] = []


def test(name: str):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


# ── /health ────────────────────────────────────────────────────────────────────

@test("/health returns status ok")
def _():
    r = client.get("/health")
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "signal-sieve"
    assert data["api_version"] == API_VERSION
    assert data["analysis_version"] == ANALYSIS_VERSION
    assert isinstance(data["uptime_seconds"], int)


# ── /openapi.json ──────────────────────────────────────────────────────────────

@test("/openapi.json returns valid spec structure")
def _():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.get_json()
    assert spec["openapi"].startswith("3.")
    assert "paths" in spec
    assert "/api/v1/analyze" in spec["paths"]
    assert "/api/v1/analyze.txt" in spec["paths"]
    assert "/health" in spec["paths"]
    assert "components" in spec
    assert "AnalyzeRequest" in spec["components"]["schemas"]
    assert "AnalyzeResponse" in spec["components"]["schemas"]
    assert "ErrorResponse" in spec["components"]["schemas"]


# ── POST /api/v1/analyze ───────────────────────────────────────────────────────

@test("POST /api/v1/analyze returns 200 with envelope")
def _():
    r = client.post("/api/v1/analyze",
                    json={"text": _SAMPLE_TEXT, "source_type": "auto"})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert d["kind"] == "heuristic_signal_triage"
    assert d["verdict_type"] == "signal_shape_assessment"
    assert "safe_use" in d
    assert d["api_version"] == API_VERSION
    assert d["analysis_version"] == ANALYSIS_VERSION


@test("POST /api/v1/analyze has request_id in body and header")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    d = r.get_json()
    assert "request_id" in d
    assert d["request_id"].startswith("req_")
    assert r.headers.get("X-Request-Id") == d["request_id"]


@test("POST /api/v1/analyze recommended_action is valid enum")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    d = r.get_json()
    assert d["recommended_action"] in _ACTION_ENUM, \
        f"unexpected action: {d['recommended_action']!r}"


@test("POST /api/v1/analyze source_type is valid enum")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    d = r.get_json()
    assert d["source_type"] in _SOURCE_TYPE_ENUM, \
        f"unexpected source_type: {d['source_type']!r}"


@test("POST /api/v1/analyze confidence_band is [low, high] array")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    d = r.get_json()
    band = d["confidence_band"]
    assert isinstance(band, list) and len(band) == 2
    assert band[0] <= band[1], f"band inverted: {band}"


@test("POST /api/v1/analyze overall_confidence == receipt_readiness_score")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    d = r.get_json()
    assert d["overall_confidence"] == d["receipt_readiness_score"]


@test("POST /api/v1/analyze includes scores, flags, questions")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    d = r.get_json()
    assert isinstance(d.get("scores"), dict)
    assert isinstance(d.get("flags"), list)
    assert isinstance(d.get("questions_to_ask"), list)


@test("POST /api/v1/analyze exposes declared-language assessment")
def _():
    r = client.post(
        "/api/v1/analyze",
        json={
            "text": "Fuentes anónimas dicen que está comprobado, pero no presentan datos.",
            "language": "es",
        },
    )
    assert r.status_code == 200
    d = r.get_json()
    assert d["language"]["lexicon"] == "es"
    assert d["language"]["calibration"] == "starter_partial"
    assert d["language_assessment"]["scores"]["lexical_available"] == 1.0


@test("POST /api/v1/analyze handles invalid language type without 500")
def _():
    r = client.post(
        "/api/v1/analyze",
        json={"text": _SAMPLE_TEXT, "language": 42},
    )
    assert r.status_code == 200
    d = r.get_json()
    assert d["language"]["mode"] == "structural_only"
    assert d["language"]["requested_language"] == "invalid"


@test("POST /api/v1/analyze with include_signal_brief=true has signal_brief")
def _():
    r = client.post("/api/v1/analyze",
                    json={"text": _SAMPLE_TEXT, "include_signal_brief": True})
    d = r.get_json()
    assert "signal_brief" in d
    assert isinstance(d.get("missing_receipts"), list)


@test("POST /api/v1/analyze with include_signal_brief=false omits signal_brief")
def _():
    r = client.post("/api/v1/analyze",
                    json={"text": _SAMPLE_TEXT, "include_signal_brief": False})
    d = r.get_json()
    assert "signal_brief" not in d


@test("POST /api/v1/analyze detects X news posts and exposes receipt assessment")
def _():
    r = client.post(
        "/api/v1/analyze",
        json={
            "text": (
                "BREAKING: Officials confirm the bridge is closed after a fire. "
                "2026-06-19 09:42 PDT statement linked from the city. "
                "Official update: https://city.example.gov/bridge-fire-update"
            ),
            "source_url": "https://x.com/example/status/1234567890",
            "source_type": "auto",
        },
    )
    assert r.status_code == 200
    d = r.get_json()
    assert d["source_genre"] == "x_news_post"
    assert d["source_type"] == "social"
    assert d["recommended_action"] == "treat_as_lead"
    assert "x_post_assessment" in d
    assert d["x_post_assessment"]["has_non_social_link"] is True
    assert d["x_post_assessment"]["receipt_score"] > 0


# ── error responses ─────────────────────────────────────────────────────────────

@test("POST /api/v1/analyze missing text -> 400 bad_request")
def _():
    r = client.post("/api/v1/analyze", json={})
    assert r.status_code == 400
    d = r.get_json()
    assert d["error"] == "bad_request"
    assert "request_id" in d


@test("POST /api/v1/analyze text too long -> 413 text_too_long")
def _():
    long_text = "word " * (MAX_TEXT_CHARS + 10)
    r = client.post("/api/v1/analyze", json={"text": long_text})
    assert r.status_code == 413
    d = r.get_json()
    assert d["error"] == "text_too_long"
    assert d.get("max_chars") == MAX_TEXT_CHARS


@test("POST /api/v1/analyze empty text -> 400")
def _():
    r = client.post("/api/v1/analyze", json={"text": "   "})
    assert r.status_code == 400


@test("POST /api/v1/analyze with API key env set, no header -> 401")
def _():
    original = os.environ.pop("SIGNAL_SIEVE_API_KEY", None)
    os.environ["SIGNAL_SIEVE_API_KEY"] = "test-secret-key"
    try:
        r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
        assert r.status_code == 401
        d = r.get_json()
        assert d["error"] == "unauthorized"
    finally:
        if original is not None:
            os.environ["SIGNAL_SIEVE_API_KEY"] = original
        else:
            os.environ.pop("SIGNAL_SIEVE_API_KEY", None)


@test("POST /api/v1/analyze with correct API key -> 200")
def _():
    original = os.environ.pop("SIGNAL_SIEVE_API_KEY", None)
    os.environ["SIGNAL_SIEVE_API_KEY"] = "test-secret-key"
    try:
        r = client.post("/api/v1/analyze",
                        json={"text": _SAMPLE_TEXT},
                        headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 200
    finally:
        if original is not None:
            os.environ["SIGNAL_SIEVE_API_KEY"] = original
        else:
            os.environ.pop("SIGNAL_SIEVE_API_KEY", None)


# ── POST /api/v1/analyze.txt ───────────────────────────────────────────────────

@test("POST /api/v1/analyze.txt returns 200 text/plain")
def _():
    r = client.post("/api/v1/analyze.txt", json={"text": _SAMPLE_TEXT})
    assert r.status_code == 200
    assert "text/plain" in r.content_type


@test("POST /api/v1/analyze.txt has stable section headers")
def _():
    r = client.post("/api/v1/analyze.txt", json={"text": _SAMPLE_TEXT})
    body = r.get_data(as_text=True)
    assert "SIGNAL SIEVE REPORT" in body
    assert "ACTION:" in body
    assert "VERDICT TYPE:" in body
    assert "SAFE USE:" in body
    assert "REQUEST ID:" in body
    assert "ANALYSIS VERSION:" in body
    assert "SOURCE" in body


@test("POST /api/v1/analyze.txt has X-Request-Id header")
def _():
    r = client.post("/api/v1/analyze.txt", json={"text": _SAMPLE_TEXT})
    assert r.headers.get("X-Request-Id", "").startswith("req_")


@test("POST /api/v1/analyze.txt missing text -> 400")
def _():
    r = client.post("/api/v1/analyze.txt", json={})
    assert r.status_code == 400


# ── source type normalization ──────────────────────────────────────────────────

@test("POST /api/v1/analyze has declared / resolved / inferred source type fields")
def _():
    r = client.post("/api/v1/analyze",
                    json={"text": _SAMPLE_TEXT, "source_type": "secondary"})
    d = r.get_json()
    assert "declared_source_type" in d, "missing declared_source_type"
    assert "resolved_source_type" in d, "missing resolved_source_type"
    assert "inferred_source_type" in d, "missing inferred_source_type"
    assert "source_type_confidence" in d, "missing source_type_confidence"
    # declared should reflect what the caller sent
    assert d["declared_source_type"] == "secondary", \
        f"declared={d['declared_source_type']!r}, expected 'secondary'"
    # source_type backward-compat alias should match resolved
    assert d["source_type"] == d["resolved_source_type"], \
        "source_type must be an alias for resolved_source_type"


@test("POST /api/v1/analyze has score_meaning field")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    d = r.get_json()
    assert "score_meaning" in d
    sm = d["score_meaning"]
    assert len(sm) > 20, "score_meaning is too short"
    # Should not claim truth probability
    assert "truth" in sm.lower() or "true" in sm.lower() or "probability" in sm.lower()


# ── additional edge cases ──────────────────────────────────────────────────────

@test("POST /api/v1/analyze with malformed JSON -> 400")
def _():
    r = client.post("/api/v1/analyze",
                    data="{ not json !!!",
                    content_type="application/json")
    # Flask returns 400 for malformed JSON when silent=True fallback returns {}
    # which then fails the text validation
    assert r.status_code == 400


@test("POST /api/v1/analyze with whitespace-only text -> 400")
def _():
    r = client.post("/api/v1/analyze", json={"text": "   \n\t  "})
    assert r.status_code == 400


@test("POST /api/v1/analyze text too long returns max_chars in body")
def _():
    long_text = "word " * (MAX_TEXT_CHARS + 50)
    r = client.post("/api/v1/analyze", json={"text": long_text})
    assert r.status_code == 413
    d = r.get_json()
    assert d.get("max_chars") == MAX_TEXT_CHARS
    assert d.get("request_id", "").startswith("req_")


@test("POST /api/v1/analyze 413 response includes suggestion field")
def _():
    long_text = "word " * (MAX_TEXT_CHARS + 50)
    r = client.post("/api/v1/analyze", json={"text": long_text})
    assert r.status_code == 413
    d = r.get_json()
    assert "suggestion" in d, "413 response should include suggestion field"
    assert len(d["suggestion"]) > 20, "suggestion field should be non-trivial"


@test("POST /api/v1/analyze authenticated mode accepts up to MAX_TEXT_CHARS_KEYED")
def _():
    original = os.environ.pop("SIGNAL_SIEVE_API_KEY", None)
    os.environ["SIGNAL_SIEVE_API_KEY"] = "test-secret-key"
    try:
        # Text just over public limit but well under keyed limit
        # "word " is 5 chars, so (MAX_TEXT_CHARS // 5 + 500) reps = ~14,500 chars
        reps = MAX_TEXT_CHARS // 5 + 500
        mid_text = "word " * reps
        assert len(mid_text) > MAX_TEXT_CHARS, \
            f"mid_text ({len(mid_text)}) should exceed public limit ({MAX_TEXT_CHARS})"
        assert len(mid_text) < MAX_TEXT_CHARS_KEYED, \
            f"mid_text ({len(mid_text)}) should be under keyed limit ({MAX_TEXT_CHARS_KEYED})"
        r = client.post("/api/v1/analyze",
                        json={"text": mid_text},
                        headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 200, \
            f"authenticated request over public limit should be 200, got {r.status_code}"
    finally:
        if original is not None:
            os.environ["SIGNAL_SIEVE_API_KEY"] = original
        else:
            os.environ.pop("SIGNAL_SIEVE_API_KEY", None)


@test("Error response always includes request_id")
def _():
    for payload in [{}, {"text": "   "}, {"text": "w" * (MAX_TEXT_CHARS + 1)}]:
        r = client.post("/api/v1/analyze", json=payload)
        d = r.get_json()
        assert "request_id" in d, f"no request_id in error for payload {payload!r}"
        assert d["request_id"].startswith("req_")


@test("POST /api/v1/analyze hype/spam text gets drop or reject action")
def _():
    hype = (
        "BUY NOW!!! This stock is going TO THE MOON guaranteed 1000% returns!!! "
        "Everyone who misses this will regret it forever. Act immediately before "
        "the window closes. You are a loser if you don't buy. This changes EVERYTHING. "
        "Insiders already made millions. Don't be left behind!!!"
    )
    r = client.post("/api/v1/analyze", json={"text": hype})
    assert r.status_code == 200
    d = r.get_json()
    assert d["recommended_action"] in ("drop", "reject", "seek_receipts"), \
        f"hype text got {d['recommended_action']!r}"
    assert d["scores"]["pressure"] > 0.5, \
        f"pressure should be high for hype, got {d['scores']['pressure']}"


@test("POST /api/v1/analyze first_hand source raises custody score")
def _():
    witness = (
        "I was present at the meeting on November 3, 2024 when the CEO announced "
        "a 12 percent workforce reduction effective Q1 2025. I personally received "
        "written confirmation. The severance was four weeks per year of service."
    )
    r_anon  = client.post("/api/v1/analyze",
                          json={"text": witness, "source_type": "anonymous"})
    r_first = client.post("/api/v1/analyze",
                          json={"text": witness, "source_type": "first_hand"})
    anon_c  = r_anon.get_json()["scores"]["source_custody"]
    first_c = r_first.get_json()["scores"]["source_custody"]
    assert first_c > anon_c, \
        f"first_hand custody {first_c:.3f} should exceed anonymous {anon_c:.3f}"


@test("GET /openapi.json documents new source type normalization fields")
def _():
    r = client.get("/openapi.json")
    spec = r.get_json()
    props = spec["components"]["schemas"]["AnalyzeResponse"]["properties"]
    assert "declared_source_type" in props
    assert "resolved_source_type" in props
    assert "inferred_source_type" in props
    assert "score_meaning" in props


# ── Phase 2: capsule ──────────────────────────────────────────────────────────

@test("POST /api/v1/analyze includes capsule in response")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    assert r.status_code == 200
    d = r.get_json()
    assert "capsule" in d, "response should include 'capsule' field"
    cap = d["capsule"]
    assert isinstance(cap, str) and len(cap) > 30
    assert "SIGNAL_SIEVE" in cap
    assert "ACTION:" in cap
    assert ANALYSIS_VERSION in cap


@test("POST /api/v1/analyze capsule contains key sections")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SAMPLE_TEXT})
    cap = r.get_json()["capsule"]
    for section in ("CUSTODY:", "SCORES:", "VERDICT:"):
        assert section in cap, f"capsule missing section: {section!r}"


@test("POST /api/v1/analyze.capsule returns plain text capsule")
def _():
    r = client.post("/api/v1/analyze.capsule", json={"text": _SAMPLE_TEXT})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert "text/plain" in r.content_type
    body = r.get_data(as_text=True)
    assert "SIGNAL_SIEVE" in body
    assert "ACTION:" in body
    assert r.headers.get("X-Request-Id", "").startswith("req_")


@test("POST /api/v1/analyze.capsule missing text -> 400")
def _():
    r = client.post("/api/v1/analyze.capsule", json={})
    assert r.status_code == 400


# ── Phase 2: compare ──────────────────────────────────────────────────────────

_SCHWAB_SNIPPET = (
    "Schwab Market Update — November 14, 2024\n"
    "10-Year Treasury Yield: 4.61%  |  WTI Crude: $68.43/bbl\n"
    "S&P 500: +0.02%  |  Nasdaq Composite: +0.06%\n"
)

_BLACKROCK_SNIPPET = (
    "BlackRock Investment Outlook, November 2024\n"
    "10-year Treasury yield: 4.56%.\n"
    "Brent crude: $72.10 per barrel.\n"
    "S&P 500 reached a record high of 5,894 on November 11, 2024.\n"
)


@test("POST /api/v1/compare returns 200 with comparison envelope")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": _SCHWAB_SNIPPET,    "source_name": "Schwab"},
            {"text": _BLACKROCK_SNIPPET, "source_name": "BlackRock"},
        ]
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert d["kind"]          == "cross_document_comparison"
    assert d["verdict_type"]  == "numeric_claim_cross_reference"
    assert d["doc_count"]     == 2
    assert "shared_claims"    in d
    assert "conflicts"        in d
    assert "alignment"        in d
    assert "triage_summary"   in d
    assert r.headers.get("X-Request-Id", "").startswith("req_")


@test("POST /api/v1/compare requires 2+ documents")
def _():
    # Zero documents
    r = client.post("/api/v1/compare", json={"documents": []})
    assert r.status_code == 400
    # One document
    r = client.post("/api/v1/compare", json={
        "documents": [{"text": _SCHWAB_SNIPPET}]
    })
    assert r.status_code == 400
    d = r.get_json()
    assert d["error"] == "bad_request"


@test("POST /api/v1/compare finds 10yr Treasury yield discrepancy")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": _SCHWAB_SNIPPET,    "source_name": "Schwab"},
            {"text": _BLACKROCK_SNIPPET, "source_name": "BlackRock"},
        ]
    })
    d = r.get_json()
    # 4.61 vs 4.56 = 0.05 spread — within 0.15 tolerance, so this aligns
    # But we need to verify the entity was found in shared_claims at minimum
    all_keys = (
        [c["entity_key"] for c in d.get("shared_claims", [])] +
        [c["entity_key"] for c in d.get("conflicts",     [])] +
        [c["entity_key"] for c in d.get("alignment",     [])]
    )
    assert "10yr-treasury-yield" in all_keys, \
        f"expected 10yr-treasury-yield in shared/conflict/alignment, got keys: {all_keys}"


@test("POST /api/v1/compare detects WTI vs Brent as different benchmarks")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": _SCHWAB_SNIPPET,    "source_name": "Schwab"},
            {"text": _BLACKROCK_SNIPPET, "source_name": "BlackRock"},
        ]
    })
    d = r.get_json()
    assert d.get("crude_note"), "expected crude_note when WTI and Brent both appear"
    assert "WTI" in d["crude_note"] or "Brent" in d["crude_note"]


@test("POST /api/v1/compare includes compare capsule")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": _SCHWAB_SNIPPET,    "source_name": "Schwab"},
            {"text": _BLACKROCK_SNIPPET, "source_name": "BlackRock"},
        ]
    })
    d = r.get_json()
    assert "capsule" in d
    cap = d["capsule"]
    assert "SIGNAL_SIEVE" in cap
    assert "COMPARE" in cap
    assert "DOCS:" in cap


@test("POST /api/v1/compare missing text in document -> 400")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": _SCHWAB_SNIPPET, "source_name": "Schwab"},
            {"source_name": "BadDoc"},           # missing text
        ]
    })
    assert r.status_code == 400
    d = r.get_json()
    assert d["error"] == "bad_request"


# ── v1.7.0: dates, benchmarks, pricing, unit guard ───────────────────────────

@test("POST /api/v1/compare shares rollout years and exact dates")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "We aim to support the shared token in 2027. Early 2027 rollout planned.",
             "source_name": "Google"},
            {"text": "We will honour the new preferences from February 1, 2027, and publishers "
                     "should act by March 1, 2027.",
             "source_name": "Bing"},
            {"text": "From 2027, nobody reads the new token yet.",
             "source_name": "Stacked"},
        ]
    })
    d = r.get_json()
    shared_years = [x["date_key"] for x in d.get("shared_years", [])]
    assert "2027" in shared_years, f"expected shared year 2027, got {shared_years}"
    unique_dates = [x["date_key"] for x in d.get("unique_dates", [])]
    assert "2027-02-01" in unique_dates, f"expected Bing 2027-02-01 unique, got {unique_dates}"
    assert "2027-03-01" in unique_dates, f"expected Bing 2027-03-01 unique, got {unique_dates}"
    assert "shared date" in d["triage_summary"], "triage should mention shared dates"
    assert "SHARED DATES:" in d["capsule"]


@test("POST /api/v1/compare aligns benchmark scores across surface forms")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "GPQA Diamond: 61.3%.", "source_name": "A"},
            {"text": "The model scored 61.3% on GPQA.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = [c["entity_key"] for c in d.get("alignment", [])]
    assert "gpqa" in aligned, f"expected gpqa aligned, got {aligned}"
    assert not d.get("conflicts"), "equal benchmark scores must not conflict"


@test("POST /api/v1/compare surfaces pricing rows as unique claims")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "API pricing: input $15 per million tokens, output $75 per million tokens.",
             "source_name": "A"},
            {"text": "The announcement describes the new model with no pricing details.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    unique_keys = [c["entity_key"] for c in d.get("unique_claims", [])]
    assert "pricing-input" in unique_keys, f"expected pricing-input unique, got {unique_keys}"
    assert "pricing-output" in unique_keys, f"expected pricing-output unique, got {unique_keys}"
    pricing_conflicts = [c for c in d.get("conflicts", [])
                         if "pricing" in c["entity_key"]]
    assert not pricing_conflicts, "a missing pricing row is unique, not a conflict"


@test("POST /api/v1/compare does not conflate counts with percentages")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "GPQA: 94 problems in the set.", "source_name": "A"},
            {"text": "GPQA: 61.3% accuracy.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    mismatches = [m for m in d.get("unit_mismatches", []) if m["entity_key"] == "gpqa"]
    assert mismatches, "expected a gpqa unit mismatch (count vs %)"
    assert sorted(mismatches[0]["units"]) == ["%", "count"]
    assert not [c for c in d.get("conflicts", []) if c["entity_key"] == "gpqa"], \
        "count vs percentage must not read as a conflict"
    assert "UNIT MISMATCHES" in d["capsule"]


# ── v1.8.0: GDPx preservation, version gate, timeline, agreement ─────────

@test("POST /api/v1/compare keeps GDPx distinct and does not manufacture a conflict from inline counts")
def _():
    # Reconstruction of the Nov 3 GDPx run: the primary says 12% on GDPx;
    # coverage says 13 models on 206 tasks. The v1.7.0 comparator collapsed
    # GDPx to GDP and compared the 13-count against 12%, reporting a
    # spread-204.0 "conflict". v1.8.0 must key GDPx separately, read the
    # inline count as unit "count", and report a unit mismatch, not a
    # conflict — and must not claim agreement when nothing compared.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "SemiAnalysis GDPx v1.0: 12%. The best agent scores just 12% on 206 long professional tasks across 11 domains.",
             "source_name": "SemiAnalysis"},
            {"text": "Coverage of GDPx: 13 models evaluated on 206 tasks.",
             "source_name": "Coverage"},
        ]
    })
    d = r.get_json()
    all_keys = (
        [c["entity_key"] for c in d.get("shared_claims", [])] +
        [c["entity_key"] for c in d.get("conflicts", [])] +
        [c["entity_key"] for c in d.get("alignment", [])] +
        [m["entity_key"] for m in d.get("unit_mismatches", [])]
    )
    assert "gdpx" in all_keys, f"expected gdpx keyed, got {all_keys}"
    assert "gdp" not in all_keys, f"GDPx must not collapse to GDP, got {all_keys}"
    assert not d.get("conflicts"), "inline counts vs percentages must not conflict"
    mismatches = [m for m in d.get("unit_mismatches", []) if m["entity_key"] == "gdpx"]
    assert mismatches, "expected a gdpx unit mismatch (count vs %)"
    assert d.get("agreement_status") == "not_established", \
        f"nothing compared, so agreement must not be established, got {d.get('agreement_status')}"
    assert "no agreement is established" in d["triage_summary"]
    assert "AGREEMENT: not established" in d["capsule"]


@test("POST /api/v1/compare does not compare the same benchmark across versions")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "GPQA v1.0: 61.3%.", "source_name": "A"},
            {"text": "GPQA v2.0: 58.1%.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    vm = [m for m in d.get("version_mismatches", []) if m["entity_key"] == "gpqa"]
    assert vm, "expected a gpqa version mismatch (v1.0 vs v2.0)"
    assert sorted(vm[0]["versions"]) == ["v1.0", "v2.0"]
    assert not [c for c in d.get("conflicts", []) if c["entity_key"] == "gpqa"], \
        "different versions must not read as a conflict"
    assert not [c for c in d.get("alignment", []) if c["entity_key"] == "gpqa"], \
        "different versions must not read as an alignment"
    assert "VERSION MISMATCHES" in d["capsule"]


@test("POST /api/v1/compare reconciles timelines instead of conflicting on them")
def _():
    # Nov 4 rollout shape: same shared year 2027, different precisions —
    # a year-only plan, an exact date, and a vague "coming months".
    # These are not contradictions and must land in the separate
    # timeline_reconciliation diagnostic, not in conflicts.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "We aim to support the token in 2027.", "source_name": "Google"},
            {"text": "We will honour the new preferences from February 1, 2027.", "source_name": "Bing"},
            {"text": "Our rollout is planned in 2027, in the coming months.", "source_name": "OpenAI"},
        ]
    })
    d = r.get_json()
    notes = [n for n in d.get("timeline_reconciliation", []) if n["year"] == "2027"]
    assert notes, "expected a 2027 timeline reconciliation note"
    assert "not a conflict" in notes[0]["note"]
    assert "timeline reconciliation" in d["triage_summary"]
    assert "TIMELINE RECONCILIATION" in d["capsule"]


@test("POST /api/v1/compare keeps presence/absence distinct from conflict")
def _():
    # Negative regression test for the presence/absence channel (Arch:
    # preserved unchanged): a doc that EXPLICITLY lacks an entity is a
    # mismatch, never a numeric conflict, and a merely-missing entity
    # stays NOT_OBSERVED.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "GPQA: 61.3% accuracy.", "source_name": "A"},
            {"text": "The table has no GPQA row.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    mm = [m for m in d.get("presence_absence_mismatches", []) if m["entity_key"] == "gpqa"]
    assert mm, "expected a gpqa presence/absence mismatch"
    assert d["presence_status"]["gpqa"]["B"] == "EXPLICITLY_ABSENT"
    assert not [c for c in d.get("conflicts", []) if c["entity_key"] == "gpqa"], \
        "explicit absence must not read as a numeric conflict"


@test("Actions stay inside the six-verb enum and presence statuses stay in vocabulary")
def _():
    # Negative regression test (Arch: enum and channel preserved
    # unchanged): no action outside the six verbs, no presence status
    # outside the four statuses, whatever v1.8.0 changed elsewhere.
    for text in [
        "BREAKTHROUGH miracle discovery proves doctors were wrong all along. Act now!",
        "The Q3 2024 SEC 10-Q filing reports revenue of $1.2 billion, up 8% year-over-year.",
        "We are announcing a new policy today. According to anonymous sources, it changes everything.",
    ]:
        r = client.post("/api/v1/analyze", json={"text": text, "source_type": "auto"})
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.get_json()["recommended_action"] in _ACTION_ENUM, \
            f"action outside the six-verb enum: {r.get_json()['recommended_action']}"
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "GPQA: 61.3%.", "source_name": "A"},
            {"text": "The table has no GPQA row.", "source_name": "B"},
            {"text": "Unrelated market note with no benchmark.", "source_name": "C"},
        ]
    })
    d = r.get_json()
    allowed = {"PRESENT", "EXPLICITLY_ABSENT", "NOT_OBSERVED", "UNKNOWN"}
    for key, status in d.get("presence_status", {}).items():
        for doc_name, value in status.items():
            assert value in allowed, f"presence status outside vocabulary: {key}/{doc_name}={value}"


@test("GET /openapi.json documents compare and capsule endpoints")
def _():
    r = client.get("/openapi.json")
    spec = r.get_json()
    paths = spec["paths"]
    assert "/api/v1/compare"          in paths, "missing /api/v1/compare in OpenAPI spec"
    assert "/api/v1/analyze.capsule"  in paths, "missing /api/v1/analyze.capsule in OpenAPI spec"
    schemas = spec["components"]["schemas"]
    assert "CompareRequest"  in schemas
    assert "CompareResponse" in schemas
    # AnalyzeResponse should document the capsule field
    ar_props = schemas["AnalyzeResponse"]["properties"]
    assert "capsule" in ar_props, "AnalyzeResponse missing capsule field in OpenAPI spec"
    assert "language" in ar_props, "AnalyzeResponse missing language field"
    assert "language_assessment" in ar_props, "AnalyzeResponse missing language assessment"


# ── v1.8.1: shot counts, whole-word entities, hosted artifacts, custody questions, follow-up gate ─

@test("POST /api/v1/compare does not mistake shot counts for scores")
def _():
    # Reconstruction of the Nov 5 MMLU case: the model card says
    # "MMLU (EM) 5-shot: ERNIE 93.3" and coverage says "MMLU five-shot
    # EM 93.3". v1.8.0 read the 5 as a competing score and reported a
    # 90.3-spread conflict. v1.8.1 must read 93.3 as the score with
    # shot_count=5 on both sides and align them genuinely.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "MMLU (EM) 5-shot: ERNIE 93.3, DeepSeek-V4-Flash 89.4.",
             "source_name": "Card"},
            {"text": "Selected results: MMLU five-shot EM 93.3 (ERNIE) vs 89.4 (DeepSeek).",
             "source_name": "Coverage"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"shot counts must not conflict, got {d.get('conflicts')}"
    aligned = [c for c in d.get("alignment", []) if c["entity_key"] == "mmlu"]
    assert aligned, "expected a genuine mmlu alignment on 93.3"
    assert aligned[0]["shot_count"] == 5, f"expected shot_count 5, got {aligned[0].get('shot_count')}"
    assert "5-shot" in aligned[0]["evaluation_conditions"]
    assert "ALIGNED: 1" in d["capsule"]


@test("POST /api/v1/compare blocks comparison across different shot counts")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "MMLU (EM) 5-shot: ERNIE 93.3.", "source_name": "A"},
            {"text": "MMLU (EM) 0-shot: ERNIE 88.1.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    mm = [m for m in d.get("evaluation_setting_mismatches", []) if m["entity_key"] == "mmlu"]
    assert mm, "expected an mmlu evaluation-setting mismatch (5-shot vs 0-shot)"
    assert mm[0]["shot_counts"] == [0, 5], f"expected shot_counts [0, 5], got {mm[0].get('shot_counts')}"
    assert not [c for c in d.get("conflicts", []) if c["entity_key"] == "mmlu"], \
        "different shot counts must not read as a conflict"
    assert not [c for c in d.get("alignment", []) if c["entity_key"] == "mmlu"], \
        "different shot counts must not read as an alignment"
    assert d.get("agreement_status") == "not_established"
    assert "EVALUATION SETTING MISMATCHES" in d["capsule"]


@test("POST /api/v1/compare compares provisionally when shot counts are unstated")
def _():
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "MMLU: 93.3.", "source_name": "A"},
            {"text": "MMLU 93.3 in the latest run.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = [c for c in d.get("alignment", []) if c["entity_key"] == "mmlu"]
    assert aligned, "expected an mmlu alignment on 93.3"
    assert aligned[0]["shot_count"] is None
    assert "provisional" in aligned[0]["evaluation_conditions"]
    assert "not stated" in aligned[0]["evaluation_conditions"]
    assert "provisional comparison" in d["triage_summary"]


@test("POST /api/v1/compare does not turn Downloads into the Dow Jones")
def _():
    # Arch's requested negative fixture (v1.8.1, gap 3): the terms
    # Downloads, Dow, and Dow Jones in different contexts. A model
    # release ("Downloads are available", "Mr. Dow reviewed the
    # release: 280 billion parameters") must never acquire a Dow Jones
    # entity, while a genuine market sentence ("Dow Jones rose 120
    # points as stocks rallied") still reads as one.
    model_release = (
        "Downloads are available in FP8 and BF16 formats. "
        "The model has 280 billion parameters and 14 billion active per token."
    )
    prose_dow = (
        "Mr. Dow reviewed the release: 280 billion parameters, 14 billion active."
    )
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": model_release, "source_name": "Release"},
            {"text": prose_dow, "source_name": "Prose"},
        ]
    })
    d = r.get_json()
    all_keys = (
        [c["entity_key"] for c in d.get("shared_claims", [])] +
        [c["entity_key"] for c in d.get("conflicts", [])] +
        [c["entity_key"] for c in d.get("alignment", [])] +
        [c["entity_key"] for c in d.get("unique_claims", [])]
    )
    assert "dow" not in all_keys, f"Downloads/prose Dow must not key as Dow Jones, got {all_keys}"
    assert not d.get("conflicts"), "no market conflict may come out of a model release"
    assert d.get("follow_up_sources") == [], \
        f"no market follow-up for a model-release story, got {d.get('follow_up_sources')}"
    assert d.get("follow_up_trace") == []
    dropped = [x for x in d.get("dropped_market_claims", []) if x["entity_key"] == "dow"]
    assert dropped, "the whole-word Dow in non-market prose must be recorded as dropped"
    assert dropped[0]["raw_value"] == "280"

    # The genuine market sentence still reads as the Dow Jones.
    r2 = client.post("/api/v1/compare", json={
        "documents": [
            {"text": model_release, "source_name": "Release"},
            {"text": "Dow Jones rose 120 points as stocks rallied.", "source_name": "Market"},
        ]
    })
    d2 = r2.get_json()
    unique = [c for c in d2.get("unique_claims", []) if c["entity_key"] == "dow"]
    assert unique, "a genuine market Dow Jones mention must still extract"
    assert unique[0]["value"] == 120.0
    assert any("Exchange closing prices" in s for s in d2.get("follow_up_sources", [])), \
        "a genuine market doc still earns the exchange follow-up"
    trace = [t for t in d2.get("follow_up_trace", []) if "dow" in t["entity_keys"]]
    assert trace, "the follow-up must carry its entity trace"


# ── v1.8.5: Nov 6 comparator failures — variant identity, following-benchmark
# guard, setting numbers, pricing tiers, found_in_docs dedup ─────────────

@test("POST /api/v1/compare does not lump benchmark variants into one conflict (v1.8.5)")
def _():
    # Nov 6 gap 1, core: SWE-bench Pro 86.3% and SWE-bench Verified
    # 82.2% are different benchmarks sharing a family name. They must
    # land in a variant mismatch, never a conflict — while the Pro
    # scores still align across docs.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "SWE-bench Pro v2: 86.3%. SWE-bench Verified: 82.2%.", "source_name": "A"},
            {"text": "The model scored 86.3% on SWE-bench Pro v2 (Hard).", "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"variants must not conflict, got {d.get('conflicts')}"
    vm = [m for m in d.get("variant_mismatches", []) if m["entity_key"] == "swe-bench"]
    assert vm, "expected a swe-bench variant mismatch"
    assert vm[0]["variants"] == ["pro", "verified"], f"got variants {vm[0]['variants']}"
    aligned = [c for c in d.get("alignment", []) if c["entity_key"] == "swe-bench"]
    assert aligned and aligned[0]["variant"] == "pro", "Pro scores must still align within their variant"
    assert [v["value"] for v in aligned[0]["values"]] == [86.3, 86.3]
    unique = [(c["entity_key"], c["value"]) for c in d.get("unique_claims", [])]
    assert ("swe-bench", 82.2) in unique, f"Verified 82.2 must be unique, got {unique}"


@test("POST /api/v1/compare does not credit a score to the preceding benchmark (v1.8.5)")
def _():
    # Nov 6 gap 1, inline forward-grab: in "93.0% on GPQA Diamond,
    # 82.2% on SWE-bench Verified, 44.1% on Humanity's Last Exam
    # (text), 13.1% on Terminal-Bench 4.0", each number belongs to the
    # benchmark AFTER it. The reverse pass mints them correctly; the
    # forward pass must not claim 82.2% for GPQA or 13.1% for HLE.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "SWE-bench Pro v2 (Hard): 86.3%. GPQA Diamond: 93.0%.", "source_name": "A"},
            {"text": ("On the model card, the lab posts 93.0% on GPQA Diamond, "
                      "82.2% on SWE-bench Verified, 44.1% on Humanity's Last Exam (text), "
                      "13.1% on Terminal-Bench 4.0."), "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"no false conflicts allowed, got {d.get('conflicts')}"
    aligned = [c["entity_key"] for c in d.get("alignment", [])]
    assert "gpqa" in aligned, "GPQA Diamond 93.0% must align across the two surface forms"
    unique = [(c["entity_key"], c["value"]) for c in d.get("unique_claims", [])]
    assert ("swe-bench", 82.2) in unique, f"Verified 82.2 must be unique, got {unique}"
    assert ("hle", 44.1) in unique, f"HLE (text) 44.1 must be unique, got {unique}"
    # v1.8.8 r2: Terminal-Bench joined the generic benchmark phrases,
    # so 13.1% now surfaces as a generic Terminal-Bench score instead
    # of staying invisible — but it is still NOT credited to HLE (the
    # preceding benchmark); the following-benchmark guard is the point
    # of this test, not the invisibility.
    assert ("generic|percentage|score|bench=terminal-bench||unstated", 13.1) in unique, \
        f"Terminal-Bench 13.1% must surface as a generic score, got {unique}"
    assert not any(k == "hle" and v == 13.1 for k, v in unique), \
        "13.1% must not be credited to HLE"


@test("POST /api/v1/compare does not mint a measurement setting as a score (v1.8.5)")
def _():
    # Nov 6 gap 1, setting number: "MMMU Pro (Standard 10)" — the 10
    # is the setting; the score (81.6%) comes later in the same line.
    # The rescan must mint the 81.6% and never the 10.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "MMMU Pro (Standard 10) reaches 81.6 percent.", "source_name": "A"},
            {"text": "MMMU Pro: 81.6%.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"setting number must not conflict, got {d.get('conflicts')}"
    aligned = [c for c in d.get("alignment", []) if c["entity_key"] == "mmmu"]
    assert aligned, "MMMU Pro 81.6% must align"
    assert [v["value"] for v in aligned[0]["values"]] == [81.6, 81.6]
    all_vals = [v["value"] for c in d.get("shared_claims", []) + d.get("unique_claims", [])
                for v in c.get("values", [c]) if isinstance(v, dict)]
    assert 10.0 not in all_vals, f"the setting 10 must never be a claim, got {all_vals}"
    claims = extract_numeric_claims("MMMU Pro (Standard 10) reaches 81.6 percent.")
    assert len(claims) == 1 and claims[0]["numeric_value"] == 81.6, \
        f"expected exactly one MMMU claim at 81.6, got {claims}"
    assert claims[0]["variant"] == "pro"


@test("POST /api/v1/compare keeps pricing tiers separate and aligns each (v1.8.5)")
def _():
    # Nov 6 gap 1, pricing tiers: input, output, cached and per-task
    # are different claims. The same row in two docs must align tier
    # by tier, never lump into one pricing conflict.
    row = ("Pricing: $0.30 per million input tokens, $1.00 per million output tokens, "
           "$0.03 cached input, $1.18 per task.")
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": row, "source_name": "A"},
            {"text": row, "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"pricing tiers must not conflict, got {d.get('conflicts')}"
    aligned = {c["entity_key"]: [v["value"] for v in c["values"]] for c in d.get("alignment", [])}
    assert aligned.get("pricing-input") == [0.3, 0.3], f"input tier must align, got {aligned}"
    assert aligned.get("pricing-output") == [1.0, 1.0], f"output tier must align, got {aligned}"
    assert aligned.get("pricing-cached") == [0.03, 0.03], f"cached tier must align, got {aligned}"
    assert aligned.get("pricing-per-task") == [1.18, 1.18], f"per-task tier must align, got {aligned}"


@test("POST /api/v1/compare deduplicates found_in_docs in shared claims (v1.8.5)")
def _():
    # Nov 6 gap 5: a doc contributing two values to one shared group
    # was listed twice in found_in_docs. The values stay; the doc
    # list must not repeat.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "GPQA Diamond: 61.3%. GPQA Diamond: 61.35%.", "source_name": "A"},
            {"text": "The model scored 61.3% on GPQA Diamond.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    shared = [c for c in d.get("shared_claims", []) if c["entity_key"] == "gpqa"]
    assert shared, "expected a shared gpqa claim"
    assert shared[0]["found_in_docs"] == ["A", "B"], \
        f"found_in_docs must dedup, got {shared[0]['found_in_docs']}"


@test("POST /api/v1/compare marks a stated-vs-unstated variant comparison provisional (v1.8.5)")
def _():
    # One stated variant against an unstated side still compares
    # (this is the existing surface-forms alignment), but the
    # comparison must carry its provisional variant note — mirroring
    # the shot-count gate.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "GPQA Diamond: 61.3%.", "source_name": "A"},
            {"text": "The model scored 61.3% on GPQA.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = [c for c in d.get("alignment", []) if c["entity_key"] == "gpqa"]
    assert aligned, "GPQA Diamond vs plain GPQA must still align"
    assert aligned[0]["variant"] == "diamond"
    assert "provisional" in aligned[0].get("variant_conditions", ""), \
        f"expected a provisional variant note, got {aligned[0].get('variant_conditions')}"
    assert not d.get("conflicts")


@test("POST /api/v1/compare treats HLE-Diamond and HLE (text) as different variants (v1.8.5)")
def _():
    # Nov 6: the HLE abbreviation makes "HLE-Diamond 28.7%" visible,
    # and it must not compare with HLE (text) 44.1% — different
    # variants, not a conflict.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "HLE-Diamond: 28.7%.", "source_name": "A"},
            {"text": "HLE (text): 44.1%.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"HLE variants must not conflict, got {d.get('conflicts')}"
    vm = [m for m in d.get("variant_mismatches", []) if m["entity_key"] == "hle"]
    assert vm and vm[0]["variants"] == ["diamond", "text"], f"got {vm}"
    unique = [(c["entity_key"], c["value"]) for c in d.get("unique_claims", [])]
    assert ("hle", 28.7) in unique and ("hle", 44.1) in unique, f"got {unique}"


@test("GET /openapi.json documents the variant_mismatches channel (v1.8.5)")
def _():
    r = client.get("/openapi.json")
    spec = r.get_json()
    props = spec["components"]["schemas"]["CompareResponse"]["properties"]
    assert "variant_mismatches" in props, "CompareResponse missing variant_mismatches"


@test("POST /api/v1/analyze keeps a model release out of the financial-market genre (v1.8.5)")
def _():
    # Nov 6 gap 4: the primary announcement and SiliconNoon were both
    # typed financial_market_snapshot. Genre must follow the doc's
    # main claim — a model release whose only financial word is
    # "index" is news, not a market snapshot.
    text = (
        "Today we are introducing Inkling 2 Small, a 594B-parameter sparse "
        "mixture-of-experts model with 17B active parameters. The model scores "
        "31 on the AA Intelligence Index and 86.3% on SWE-bench Pro v2 (Hard). "
        "Pricing is $0.30 per million input tokens and $1.00 per million output "
        "tokens, with cached input at $0.03 per million tokens and $1.18 per task."
    )
    r = client.post("/api/v1/analyze", json={
        "text": text, "source_type": "primary",
        "source_url": "https://thinkingmachines.ai/news/inkling-2-small/",
    })
    d = r.get_json()
    assert d.get("source_genre") != "financial_market_snapshot", \
        f"model release must not type as a market snapshot, got {d.get('source_genre')}"


@test("POST /api/v1/analyze keeps a namespace-matched model card as candidate primary with uncertain ownership (v1.8.3)")
def _():
    # Nov 5 gap 1 via the API, re-decided by Arch's notes 2026-11-06
    # (v1.8.3): Baidu's model card on Hugging Face, declared primary.
    # The huggingface.co/baidu namespace MATCHES the claimed
    # publisher — that fact is recorded — but a namespace match is
    # NOT independently verified organizational ownership, and no
    # trusted verification mechanism exists yet (no verified-org
    # registry; source URLs are caller-supplied). So the card is a
    # candidate primary with uncertain ownership: confidence capped at
    # 0.60, primary_source_present false, the uncertain-ownership
    # flag present, and the custody question asked — never the
    # adaptive verified-primary question. Effective type stays
    # "primary" (declared); it is a lead, not an anchor.
    text = (
        "Baidu ERNIE Team — ERNIE-5.5-Flash-Base model card "
        "(Hugging Face, baidu/ERNIE-5.5-Flash-Base, released 5 November 2026). "
        "ERNIE 5.5 Flash is a sparse Mixture-of-Experts model with 280B total "
        "parameters and 14B activated per token, pre-trained from scratch on "
        "33.5T tokens. By releasing the weights, we let developers build on it. "
        "License: ERNIE-5.5-Flash-Base is provided under the ERNIE 5.5 Flash "
        "Base Model License Agreement v1.0. Copyright 2026 Baidu."
    )
    r = client.post("/api/v1/analyze", json={
        "text": text,
        "source_type": "primary",
        "source_url": "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base",
    })
    d = r.get_json()
    assert d["source_type"] == "primary", f"candidate primary must stay primary, got {d['source_type']}"
    assert d["source_type_confidence"] <= 0.60, \
        f"uncertain ownership must cap confidence at 0.60, got {d['source_type_confidence']}"
    assert d["recommended_action"] != "use_as_anchor", "uncertain ownership must never anchor"
    assert d["custody_breakdown"]["primary_source_present"] is False
    assert d["custody_breakdown"]["hosted_first_party_artifact"] is False
    # v1.8.3: the namespace fact is recorded independently of ownership.
    assert d["custody_breakdown"]["hosted_artifact_candidate"] is True
    assert d["custody_breakdown"]["hosted_namespace_matched"] is True
    assert d["custody_breakdown"]["hosted_ownership_status"] == "candidate_uncertain"
    assert d["custody_breakdown"]["source_url_provenance"] == "caller_supplied"
    assert any("uncertain ownership" in f for f in d["flags"]), f"ownership flag missing, got {d['flags']}"
    assert any("namespace matched" in f for f in d["flags"]), \
        f"flag must name the namespace match, got {d['flags']}"
    # v1.8.6 (P4): a hosted candidate speaks as the originator, so the
    # custody question is the adapted one — what independent receipt
    # would authenticate authorship — not the generic "evidence would
    # establish" question.
    assert any("independent receipt" in q for q in d["questions_to_ask"]), \
        f"expected a custody question, got {d['questions_to_ask']}"
    assert not any("implementation dates" in q for q in d["questions_to_ask"])
    assert not any("closest primary source" in q for q in d["questions_to_ask"])


@test("POST /api/v1/analyze asks a custody question for a declared primary with uncertain custody")
def _():
    # Nov 5 gap 4, two-stage system: declared primary, custody
    # uncertain (contradiction, conf 0.40) — the question must be
    # what evidence would establish authorship, never "where is the
    # original source?" for a doc claiming to BE it.
    r = client.post("/api/v1/analyze", json={
        "text": "According to reports, the new model is out today. Sources say it is "
                "fast and cheap, and everyone is talking about it. The launch happened "
                "quietly, with no official word yet.",
        "source_type": "primary",
    })
    d = r.get_json()
    assert any("evidence would establish" in q for q in d["questions_to_ask"]), \
        f"expected a custody question, got {d['questions_to_ask']}"
    assert not any("closest primary source" in q for q in d["questions_to_ask"])
    assert any("contradicts" in f for f in d["flags"])
    assert d["signal_brief"]["key_signals"], "downgrade with no cue must carry the no-cue sentinel"
    assert any("No supporting cue" in s for s in d["signal_brief"]["key_signals"])


@test("Hosted-artifact evidence requires trusted ownership, not a namespace match (v1.8.3)")
def _():
    # Arch's hierarchy (v1.8.1, gap 1) as tightened by her notes
    # 2026-11-06 (v1.8.3): an org-name match alone is NOT sufficient,
    # and neither is a namespace match. Level 1 (org named as
    # releaser/author) AND level 2 (official link) or level 3 (release
    # metadata) make a candidate — but only TRUSTED ownership evidence
    # verifies it, and no trusted mechanism exists in this system yet
    # (no verified-org registry; source URLs are caller-supplied).
    # So even the Baidu card — namespace matched — does NOT establish
    # hosted first-party evidence today. The wrapper exists for the
    # day a trusted channel lands.
    card = (
        "Baidu ERNIE Team — ERNIE-5.5-Flash-Base model card "
        "(Hugging Face, baidu/ERNIE-5.5-Flash-Base, released 5 November 2026). "
        "By releasing the weights, we let developers build on it. "
        "License: provided under the ERNIE 5.5 Flash Base Model License "
        "Agreement v1.0. Copyright 2026 Baidu."
    )
    fires, _reason = _hosted_release_artifact(card, "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base")
    assert not fires, "a namespace match alone must not establish hosted first-party evidence (v1.8.3)"
    # The namespace fact itself is still recorded independently.
    assert _hosted_namespace_matched(card, "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base", "baidu")[0]
    # Bare org-name match: Baidu mentioned, but never as releaser/author,
    # and no release metadata.
    assert not _hosted_release_artifact(
        "I love Baidu. Their models are great. Check out this page.",
        "https://huggingface.co/baidu/ernie")[0], "a bare org-name match must not establish ownership"
    # Right text, wrong host: not an artifact host.
    assert not _hosted_release_artifact(
        card, "https://example.com/baidu/ERNIE-5.5-Flash-Base")[0], \
        "a non-artifact host must not establish hosted ownership"
    # No URL at all: ownership cannot be established.
    assert not _hosted_release_artifact(card, "")[0]


@test("Hosted-artifact candidate, namespace match, and ownership are three separate questions (v1.8.3)")
def _():
    # Arch objection 2026-11-05 + Arch notes 2026-11-06: a document
    # must not certify itself merely by looking sufficiently official,
    # and a namespace match is not proof of verified organizational
    # control. Q1 (behaves like a first-party release artifact), Q2a
    # (the hosting location matches the claimed identity) and Q2b
    # (ownership independently established by trusted evidence) are
    # separate functions. The verified-only wrapper fires only when
    # Q1 and Q2b hold — and Q2b is currently unreachable, because no
    # trusted verification mechanism exists in this system.
    baidu = (
        "Baidu ERNIE Team — ERNIE-5.5-Flash-Base model card "
        "(Hugging Face, baidu/ERNIE-5.5-Flash-Base, released 5 November 2026). "
        "By releasing the weights, we let developers build on it. "
        "License: provided under the ERNIE 5.5 Flash Base Model License "
        "Agreement v1.0. Copyright 2026 Baidu."
    )
    fake = (
        "Acme Nova Team — Nova-7-Base model card "
        "(Hugging Face, randomuser/nova-7-base, released 5 November 2026). "
        "By releasing the weights, we let developers build on it. "
        "License: provided under the Acme Nova Model License Agreement v1.0. "
        "Copyright 2026 Acme. For more information, visit https://www.acme.com/nova-7-base"
    )
    # Baidu: candidate AND namespace matched (huggingface.co/baidu) —
    # but ownership NOT verified: a namespace match is not trusted
    # evidence, and no trusted mechanism exists.
    cand, org, _r = _hosted_artifact_candidate(baidu, "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base")
    assert cand and org == "baidu", f"Baidu card must be a candidate claiming baidu, got {cand}, {org!r}"
    ns, _r = _hosted_namespace_matched(baidu, "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base", org)
    assert ns, "Baidu namespace must match"
    ver, _r = _hosted_ownership_verified(baidu, "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base", org)
    assert not ver, "a namespace match alone must not verify ownership (v1.8.3)"
    assert not _hosted_release_artifact(baidu, "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base")[0]
    # Fake: candidate fires (it behaves like a release artifact) but
    # the namespace does NOT match ('randomuser' != claimed 'acme'),
    # so the verified-only wrapper must stay silent.
    cand, org, _r = _hosted_artifact_candidate(fake, "https://huggingface.co/randomuser/nova-7-base")
    assert cand and org == "acme", f"fake must be a candidate claiming acme, got {cand}, {org!r}"
    ns, _r = _hosted_namespace_matched(fake, "https://huggingface.co/randomuser/nova-7-base", org)
    assert not ns, "an unrelated namespace must not match"
    ver, _r = _hosted_ownership_verified(fake, "https://huggingface.co/randomuser/nova-7-base", org)
    assert not ver, "an unrelated namespace must not verify ownership"
    assert not _hosted_release_artifact(fake, "https://huggingface.co/randomuser/nova-7-base")[0], \
        "a self-certifying fake must not establish hosted first-party evidence"
    # Same fake, matching namespace: the namespace matches — and
    # ownership STILL does not verify (v1.8.3: the match is recorded,
    # not promoted). The wrapper stays silent.
    assert _hosted_namespace_matched(fake, "https://huggingface.co/acme/nova-7-base", "acme")[0]
    assert not _hosted_ownership_verified(fake, "https://huggingface.co/acme/nova-7-base", "acme")[0]
    assert not _hosted_release_artifact(fake, "https://huggingface.co/acme/nova-7-base")[0]
    # Same fake, hosted on the org's own domain: the namespace matches
    # via own-domain hosting — and ownership STILL does not verify.
    assert _hosted_namespace_matched(fake, "https://acme.com/nova-7-base", "acme")[0]
    assert not _hosted_ownership_verified(fake, "https://acme.com/nova-7-base", "acme")[0]
    # Bare org-name match: not even a candidate.
    cand, _org, _r = _hosted_artifact_candidate(
        "I love Baidu. Their models are great. Check out this page.",
        "https://huggingface.co/baidu/ernie")
    assert not cand, "a bare org-name match must not be a candidate"


@test("POST /api/v1/analyze keeps a self-certifying fake as candidate primary with uncertain ownership")
def _():
    # Arch objection 2026-11-05, end to end: the Acme fake claims
    # ownership with plausible metadata and a self-referential link,
    # hosted under an unrelated namespace. It must stay a candidate
    # primary with uncertain ownership — confidence capped below the
    # 0.70 verified-primary bar, never an anchor, flagged by name,
    # and the custody question asks what evidence would establish
    # authorship.
    text = (
        "Acme Nova Team — Nova-7-Base model card "
        "(Hugging Face, randomuser/nova-7-base, released 5 November 2026). "
        "Nova-7-Base is a sparse Mixture-of-Experts model with 120B total "
        "parameters and 8B activated per token, pre-trained from scratch on "
        "18T tokens. By releasing the weights, we let developers build on it. "
        "License: provided under the Acme Nova Model License Agreement v1.0. "
        "Copyright 2026 Acme. For more information, visit https://www.acme.com/nova-7-base"
    )
    r = client.post("/api/v1/analyze", json={
        "text": text,
        "source_type": "primary",
        "source_url": "https://huggingface.co/randomuser/nova-7-base",
    })
    d = r.get_json()
    assert d["source_type"] == "primary", f"candidate primary must stay primary, got {d['source_type']}"
    assert d["source_type_confidence"] <= 0.60, \
        f"uncertain ownership must cap confidence at 0.60, got {d['source_type_confidence']}"
    assert d["recommended_action"] != "use_as_anchor", "uncertain ownership must never anchor"
    assert any("uncertain ownership" in f for f in d["flags"]), f"ownership flag missing, got {d['flags']}"
    assert d["custody_breakdown"]["primary_source_present"] is False
    assert d["custody_breakdown"]["hosted_artifact_candidate"] is True
    assert d["custody_breakdown"]["hosted_namespace_matched"] is False
    assert d["custody_breakdown"]["hosted_ownership_status"] == "candidate_uncertain"
    assert d["custody_breakdown"]["hosted_first_party_artifact"] is False
    assert d["custody_breakdown"]["source_url_provenance"] == "caller_supplied"
    # v1.8.6 (P4): hosted candidate -> adapted custody question.
    assert any("independent receipt" in q for q in d["questions_to_ask"]), \
        f"expected a custody question, got {d['questions_to_ask']}"


@test("Namespace impersonation must not match: a lookalike slug is not the org (v1.8.3)")
def _():
    # Arch notes 2026-11-06: someone might register a plausible
    # organization name without being authorized to represent that
    # organization. A slug that merely contains the org name — or
    # appends a plausible suffix — is not the org's namespace.
    # Q1 still fires (the doc behaves like a release artifact), but
    # the namespace must NOT match and ownership stays uncertain.
    fake = (
        "Acme Nova Team — Nova-7-Base model card "
        "(Hugging Face, acme-official/nova-7-base, released 5 November 2026). "
        "By releasing the weights, we let developers build on it. "
        "License: provided under the Acme Nova Model License Agreement v1.0. "
        "Copyright 2026 Acme. For more information, visit https://www.acme.com/nova-7-base"
    )
    cand, org, _r = _hosted_artifact_candidate(fake, "https://huggingface.co/acme-official/nova-7-base")
    assert cand and org == "acme", f"impersonator must still be a candidate claiming acme, got {cand}, {org!r}"
    assert not _hosted_namespace_matched(fake, "https://huggingface.co/acme-official/nova-7-base", "acme")[0], \
        "a lookalike slug ('acme-official') must not match the org namespace"
    assert not _hosted_namespace_matched(fake, "https://huggingface.co/xacme/nova-7-base", "acme")[0], \
        "a slug merely containing the org name ('xacme') must not match"
    assert not _hosted_namespace_matched(fake, "https://huggingface.co/acme2/nova-7-base", "acme")[0], \
        "a slug with a trailing digit ('acme2') must not match"
    assert not _hosted_ownership_verified(fake, "https://huggingface.co/acme-official/nova-7-base", "acme")[0]
    assert not _hosted_release_artifact(fake, "https://huggingface.co/acme-official/nova-7-base")[0]
    # End to end: the impersonator is a capped candidate, never an anchor.
    r = client.post("/api/v1/analyze", json={
        "text": fake,
        "source_type": "primary",
        "source_url": "https://huggingface.co/acme-official/nova-7-base",
    })
    d = r.get_json()
    assert d["source_type_confidence"] <= 0.60
    assert d["recommended_action"] != "use_as_anchor"
    assert d["custody_breakdown"]["hosted_namespace_matched"] is False
    assert d["custody_breakdown"]["hosted_ownership_status"] == "candidate_uncertain"
    assert any("uncertain ownership" in f for f in d["flags"])


@test("Misleading hostnames must not match the namespace or verify ownership (v1.8.3)")
def _():
    # Arch notes 2026-11-06: an organization-name match inside an
    # arbitrary hostname is not sufficient to establish ownership of
    # that hostname. The host must BE an artifact host (exact or a
    # subdomain of it) or the org's own registrable domain — a
    # hostname that merely mentions the org, or the artifact host,
    # anywhere in it is not a match.
    fake = (
        "Acme Nova Team — Nova-7-Base model card "
        "(Hugging Face, acme/nova-7-base, released 5 November 2026). "
        "By releasing the weights, we let developers build on it. "
        "License: provided under the Acme Nova Model License Agreement v1.0. "
        "Copyright 2026 Acme."
    )
    # Artifact-host lookalikes: the host is not huggingface.co.
    assert not _hosted_namespace_matched(fake, "https://huggingface.co.attacker.com/acme/nova-7-base", "acme")[0], \
        "huggingface.co.attacker.com is not an artifact host"
    assert not _hosted_namespace_matched(fake, "https://huggingface.co.evil.com/acme/nova-7-base", "acme")[0], \
        "huggingface.co.evil.com is not an artifact host"
    assert not _hosted_namespace_matched(fake, "https://acme.huggingface.co.evil/nova-7-base", "acme")[0], \
        "an org-name label inside an arbitrary hostname is not the org's domain"
    assert not _hosted_namespace_matched(fake, "https://evil.com/acme/nova-7-base", "acme")[0], \
        "an arbitrary host with the org as a path segment is not a namespace match"
    assert not _hosted_namespace_matched(fake, "https://notacme.com/nova-7-base", "acme")[0], \
        "a domain merely containing the org name is not the org's domain"
    # None of these verifies ownership either.
    for url in (
        "https://huggingface.co.attacker.com/acme/nova-7-base",
        "https://huggingface.co.evil.com/acme/nova-7-base",
        "https://acme.huggingface.co.evil/nova-7-base",
        "https://evil.com/acme/nova-7-base",
    ):
        assert not _hosted_ownership_verified(fake, url, "acme")[0]
        assert not _hosted_release_artifact(fake, url)[0]
    # Positive controls for the tightened own-domain rule: the org slug
    # must be the registrable-domain label (before the TLD, or before
    # a known second-level public suffix).
    assert _hosted_namespace_matched(fake, "https://acme.com/nova-7-base", "acme")[0]
    assert _hosted_namespace_matched(fake, "https://www.acme.com/nova-7-base", "acme")[0]
    assert _hosted_namespace_matched(fake, "https://research.acme.com/nova-7-base", "acme")[0]
    assert _hosted_namespace_matched(fake, "https://acme.co.uk/nova-7-base", "acme")[0]
    # ...and the own-domain match is still only a namespace fact, never
    # ownership verification.
    assert not _hosted_ownership_verified(fake, "https://acme.com/nova-7-base", "acme")[0]


# ── v1.8.4: adjacent-window routes (Arch review 2026-11-06) ─────────────

@test("A quoted DOI is not first-party evidence (v1.8.4)")
def _():
    # Arch review 2026-11-06: a document could contain a valid DOI
    # referring to somebody else's research. A DOI proves the cited
    # work exists; it does not prove this document authored it. The
    # same holds for in-text mentions of arxiv.org / sec.gov / EDGAR:
    # mentioning a source is reporting on it, not being it. None of
    # these may satisfy _has_first_party_primary_evidence.
    doi_quote = (
        "A new study (doi:10.1038/s41586-024-12345-6) reports that a novel "
        "catalyst doubles reaction efficiency. The paper's abstract describes "
        "the methodology and a sample size of 240 participants."
    )
    assert not _has_first_party_primary_evidence(doi_quote, ""), \
        "a quoted DOI must not count as first-party evidence"
    assert not _has_first_party_primary_evidence(
        "For the full results see doi:10.1038/s41586-024-12345-6.", ""), \
        "a bare DOI citation must not count as first-party evidence"
    edgar = (
        "According to the EDGAR filing, the company reported revenue of $4.2 "
        "billion. The filing describes the methodology and the sample size."
    )
    assert not _has_first_party_primary_evidence(edgar, ""), \
        "mentioning an EDGAR filing must not count as first-party evidence"
    assert not _has_first_party_primary_evidence(
        "The preprint is available on arxiv.org. The abstract describes a new methodology.", ""), \
        "mentioning arxiv.org must not count as first-party evidence"
    assert not _has_first_party_primary_evidence(
        "The sec.gov filing shows the fund held 42 positions.", ""), \
        "mentioning sec.gov must not count as first-party evidence"
    # What DOES count as research evidence: the authors speaking in
    # the first person about work they did. The pinned trial abstract
    # passes through this route, not its DOI.
    abstract = (
        "Abstract. We conducted a randomized controlled trial of 214 participants "
        "(n=214) to evaluate the effect of the intervention. The primary "
        "endpoint changed by 4.2 points (95% CI 2.1-6.3, p=0.003) at 12 weeks."
    )
    assert _first_person_research(abstract)[0], \
        "first-person research conduct must count as authorship evidence"
    assert _has_first_party_primary_evidence(abstract, ""), \
        "the trial abstract must still pass the gate via research voice"
    abstract_no_doi = abstract  # no DOI in this text at all
    assert _has_first_party_primary_evidence(abstract_no_doi, ""), \
        "the abstract must pass the gate with no DOI present"
    # A third party describing the same trial does NOT inherit the voice.
    summary = (
        "A new study reports a randomized controlled trial of 214 participants. "
        "According to the paper, the primary endpoint changed by 4.2 points."
    )
    assert not _first_person_research(summary)[0], \
        "third-party attribution must veto the research voice"
    assert not _has_first_party_primary_evidence(summary, ""), \
        "a third-party summary must not pass the gate"


@test("Academic hosts match exactly or as subdomains, never as substrings (v1.8.4)")
def _():
    # Arch review 2026-11-06: the academic-host check used substring
    # matching, so an untrusted hostname merely containing a
    # recognizable academic string counted as that institution.
    assert _host_is_or_subdomain("stanford.edu", "stanford.edu")
    assert _host_is_or_subdomain("www.stanford.edu", "stanford.edu")
    assert _host_is_or_subdomain("arxiv.org", "arxiv.org")
    assert _host_is_or_subdomain("www.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov")
    assert _host_is_or_subdomain("pubmed.ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov")
    assert not _host_is_or_subdomain("stanford.edu.attacker.com", "stanford.edu"), \
        "stanford.edu.attacker.com is not Stanford"
    assert not _host_is_or_subdomain("notnature.com", "nature.com"), \
        "notnature.com is not Nature"
    assert not _host_is_or_subdomain("arxiv.org.evil.com", "arxiv.org"), \
        "arxiv.org.evil.com is not arXiv"
    assert not _host_is_or_subdomain("pubmed.ncbi.evil.com", "pubmed.ncbi.nlm.nih.gov"), \
        "a lookalike NCBI prefix on an untrusted host is not NCBI"
    # End to end through the evidence function: attacker hosts must
    # not satisfy first-party evidence, genuine ones must.
    plain = (
        "A new benchmark for frontier models was released this week. The report "
        "covers 206 tasks across 11 domains."
    )
    assert not _has_first_party_primary_evidence(plain, "https://stanford.edu.attacker.com/paper.pdf")
    assert not _has_first_party_primary_evidence(plain, "https://notnature.com/article")
    assert not _has_first_party_primary_evidence(plain, "https://arxiv.org.evil.com/paper")
    assert _has_first_party_primary_evidence(plain, "https://www.stanford.edu/news/benchmark"), \
        "a genuine academic host must still count as first-party evidence"


@test("POST /api/v1/analyze: a quoted DOI must not anchor an unverified document (v1.8.4)")
def _():
    # End to end: the DOI-quote case that anchored (use_as_anchor)
    # before v1.8.4. The inference drops to secondary with the
    # custody-uncertain flag, and the generic question is asked.
    text = (
        "A new study (doi:10.1038/s41586-024-12345-6) reports that a novel "
        "catalyst doubles reaction efficiency. The paper's abstract describes "
        "the methodology and a sample size of 240 participants, with results "
        "significant at p=0.01."
    )
    r = client.post("/api/v1/analyze", json={"text": text, "source_type": "auto"})
    d = r.get_json()
    assert d["source_type"] == "secondary", \
        f"a DOI quote must not report as primary, got {d['source_type']}"
    assert d["recommended_action"] != "use_as_anchor", "a DOI quote must never anchor"
    assert any("Source custody uncertain" in f for f in d["flags"]), \
        f"custody-uncertain flag missing, got {d['flags']}"
    assert any("closest primary source" in q for q in d["questions_to_ask"]), \
        f"expected the generic primary-source question, got {d['questions_to_ask']}"


@test("POST /api/v1/analyze: a hosted fake stays capped when a DOI or announcement voice is added (v1.8.4)")
def _():
    # End to end: before v1.8.4, one quoted DOI sentence — or one
    # "Today we are releasing..." sentence — counted as independent
    # first-party evidence and LIFTED the hosted-ownership cap, so the
    # Acme fake anchored at 0.90. The cap now lifts only on verified
    # ownership (Q2b, currently unreachable). Both variants stay capped
    # candidates exactly like the unchanged fake.
    base = (
        "Acme Nova Team — Nova-7-Base model card "
        "(Hugging Face, randomuser/nova-7-base, released 5 November 2026). "
        "Nova-7-Base is a sparse Mixture-of-Experts model with 120B total "
        "parameters and 8B activated per token, pre-trained from scratch on "
        "18T tokens. By releasing the weights, we let developers build on it. "
        "License: provided under the Acme Nova Model License Agreement v1.0. "
        "Copyright 2026 Acme. For more information, visit https://www.acme.com/nova-7-base"
    )
    for extra in (
        "\n\nFor the full evaluation protocol see doi:10.48550/arXiv.2601.00001.",
        "\n\nToday we are releasing Nova-7-Base. We are announcing the weights "
        "are available now, and we are publishing the full evaluation results "
        "alongside the model card.",
    ):
        r = client.post("/api/v1/analyze", json={
            "text": base + extra,
            "source_type": "primary",
            "source_url": "https://huggingface.co/randomuser/nova-7-base",
        })
        d = r.get_json()
        assert d["source_type_confidence"] <= 0.60, \
            f"fake+extra must stay capped at 0.60, got {d['source_type_confidence']}"
        assert d["recommended_action"] != "use_as_anchor", \
            "fake+extra must never anchor"
        assert any("uncertain ownership" in f for f in d["flags"]), \
            f"ownership flag missing, got {d['flags']}"
        assert d["custody_breakdown"]["hosted_ownership_status"] == "candidate_uncertain"
        assert d["custody_breakdown"]["primary_source_present"] is False


@test("POST /api/v1/analyze: a misleading academic hostname must not earn primary type (v1.8.4)")
def _():
    # End to end: the same plain article on stanford.edu.attacker.com
    # reported as primary at 0.75 before v1.8.4. Strict host matching
    # leaves it unknown — and it must never anchor.
    text = (
        "A new benchmark for frontier models was released this week. The report "
        "covers 206 tasks across 11 domains and finds the best agent scores just "
        "12 percent. Coverage of the launch continues across the trade press."
    )
    for url in (
        "https://stanford.edu.attacker.com/paper.pdf",
        "https://arxiv.org.evil.com/paper/1234",
    ):
        r = client.post("/api/v1/analyze", json={
            "text": text, "source_type": "auto", "source_url": url,
        })
        d = r.get_json()
        assert d["source_type"] == "unknown", \
            f"attacker host {url} must not earn a type, got {d['source_type']}"
        assert d["recommended_action"] != "use_as_anchor"
    # Positive control: the genuine host still counts as evidence
    # (the gate passes, so no custody-uncertain flag fires).
    r = client.post("/api/v1/analyze", json={
        "text": text, "source_type": "auto",
        "source_url": "https://www.stanford.edu/news/benchmark",
    })
    d = r.get_json()
    assert not any("Source custody uncertain" in f for f in d["flags"]), \
        f"genuine academic host must pass the gate, got {d['flags']}"


@test("extract_numeric_claims never mints a claim from a shot count or a model name")
def _():
    # Direct extraction checks for the v1.8.1 guards: the "5" in
    # "5-shot", the "3" in "Hy3", the "8" in "FP8", and the "1" in
    # "Pass@1" are conditions and name fragments, not values.
    claims = extract_numeric_claims(
        "MMLU (EM) 5-shot: ERNIE 93.3, DeepSeek-V4-Flash 89.4. "
        "HumanEval+ (Pass@1) 1-shot: ERNIE 82.3. "
        "Downloads are available in FP8 and BF16 formats."
    )
    by_key = {}
    for c in claims:
        by_key.setdefault(c["entity_key"], []).append(c)
    assert [c["numeric_value"] for c in by_key.get("mmlu", [])] == [93.3], \
        f"mmlu must be only the 93.3 score, got {by_key.get('mmlu')}"
    assert by_key["mmlu"][0]["shot_count"] == 5
    assert [c["numeric_value"] for c in by_key.get("humaneval", [])] == [82.3], \
        f"humaneval must be only the 82.3 score, got {by_key.get('humaneval')}"
    assert by_key["humaneval"][0]["shot_count"] == 1
    assert "dow" not in by_key, f"Downloads/FP8 must not mint a Dow claim, got {by_key.get('dow')}"


# ── v1.8.6: generic quantitative fallback + date roles ───────────────────────

@test("POST /api/v1/compare aligns vendor performance claims across docs (P2)")
def _():
    # Nov 8 gap 1: the 23% / 9% / 30-node / 200-turn / 26 / 16 claims
    # extracted nothing before the generic quantitative fallback. The
    # context numbers (30-node, 200-turn) are part of the claim's
    # identity, so a bare 23% elsewhere would not merge with these.
    a = ("A 30-node workflow's 200-turn run is 23% faster with 9% lower "
         "peak memory. The release graduates 26 modules to stable, "
         "including 16 features.")
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": a, "source_name": "A"},
            {"text": a, "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = {c["entity_key"]: [v["value"] for v in c.get("values", [])]
               for c in d.get("alignment", [])}
    # v1.8.8: "23% faster" and "9% lower" are relative deltas, so the
    # keys carry the "rel" marker — the figures still align, under the
    # relative identity.
    assert "generic|percentage|speed|nodes=30;turns=200;rel||unstated" in aligned, \
        f"expected speed aligned, got {list(aligned)}"
    assert aligned["generic|percentage|speed|nodes=30;turns=200;rel||unstated"] == [23.0, 23.0]
    assert aligned.get("generic|percentage|memory|nodes=30;turns=200;rel||unstated") == [9.0, 9.0], \
        f"expected memory aligned, got {list(aligned)}"
    assert aligned.get("generic|task_count|modules|||unstated") == [26.0, 26.0]
    assert aligned.get("generic|task_count|features|||unstated") == [16.0, 16.0]
    assert not d.get("conflicts"), f"identical vendor claims must not conflict, got {d.get('conflicts')}"


@test("POST /api/v1/compare keeps two subjects of one family apart (P2)")
def _():
    # Nov 7: GPT-6 Cyber 99.93% vs GPT-5.6 Cyber 94.49% avoidance are
    # two values of one family, not a conflict. Each subject's value
    # aligns with its own twin; nothing conflicts.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "GPT-6 Cyber avoided exploiting out-of-scope services in "
                     "99.93% of tests, up from 94.49% for GPT-5.6 Cyber.",
             "source_name": "A"},
            {"text": "GPT-6 Cyber avoided exploiting out-of-scope services in "
                     "99.93% of tests. GPT-5.6 Cyber avoided exploiting "
                     "out-of-scope services in 94.49% of tests.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = {c["entity_key"]: [v["value"] for v in c.get("values", [])]
               for c in d.get("alignment", [])}
    assert aligned.get("generic|percentage|out-of-scope avoidance|||gpt-6-cyber") == [99.93, 99.93], \
        f"expected GPT-6 99.93 aligned, got {aligned}"
    assert aligned.get("generic|percentage|out-of-scope avoidance|||gpt-5-6-cyber") == [94.49, 94.49], \
        f"expected GPT-5.6 94.49 aligned, got {aligned}"
    assert not d.get("conflicts"), f"two subjects must not conflict, got {d.get('conflicts')}"


@test("POST /api/v1/compare aligns number words with digits (P2)")
def _():
    # Nov 8: "Twenty-six modules" and "26 modules" are the same claim;
    # "Sixteen features" and "16 features" likewise.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "The release graduates 26 modules to stable, including 16 features.",
             "source_name": "A"},
            {"text": "The release graduates Twenty-six modules to stable, including Sixteen features.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = {c["entity_key"]: [v["value"] for v in c.get("values", [])]
               for c in d.get("alignment", [])}
    assert aligned.get("generic|task_count|modules|||unstated") == [26.0, 26.0], \
        f"expected modules 26 aligned, got {aligned}"
    assert aligned.get("generic|task_count|features|||unstated") == [16.0, 16.0], \
        f"expected features 16 aligned, got {aligned}"
    assert not d.get("conflicts")


@test("POST /api/v1/compare extracts ISO timestamps as dates (P5)")
def _():
    # Nov 7 gap 4: "2026-11-06T10:15:00Z" extracted nothing before the
    # ISO-with-time alternative was added to the date regex.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Updated 2026-11-06T10:15:00Z. The release is out.",
             "source_name": "A"},
            {"text": "Updated 2026-11-06T10:15:00Z. The release is out.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    shared = {x["date_key"]: x for x in d.get("shared_dates", [])}
    assert "2026-11-06" in shared, f"expected ISO timestamp shared, got {list(shared)}"
    assert shared["2026-11-06"]["roles"] == {"A": ["updated"], "B": ["updated"]}, \
        f"expected updated roles, got {shared['2026-11-06']['roles']}"


@test("POST /api/v1/compare labels fetched vs published date roles (P5)")
def _():
    # A "Fetched 2026-11-08" feed header is a fetched date, not a
    # published one; a "Published: November 7, 2026" line is a
    # published date even when the previous sentence was a fetch.
    # (The nearest label cue wins — a fetched header must not bleed
    # into the next sentence's published date.)
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Fetched 2026-11-08. Published: November 7, 2026. "
                     "The release is out.", "source_name": "A"},
            {"text": "Published: November 7, 2026. The release is out.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    shared = {x["date_key"]: x for x in d.get("shared_dates", [])}
    assert "2026-11-07" in shared, f"expected 2026-11-07 shared, got {list(shared)}"
    assert shared["2026-11-07"]["roles"] == {"A": ["published"], "B": ["published"]}, \
        f"expected published roles, got {shared['2026-11-07']['roles']}"
    unique = {x["date_key"]: x for x in d.get("unique_dates", [])}
    assert "2026-11-08" in unique, f"expected 2026-11-08 unique, got {list(unique)}"
    assert unique["2026-11-08"]["roles"] == {"A": ["fetched"]}, \
        f"expected fetched role, got {unique['2026-11-08']['roles']}"


@test("POST /api/v1/compare treats an up-to ceiling as compatible (P2)")
def _():
    # "up to 95% completion" is a ceiling: a plain 95% and a plain 90%
    # both sit under it, so neither is a conflict.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "The model reached up to 95% completion on the tasks.",
             "source_name": "A"},
            {"text": "The model reached 95% completion on the tasks.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = {c["entity_key"]: [v["value"] for v in c.get("values", [])]
               for c in d.get("alignment", [])}
    assert aligned.get("generic|percentage|completion|||unstated") == [95.0, 95.0], \
        f"expected ceiling 95 aligned with plain 95, got {aligned}"
    assert not d.get("conflicts")
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "The model reached up to 95% completion on the tasks.",
             "source_name": "A"},
            {"text": "The model reached 90% completion on the tasks.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not [c for c in d.get("conflicts", [])
                if c["entity_key"] == "generic|percentage|completion|||unstated"], \
        "a value under the ceiling must not conflict with it"


@test("POST /api/v1/compare never mints a claim from a version string (P2)")
def _():
    # "v2.14.0" and "GPT-6" are versions and model names, not values.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Release v2.14.0 is available. GPT-6 is the new model.",
             "source_name": "A"},
            {"text": "Release v2.14.0 is available. GPT-6 is the new model.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    keys = ([c["entity_key"] for c in d.get("shared_claims", [])] +
            [c["entity_key"] for c in d.get("alignment", [])] +
            [c["entity_key"] for c in d.get("unique_claims", [])])
    assert not keys, f"version strings must mint no claims, got {keys}"


@test("POST /api/v1/compare never mints a claim from a version string (P2)")
def _():
    # "v2.14.0" and "GPT-6" are versions and model names, not values.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Release v2.14.0 is available. GPT-6 is the new model.",
             "source_name": "A"},
            {"text": "Release v2.14.0 is available. GPT-6 is the new model.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    keys = ([c["entity_key"] for c in d.get("shared_claims", [])] +
            [c["entity_key"] for c in d.get("alignment", [])] +
            [c["entity_key"] for c in d.get("unique_claims", [])])
    assert not keys, f"version strings must mint no claims, got {keys}"


# ── v1.8.7: typed assertions, minimum identity, fail-closed gate ────────

@test("POST /api/v1/compare never mints a model version as a measurement (v1.8.7)")
def _():
    # Run 11 live failure: Fable 5.2, Opus 5.5, FrontierCode 2.0,
    # OSWorld 2.1, CL-Bench 1.0 minted as scores/measurements.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("Fable 5.2 scored 49.1% on FrontierCode 2.0. Opus 5.5 scored "
                      "45.5%. On OSWorld 2.1, Fable 5.2 scored 74.2% on Humanity's "
                      "Last Exam. CL-Bench 1.0 results: Opus 5.5 0.655 and 0.838."),
             "source_name": "A"},
            {"text": ("Fable 5.2 scored 49.1% on FrontierCode 2.0. Opus 5.5 scored "
                      "45.5%. On OSWorld 2.1, Fable 5.2 scored 74.2% on Humanity's "
                      "Last Exam. CL-Bench 1.0 results: Opus 5.5 0.655 and 0.838."),
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    all_vals = ([v["value"] for c in d.get("alignment", []) for v in c.get("values", [])] +
                [v["value"] for c in d.get("conflicts", []) for v in c.get("values", [])] +
                [c["value"] for c in d.get("unique_claims", [])])
    for version in (5.2, 5.5, 2.0, 2.1, 1.0):
        assert version not in all_vals, f"version {version} minted as a value: {all_vals}"
    assert 49.1 in all_vals and 45.5 in all_vals and 74.2 in all_vals, \
        f"real scores must mint, got {all_vals}"
    assert not d.get("conflicts"), f"identical scores must not conflict, got {d.get('conflicts')}"


@test("POST /api/v1/compare never mints a month-day date as a count (v1.8.7)")
def _():
    # Run 11 live failure: "April 7" minted as 7 findings disclosed,
    # conflicting with 1,829 CVEs.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("Anthropic disclosed Claude Mythos Preview on April 7. "
                      "Conducted against 1,829 previously disclosed vulnerabilities."),
             "source_name": "A"},
            {"text": ("Anthropic disclosed Claude Mythos Preview on April 7. "
                      "Conducted against 1,829 previously disclosed vulnerabilities."),
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    all_vals = ([v["value"] for c in d.get("alignment", []) for v in c.get("values", [])] +
                [c["value"] for c in d.get("unique_claims", [])])
    assert 7.0 not in all_vals, f"April 7 minted as a count: {all_vals}"
    assert 1829.0 in all_vals, f"1,829 CVEs must mint, got {all_vals}"
    assert not d.get("conflicts"), f"date vs CVE count must not conflict, got {d.get('conflicts')}"


@test("POST /api/v1/compare keeps pricing tiers and eras apart (v1.8.7)")
def _():
    # Run 11 live failure: $8/$16/$80 and $10/$50 (prior) conflated;
    # fast-mode prices swapped with standard ones.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("Pricing: $8 per million input tokens and $40 per million output "
                      "tokens. Fast mode: $16 per million input tokens and $80 per "
                      "million output tokens. Previously $10 per million input tokens "
                      "and $50 per million output tokens."),
             "source_name": "A"},
            {"text": ("Pricing: $8 per million input tokens and $40 per million output "
                      "tokens. Fast mode: $16 per million input tokens and $80 per "
                      "million output tokens. Previously $10 per million input tokens "
                      "and $50 per million output tokens."),
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = {}
    for c in d.get("alignment", []):
        aligned.setdefault(c["entity_key"], []).append(
            sorted(v["value"] for v in c.get("values", [])))
    assert [8.0, 8.0] in aligned.get("pricing-input", []), \
        f"current input must align at 8, got {aligned}"
    assert [40.0, 40.0] in aligned.get("pricing-output", []), \
        f"current output must align at 40, got {aligned}"
    assert [16.0, 16.0] in aligned.get("pricing-fast-input", []), \
        f"fast input must align at 16, got {aligned}"
    assert [80.0, 80.0] in aligned.get("pricing-fast-output", []), \
        f"fast output must align at 80, got {aligned}"
    assert [10.0, 10.0] in aligned.get("pricing-input", []), \
        f"prior input must align at 10 in its own era, got {aligned}"
    assert [50.0, 50.0] in aligned.get("pricing-output", []), \
        f"prior output must align at 50 in its own era, got {aligned}"
    assert not d.get("conflicts"), f"identical pricing must not conflict, got {d.get('conflicts')}"


@test("POST /api/v1/compare keeps two models' per-task costs apart (v1.8.7)")
def _():
    # Run 11 live failure: $9.08 (Fable 5.2 High) vs $12.82 (Opus 5.5
    # Max) read as a conflict — different models' costs are a
    # comparison, not a dispute.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("Fable 5.2 costs $9.08 per task. Opus 5.5 costs $12.82 per task."),
             "source_name": "A"},
            {"text": ("Fable 5.2 costs $9.08 per task. Opus 5.5 costs $12.82 per task."),
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = {}
    for c in d.get("alignment", []):
        aligned.setdefault(c["entity_key"], []).append(
            sorted(v["value"] for v in c.get("values", [])))
    assert [9.08, 9.08] in aligned.get("pricing-per-task", []), \
        f"Fable 5.2 per-task cost must align, got {aligned}"
    assert [12.82, 12.82] in aligned.get("pricing-per-task", []), \
        f"Opus 5.5 per-task cost must align, got {aligned}"
    assert not d.get("conflicts"), f"two models' costs must not conflict, got {d.get('conflicts')}"


@test("POST /api/v1/compare never lets one document self-conflict (v1.8.7)")
def _():
    # Arch gate point 3: a benchmark table row is a value set with
    # different qualified identities, not a dispute.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("- FrontierCode 2.0 agentic coding: Fable 5.2 49.1%, Fable 5.1 32.4%, "
                      "Opus 5.5 45.5%, GPT-6 Astra 42.8%."),
             "source_name": "A"},
            {"text": "No benchmark table in this document.",
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"one doc can never self-conflict, got {d.get('conflicts')}"
    vals = sorted(c["value"] for c in d.get("unique_claims", [])
                  if "score" in c["entity_key"])
    assert 49.1 in vals and 32.4 in vals and 45.5 in vals and 42.8 in vals, \
        f"table-row scores must surface as uniques, got {vals}"


@test("POST /api/v1/compare mints table-row benchmark scores (v1.8.7)")
def _():
    # Run 11 live failure: the FrontierCode table rows minted nothing.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("- FrontierCode 2.0 agentic coding: Fable 5.2 49.1%, Fable 5.1 32.4%, "
                      "Opus 5.5 45.5%, GPT-6 Astra 42.8%."),
             "source_name": "A"},
            {"text": ("- FrontierCode 2.0 agentic coding: Fable 5.2 49.1%, Fable 5.1 32.4%, "
                      "Opus 5.5 45.5%, GPT-6 Astra 42.8%."),
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    aligned = {c["entity_key"]: sorted(v["value"] for v in c.get("values", []))
               for c in d.get("alignment", [])}
    fc = [k for k in aligned if "frontiercode" in k.lower() or "score" in k]
    assert fc, f"FrontierCode rows must align, got {aligned}"
    all_vals = sorted(v for vs in aligned.values() for v in vs)
    assert 49.1 in all_vals and 45.5 in all_vals, f"table scores must align, got {all_vals}"
    assert not d.get("conflicts")


@test("POST /api/v1/compare mints durations and retention, not bare versions (v1.8.7)")
def _():
    # Run 11 live failure: "30 hours" / "30-day retention" unextracted;
    # the Fable 5.2 version took the retention cue instead.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("A team of 1,000 Fable 5.2 agents fully formalized Fermat's Last "
                      "Theorem in just 30 hours. Fable 5.2 and Mythos 5.2 require "
                      "30-day data retention."),
             "source_name": "A"},
            {"text": ("A team of 1,000 Fable 5.2 agents fully formalized Fermat's Last "
                      "Theorem in just 30 hours. Fable 5.2 and Mythos 5.2 require "
                      "30-day data retention."),
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    all_vals = ([v["value"] for c in d.get("alignment", []) for v in c.get("values", [])] +
                [c["value"] for c in d.get("unique_claims", [])])
    assert 30.0 in all_vals, f"30 hours / 30-day retention must mint, got {all_vals}"
    assert 1000.0 in all_vals, f"1,000 agents must mint, got {all_vals}"
    assert not d.get("conflicts")


@test("POST /api/v1/compare lists bare pricing as unmatched, never compared (v1.8.7)")
def _():
    # Arch gate point 4: an unresolvable tier is fail-closed. The
    # amount mints (a "per million" minting word) but no tier word
    # can be assigned, so it lands in unmatched_claims, never compared.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "The model costs $10.02 per million tokens.", "source_name": "A"},
            {"text": "The model costs $10.02 per million tokens.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), "bare pricing must never conflict"
    unmatched = d.get("unmatched_claims", [])
    assert unmatched, "bare pricing must land in unmatched_claims"
    assert all(u["entity_key"] == "pricing" for u in unmatched), \
        f"unmatched must be bare pricing, got {unmatched}"
    assert d.get("extraction_coverage"), "extraction_coverage must be present"


@test("POST /api/v1/compare aligns a CVE count across phrasings (v1.8.7)")
def _():
    # Run 11 live miss: "1,829 CVEs" (techtimes) vs "1,829 previously
    # disclosed vulnerabilities" (infoworld) minted under two families
    # ("vulnerabilities" vs "findings disclosed") and never aligned.
    # A CVE/vulnerabilities token right after the number names the
    # family before the nearer "disclosed" cue can claim it.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("On an internal CVE Rediscovery run in October across 1,829 CVEs, "
                      "Mythos 5.2 scored 89.6%."),
             "source_name": "A"},
            {"text": ("Conducted against 1,829 previously disclosed vulnerabilities, "
                      "the test assessed whether models could rediscover flaws."),
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"CVE count must never conflict, got {d.get('conflicts')}"
    aligned = [c for c in d.get("alignment", [])
               if any(v["value"] == 1829.0 for v in c.get("values", []))]
    assert aligned, f"1,829 CVE count must align, got {d.get('alignment')}"
    assert "vulnerabilities" in aligned[0]["entity_key"], \
        f"1,829 must be a vulnerabilities count, got {aligned[0]['entity_key']}"


@test("POST /api/v1/compare attaches a bench-less claim to the one stated-benchmark group (v1.8.7)")
def _():
    # Nov 7 regression: the 51% relative improvement (tsn names neither
    # benchmark nor model) and the ~16 point gap (tsn names the model,
    # primary names the benchmark) must both align with the primary's
    # benched claims — and two different models' scores must NOT
    # collapse into one bench-less group (74.6 vs 75.3 are different
    # models on different benchmarks).
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("GPT-6 Cyber scored 48.5% on ExploitGym, a 51% relative improvement "
                      "over GPT-5.6 Cyber's 32.1%. The gap is about 16 percentage points."),
             "source_name": "A"},
            {"text": ("GPT-6 Cyber's 48.5% on ExploitGym is a 51% relative improvement. "
                      "The percentage-point gap is about 16."),
             "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"bench-less attach must never conflict, got {d.get('conflicts')}"
    aligned_vals = [v["value"] for c in d.get("alignment", []) for v in c.get("values", [])]
    assert 51.0 in aligned_vals, f"51% relative improvement must align, got {aligned_vals}"
    assert 16.0 in aligned_vals, f"~16 point gap must align, got {aligned_vals}"


@test("POST /api/v1/analyze reads a multi-word secondary header org (v1.8.7)")
def _():
    # Run 11 live failure: TechTimes -> "understanding", InfoWorld ->
    # "greyhound", Medium -> "robodojo" (quoted cited sources).
    r = client.post("/api/v1/analyze", json={
        "text": ("COVERAGE — TechTimes\n"
                 "Anthropic announced Claude Fable 5.2 today. The new model "
                 "scored 49.1% on FrontierCode 2.0."),
        "source_type": "auto",
    })
    d = r.get_json()
    org = d["custody_breakdown"]["claimed_publisher_org"]
    assert org == "techtimes", \
        f"header org must be techtimes, got {org}"
    r = client.post("/api/v1/analyze", json={
        "text": ("SECONDARY — AI Weekly\n"
                 "A new model scored 49.1% on FrontierCode 2.0 this week."),
        "source_type": "auto",
    })
    d = r.get_json()
    org = d["custody_breakdown"]["claimed_publisher_org"]
    assert org == "ai weekly", \
        f"multi-word header org must be preserved, got {org}"


@test("POST /api/v1/analyze does not infer anonymous from MacRumors (v1.8.7)")
def _():
    # Run 11 live failure: "rumor" inside "MacRumors" inferred anonymous.
    r = client.post("/api/v1/analyze", json={
        "text": ("MacRumors reports that Apple will release a new chip next year. "
                 "The report cites supply-chain sources and a launch window."),
        "source_type": "auto",
    })
    d = r.get_json()
    assert d["inferred_source_type"] != "anonymous", \
        f"MacRumors must not infer anonymous, got {d['inferred_source_type']}"


@test("POST /api/v1/analyze skips the quote question with no attributed quote (v1.8.7)")
def _():
    # Run 11 live failure: generic quote-attribution questions asked
    # when no attributed quote exists.
    r = client.post("/api/v1/analyze", json={
        "text": ("The new model scored 49.1% on FrontierCode 2.0. "
                 "Pricing is $8 per million input tokens."),
        "source_type": "auto",
    })
    d = r.get_json()
    qs = d.get("follow_up_questions", [])
    assert not [q for q in qs if "quote" in q.lower() or "attributed" in q.lower()], \
        f"no quote question without an attributed quote, got {qs}"


@test("POST /api/v1/analyze marks an unauthenticated anchor origin as claimed (v1.8.7)")
def _():
    # Run 10/11 live failure: use_as_anchor with primary_source_present
    # false displayed without any distinction from authenticated origin.
    r = client.post("/api/v1/analyze", json={
        "text": ("Anthropic announced Claude Fable 5.2 today. The new model "
                 "scored 49.1% on FrontierCode 2.0 and costs $8 per million "
                 "input tokens. The announcement is dated November 9, 2026."),
        "source_type": "auto",
    })
    d = r.get_json()
    if d["recommended_action"] == "use_as_anchor":
        assert "claimed, not independently authenticated" in d["verdict"], \
            f"anchor verdict must carry the origin distinction, got {d['verdict']}"


# ── v1.8.8 residuals (run 11) ─────────────────────────────────────────────

@test("POST /api/v1/compare does not double-mint a table-row score (v1.8.8)")
def _():
    # Run 11: the primary's CVE row minted 89.6% TWICE (once as a
    # "vulnerabilities" claim from the sentence loop, once as a
    # score/bench claim from the table-row pass) because the upgrade
    # comparison kept the list bullet on one side. The same doc twice
    # must align each CVE value exactly twice — never four times.
    text = ("- CVE Rediscovery: Mythos 5.2 89.6%, Mythos 5.1 78.3%, "
            "Opus 5.5 81.3% without safeguards.")
    r = client.post("/api/v1/compare", json={
        "documents": [{"text": text, "source_name": "A"},
                      {"text": text, "source_name": "B"}],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"same doc twice must never conflict, got {d.get('conflicts')}"
    for val in (89.6, 78.3, 81.3):
        hits = [a for a in d.get("alignment", [])
                if val in [v["value"] for v in a.get("values", [])]]
        assert len(hits) == 1, f"{val} must align in exactly one group, got {len(hits)}"
        assert len(hits[0]["values"]) == 2, \
            f"{val} must have exactly 2 values (one per doc), got {len(hits[0]['values'])}"


@test("POST /api/v1/compare aligns InfoWorld 89.6% three-way (v1.8.8)")
def _():
    # Run 11: InfoWorld's prose phrasing ("identified 89.6% of
    # vulnerabilities in its internal CVE Rediscovery evaluation")
    # minted as a vulnerabilities count and never attached to the
    # table-row score. It must now align 3-way with the primary and
    # TechTimes; the 78.3% and 81.3% baselines align 2-way each.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("- CVE Rediscovery: Mythos 5.2 89.6%, Mythos 5.1 78.3%, "
                      "Opus 5.5 81.3% without safeguards."),
             "source_name": "primary"},
            {"text": ("Anthropic said Mythos 5.2 identified 89.6% of vulnerabilities "
                      "in its internal CVE Rediscovery evaluation, compared with "
                      "78.3% for Mythos 5.1."),
             "source_name": "infoworld"},
            {"text": ("On an internal CVE Rediscovery run in October across 1,829 CVEs, "
                      "Mythos 5.2 scored 89.6%, against 81.3% for Opus 5.5 without safeguards."),
             "source_name": "techtimes"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"CVE rows must never conflict, got {d.get('conflicts')}"
    by_key = {a["entity_key"]: [v["value"] for v in a.get("values", [])]
              for a in d.get("alignment", [])}
    assert by_key.get("generic|percentage|score|bench=cve-rediscovery||mythos-5-2") == [89.6, 89.6, 89.6], \
        f"89.6% must align 3-way, got {by_key}"
    assert sorted(by_key.get("generic|percentage|score|bench=cve-rediscovery||mythos-5-1", [])) == [78.3, 78.3], \
        f"78.3% must align 2-way, got {by_key}"
    assert sorted(by_key.get("generic|percentage|score|bench=cve-rediscovery||opus-5-5", [])) == [81.3, 81.3], \
        f"81.3% must align 2-way, got {by_key}"


@test("POST /api/v1/compare aligns bare-fragment row scores (v1.8.8)")
def _():
    # Run 11 medium coverage 0.38: bare "Model N%" fragments in later
    # clauses of a benchmark row line ("Fable 5.1 31.4% matches")
    # never minted, because the row parser only reads a row's first
    # clause and the sentence has no metric cue. They must now mint
    # as that row's benchmark score and align with the primary's rows.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("- AutomationBench: Fable 5.2 51.4%, Opus 5.5 42.5%, Astra 41.4%, "
                      "Carbon 53.9% — owner board: 51.45%, 42.47%, 41.40%, 53.88%; matches. "
                      "Fable 5.1 31.4% matches (without fallback 22.37%).\n"
                      "- Terminal-Bench-Science 0.1: Astra 68.1% matches. Fable 5.1 52.6% "
                      "and Opus 5.5 58.7% differ: owner board shows 40.0% and 63.3%."),
             "source_name": "medium"},
            {"text": ("- AutomationBench: Fable 5.2 51.4%, Fable 5.1 31.4%, Opus 5.5 42.5%, "
                      "Astra 41.4%, Carbon 53.9%\n"
                      "- Terminal-Bench-Science: Fable 5.2 74.6%, Mythos 5.2 75.3%, "
                      "Fable 5.1 52.6%, Opus 5.5 58.7%, Astra 68.1%, Carbon 64.8%"),
             "source_name": "primary"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"fragment scores must never conflict, got {d.get('conflicts')}"
    by_key = {a["entity_key"]: [v["value"] for v in a.get("values", [])]
              for a in d.get("alignment", [])}
    assert sorted(by_key.get("generic|percentage|score|bench=automationbench||fable-5-1", [])) == [31.4, 31.4], \
        f"31.4% AutomationBench fragment must align, got {by_key}"
    assert sorted(by_key.get("generic|percentage|score|bench=terminal-bench-science||fable-5-1", [])) == [52.6, 52.6], \
        f"52.6% Terminal-Bench-Science fragment must align, got {by_key}"


@test("POST /api/v1/analyze omits x_post_assessment for articles, keeps it for X posts (v1.8.8)")
def _():
    # Run 11: every analyze result carried "x_post_assessment": {} —
    # the live v1.8.6 response included the empty dict for article
    # genres. v1.8.7 omits the key unless the genre is x_news_post;
    # this pins that behaviour so it cannot regress silently.
    r = client.post("/api/v1/analyze", json={
        "text": ("Anthropic released Claude Fable 5.2 on 9 November 2026. "
                 "FrontierCode 2.0 scores 49.1%."),
        "source_type": "secondary",
        "source_name": "test-article",
        "source_url": "https://example.com/article",
    })
    d = r.get_json()
    assert d.get("source_genre") != "x_news_post"
    assert "x_post_assessment" not in d, \
        f"article genres must omit x_post_assessment, got {d.get('x_post_assessment')}"
    r = client.post("/api/v1/analyze", json={
        "text": ("BREAKING: Officials confirm the bridge is closed after a fire. "
                 "2026-06-19 09:42 PDT statement linked from the city. "
                 "Official update: https://city.example.gov/bridge-fire-update"),
        "source_url": "https://x.com/example/status/1234567890",
        "source_type": "auto",
    })
    d = r.get_json()
    assert d.get("source_genre") == "x_news_post"
    assert "x_post_assessment" in d and d["x_post_assessment"], \
        "x_news_post must carry a non-empty x_post_assessment"


@test("POST /api/v1/compare mints a split-as block figure (v1.8.8)")
def _():
    # Run 11 medium: '"about 55% fewer blocks", split as "about 60%
    # fewer in cybersecurity..."' — the 60% never minted because the
    # "blocks" cue belongs to the 55% and no comparison marker joined
    # them. "split as" is now a comparison marker, so the 60% inherits
    # the blocks family — and the compare unifies the medium's unstated
    # subject with the primary's fable-5-1, so it aligns 2-way.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ('Anthropic adds "up to 30% faster" output, "about 55% fewer blocks", '
                      'split as "about 60% fewer in cybersecurity and 40% fewer in biology".'),
             "source_name": "medium"},
            {"text": ("Based on early testing, Anthropic observed about 55% fewer blocks with "
                      "Fable 5.2 compared with Fable 5.1 — about 60% fewer in cybersecurity "
                      "and 40% fewer in biology."),
             "source_name": "primary"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"block figures must never conflict, got {d.get('conflicts')}"
    by_key = {a["entity_key"]: [v["value"] for v in a.get("values", [])]
              for a in d.get("alignment", [])}
    # v1.8.8: "60% fewer" is a relative delta, so the key carries the
    # "rel" marker — the figure still aligns 2-way as a blocks claim.
    # v1.8.9 correction (Fix 3): the 60% is Fable 5.2's — the measured
    # thing — not Fable 5.1's; Fable 5.1 is the comparison baseline
    # ("55% fewer blocks with Fable 5.2 compared with Fable 5.1").
    # The old key fable-5-1 encoded the baseline-as-subject bug.
    assert sorted(by_key.get("generic|percentage|blocks|rel||fable-5-2", [])) == [60.0, 60.0], \
        f"medium's 60% must align as a blocks claim, got {by_key}"


@test("POST /api/v1/compare splits a sequenced price change instead of conflicting (v1.8.8)")
def _():
    # Nov 11 live failure: "$2 vs $4 vs $0.10" and "$10 vs $20" read as
    # conflicts — a sequenced price change ("$2 through Nov. 30, then
    # $4 starting Dec. 1") is two identities, not a dispute, and a
    # cached price ("Cached input: $0.10") is its own tier.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("Carbon costs $2 per million input tokens and $10 per million "
                      "output tokens through November 30, 2026, then $4 and $20 "
                      "starting December 1."),
             "source_name": "trendy"},
            {"text": ("Carbon costs $2 per million input tokens and $10 per million "
                      "output tokens."),
             "source_name": "primary"},
            {"text": ("- Cached input: $0.10 through November 30, 2026; $0.20 starting "
                      "December 1, 2026."),
             "source_name": "price-list"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"sequenced prices must never conflict, got {d.get('conflicts')}"
    aligned = {c["entity_key"]: [v["value"] for v in c.get("values", [])]
               for c in d.get("alignment", [])}
    assert sorted(aligned.get("pricing-input", [])) == [2.0, 2.0], \
        f"current input must align at 2, got {aligned}"
    assert sorted(aligned.get("pricing-output", [])) == [10.0, 10.0], \
        f"current output must align at 10, got {aligned}"
    nc = {n["entity_key"] for n in d.get("not_comparable", [])}
    assert "pricing-input" in nc and "pricing-output" in nc, \
        f"split pricing keys must be recorded not_comparable, got {nc}"
    cached = [c for c in d.get("unique_claims", [])
              if c.get("entity_key") == "pricing-cached"]
    assert cached and cached[0]["value"] == 0.1, \
        f"$0.10 must surface as a unique pricing-cached claim, got {cached}"


@test("POST /api/v1/compare treats an undated price as not comparable with a dated one (v1.8.8 repair)")
def _():
    # Arch audit 2026-11-13, fresh counterexample in the actual
    # candidate, sentences verbatim: A states "$2 today" (current)
    # and B states "$4 per million tokens" with no date at all. An
    # undated price must not contradict a dated one — and must not
    # silently align with one either. The undated era is "unknown"
    # and partitions apart.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("Carbon input costs $2 today, the price will "
                      "increase to $4 on December 1."),
             "source_name": "A"},
            {"text": "Carbon input costs $4 per million tokens.",
             "source_name": "B"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"undated vs dated must never conflict, got {d.get('conflicts')}"
    nc = [n for n in d.get("not_comparable", [])
          if n.get("entity_key") == "pricing-input"]
    assert nc, f"undated vs dated input must be recorded not_comparable, got {d.get('not_comparable')}"
    eras = {i["era"]: [v["value"] for v in i["values"]] for i in nc[0]["identities"]}
    assert eras.get("current") == [2.0] and eras.get("unknown") == [4.0], \
        f"identities must be current $2 vs unknown $4, got {eras}"
    # Source spans preserved: every value carries the excerpt it came from.
    cur = [i for i in nc[0]["identities"] if i["era"] == "current"][0]
    assert "$2 today" in cur["values"][0].get("context", ""), \
        f"current $2 span must show its sentence, got {cur['values']}"
    for ident in nc[0]["identities"]:
        for v in ident["values"]:
            assert v.get("context"), f"not_comparable value must carry its source span, got {v}"
    assert not [a for a in d.get("alignment", [])
                if a.get("entity_key") == "pricing-input"], \
        f"dated $2 and undated $4 must not align, got {d.get('alignment')}"


@test("POST /api/v1/compare does not silently align a dated price with an undated one of the same value (v1.8.8 repair)")
def _():
    # Same figure, different temporal identity: "$3 today" and an
    # undated "$3" are not the same claim. They must split into a
    # not_comparable record, never an alignment.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Carbon input costs $3 per million tokens today.",
             "source_name": "A"},
            {"text": "Carbon input costs $3 per million tokens.",
             "source_name": "B"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"same-value dated vs undated must never conflict, got {d.get('conflicts')}"
    nc = [n for n in d.get("not_comparable", [])
          if n.get("entity_key") == "pricing-input"]
    assert nc, f"same-value dated vs undated must be recorded not_comparable, got {d.get('not_comparable')}"
    eras = {i["era"] for i in nc[0]["identities"]}
    assert eras == {"current", "unknown"}, f"identities must be current vs unknown, got {eras}"
    assert not [a for a in d.get("alignment", [])
                if a.get("entity_key") == "pricing-input"], \
        f"dated and undated $3 must not align, got {d.get('alignment')}"


@test("POST /api/v1/compare still compares two undated prices (v1.8.8 repair)")
def _():
    # The fail-closed rule is dated-vs-undated only. Two reports that
    # both state no era still compare with each other — that is the
    # November 11 replay's behaviour, where all four $2/$10 reports
    # state no era and align.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Carbon input costs $2 per million tokens.",
             "source_name": "A"},
            {"text": "Carbon input costs $3 per million tokens.",
             "source_name": "B"},
        ],
    })
    d = r.get_json()
    assert len(d.get("conflicts", [])) == 1, \
        f"two undated prices that disagree must still conflict, got {d.get('conflicts')}"
    assert d["conflicts"][0]["entity_key"] == "pricing-input", \
        f"conflict must be on pricing-input, got {d.get('conflicts')}"
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Carbon input costs $2 per million tokens.",
             "source_name": "A"},
            {"text": "Carbon input costs $2 per million tokens.",
             "source_name": "B"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"two undated $2 must not conflict, got {d.get('conflicts')}"
    aligned = {c["entity_key"]: [v["value"] for v in c.get("values", [])]
               for c in d.get("alignment", [])}
    assert sorted(aligned.get("pricing-input", [])) == [2.0, 2.0], \
        f"two undated $2 must align, got {aligned}"


@test("POST /api/v1/compare still contradicts two dated prices that disagree (v1.8.8 repair)")
def _():
    # Two prices that both state they are current and disagree ARE a
    # contradiction. The repair must not turn every price dispute
    # into a shrug.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Carbon input costs $2 per million tokens today.",
             "source_name": "A"},
            {"text": "Carbon input costs $3 per million tokens now.",
             "source_name": "B"},
        ],
    })
    d = r.get_json()
    assert len(d.get("conflicts", [])) == 1, \
        f"two dated prices that disagree must still conflict, got {d.get('conflicts')}"
    assert d["conflicts"][0]["entity_key"] == "pricing-input", \
        f"conflict must be on pricing-input, got {d.get('conflicts')}"


@test("POST /api/v1/compare reads current markers in equivalent wording (v1.8.8 repair)")
def _():
    # Held-out phrasing check, not the exact audit sentences: the
    # same dated-vs-undated split must hold when the current marker
    # is "currently" before the amount, or "at present" just after
    # it. A marker in an earlier clause must not leak onto a later
    # price in the same sentence.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Carbon input currently costs $5 per million tokens.",
             "source_name": "A"},
            {"text": "Carbon input costs $5 per million tokens.",
             "source_name": "B"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"'currently' vs undated must never conflict, got {d.get('conflicts')}"
    nc = [n for n in d.get("not_comparable", [])
          if n.get("entity_key") == "pricing-input"]
    assert nc and {i["era"] for i in nc[0]["identities"]} == {"current", "unknown"}, \
        f"'currently' must read as current vs unknown, got {d.get('not_comparable')}"
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Carbon input costs $2 per million tokens at present.",
             "source_name": "A"},
            {"text": "Carbon input costs $2 per million tokens.",
             "source_name": "B"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"'at present' vs undated must never conflict, got {d.get('conflicts')}"
    nc = [n for n in d.get("not_comparable", [])
          if n.get("entity_key") == "pricing-input"]
    assert nc and {i["era"] for i in nc[0]["identities"]} == {"current", "unknown"}, \
        f"'at present' must read as current vs unknown, got {d.get('not_comparable')}"
    assert not [a for a in d.get("alignment", [])
                if a.get("entity_key") == "pricing-input"], \
        f"dated and undated $2 must not align, got {d.get('alignment')}"


@test("POST /api/v1/compare keeps per-task prices apart by thinking level and model (v1.8.8)")
def _():
    # Nov 11 live failure: "$7.76 vs $6.59 vs $9.95" read as a
    # conflict — three different qualified identities: Carbon at high
    # thinking, Carbon at medium, and Argon.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("Artificial Analysis reports that running its Intelligence Index "
                      "cost $7.76 per task on Carbon at high thinking, $6.59 at medium."),
             "source_name": "trendy"},
            {"text": ("AA measurement 60.0 (high), second, $7.76 per task. 9.6 points "
                      "above Argon (high) at 50.4, at about 22% less per task "
                      "(Argon: $9.95)."),
             "source_name": "sw"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"per-task prices must never conflict, got {d.get('conflicts')}"
    aligned = {c["entity_key"]: [v["value"] for v in c.get("values", [])]
               for c in d.get("alignment", [])}
    assert sorted(aligned.get("pricing-per-task", [])) == [7.76, 7.76], \
        f"Carbon at high thinking must align at 7.76, got {aligned}"
    nc = {n["entity_key"] for n in d.get("not_comparable", [])}
    assert "pricing-per-task" in nc, \
        f"split per-task key must be recorded not_comparable, got {nc}"


@test("POST /api/v1/compare keeps a relative delta apart from an absolute rate, and a plan price apart from a per-unit price (v1.8.8)")
def _():
    # Nov 11 live failure: "92% vs 22%" and "$19.99 vs $8.00" read as
    # conflicts — a relative delta ("22% less per task") is not an
    # absolute pass rate ("92% of tasks"), and a subscription-plan
    # price ("the $19.99 plan") is not a per-unit API price ("$8 input").
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": ("New results show that on our 375-task agentic honesty benchmark, "
                      "it passed 92% of tasks. Google AI Pro, the $19.99 plan, is not "
                      "named in the launch post."),
             "source_name": "primary"},
            {"text": ("At 64.6 on AA's Intelligence Index v5 it passes Gemini 4 Carbon "
                      "(60.0) for first place, and at $8 input and $40 output it costs "
                      "20% less than Fable 5.1. 9.6 points above Argon (high) at 50.4, "
                      "at about 22% less per task (Argon: $9.95)."),
             "source_name": "sw"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"generic figures must never conflict, got {d.get('conflicts')}"
    keys = {c["entity_key"] for c in d.get("unique_claims", [])}
    assert "generic|percentage|tasks|rel||unstated" in keys, \
        f"22% less must mint as a relative tasks claim, got {keys}"
    assert "generic|cost|cost|plan||unstated" in keys, \
        f"$19.99 plan must mint as a plan cost claim, got {keys}"


@test("POST /api/v1/analyze takes the publisher from the feed declaration, never the body (v1.8.8)")
def _():
    # Nov 11 live failure: SQ Magazine -> "cloud", TrendyTechTribe ->
    # "antigravity", SW Architect -> "aa" — words found in the body
    # became the claimed publisher. The publisher is the feed's own
    # declaration (source_name, then the feed header); a body mention
    # is recorded as mentioned, never as the publisher.
    r = client.post("/api/v1/analyze", json={
        "text": ("Google Antigravity's coding tool is included for Ultra subscribers. "
                 "The new model scored 49.1% on FrontierCode 2.0."),
        "source_type": "secondary",
        "source_name": "SQ Magazine — Google Unleashes Gemini 4 Carbon, Its Most Powerful AI Model",
        "source_url": "https://sqmagazine.co.uk/google-releases-gemini-4-carbon/",
    })
    d = r.get_json()
    cb = d["custody_breakdown"]
    assert cb["claimed_publisher_org"] == "sq magazine", \
        f"publisher must be sq magazine, got {cb['claimed_publisher_org']}"
    assert cb["claimed_publisher_org_reason"] == "feed-declared publisher (source name)", \
        f"reason must name the source name, got {cb['claimed_publisher_org_reason']}"
    assert cb["feed_declared_publisher"] == "sq magazine"
    assert "antigravity" in cb["organizations_mentioned"], \
        f"antigravity must be recorded as mentioned, got {cb['organizations_mentioned']}"
    # No source_name and no feed header: the publisher is UNKNOWN,
    # with a reason — never a body word.
    r = client.post("/api/v1/analyze", json={
        "text": ("Antigravity Team announced a coding tool. AA's top spot "
                 "went to Fable 5.2 at 64.6. The new model scored 49.1% on "
                 "FrontierCode 2.0."),
        "source_type": "secondary",
    })
    d = r.get_json()
    cb = d["custody_breakdown"]
    assert cb["claimed_publisher_org"] is None, \
        f"publisher must be unknown without a declaration, got {cb['claimed_publisher_org']}"
    assert cb["claimed_publisher_org_reason"] == \
        "no publisher declared in feed header or source name", \
        f"reason must say no publisher was declared, got {cb['claimed_publisher_org_reason']}"
    assert "antigravity" in cb["organizations_mentioned"]
    assert "aa" in cb["organizations_mentioned"]


@test("POST /api/v1/analyze keeps a benchmark name out of the data providers (v1.8.8)")
def _():
    # Nov 11 live failure: "SEC-bench Pro" was read as the US SEC
    # regulator and named as a data provider on 3 of 4 docs.
    r = client.post("/api/v1/analyze", json={
        "text": ("On SEC-bench Pro, the model scored 58.8%. The benchmark "
                 "is run by an independent lab and the results were "
                 "published this week."),
        "source_type": "secondary",
        "source_name": "bench-note",
    })
    d = r.get_json()
    providers = d["custody_breakdown"]["named_data_providers"]
    assert "SEC" not in providers, \
        f"SEC-bench must never name the SEC as a data provider, got {providers}"
    r = client.post("/api/v1/analyze", json={
        "text": ("The company filed its quarterly report with the SEC. "
                 "The filing discloses revenue of $1.2 billion."),
        "source_type": "secondary",
        "source_name": "filing-note",
    })
    d = r.get_json()
    providers = d["custody_breakdown"]["named_data_providers"]
    assert "SEC" in providers, \
        f"a real SEC filing reference must still name the SEC, got {providers}"


# ── v1.8.8 r2: run-17 gaps + bracketed price-list tiering leak ──────────────

@test("POST /api/v1/compare tiers a bracketed price list without leaking (v1.8.8 r2)")
def _():
    # Held-out tiering leak: "[$4 in · $0.2 cached · $20 out / 1M]"
    # mis-tiered the $20 output price as pricing-cached (a false
    # conflict). Each amount now takes the tier word in its own "·"
    # segment, including the "in"/"out" abbreviations, and a tier word
    # that directly trails a $ amount belongs to that amount.
    row = "Pricing: [$4 in · $0.2 cached · $20 out / 1M] for the new model."
    claims = extract_numeric_claims(row)
    tiers = {(c["entity_key"], c["numeric_value"]) for c in claims}
    assert ("pricing-input", 4.0) in tiers, f"$4 must be input, got {tiers}"
    assert ("pricing-cached", 0.2) in tiers, f"$0.2 must be cached, got {tiers}"
    assert ("pricing-output", 20.0) in tiers, f"$20 must be output, got {tiers}"
    assert not any(k == "pricing-cached" and v == 20.0 for k, v in tiers), \
        "$20 must never be cached"
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": row, "source_name": "A"},
            {"text": row, "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"bracketed list must not conflict, got {d.get('conflicts')}"
    aligned = {c["entity_key"]: [v["value"] for v in c["values"]] for c in d.get("alignment", [])}
    assert aligned.get("pricing-input") == [4.0, 4.0], f"input must align, got {aligned}"
    assert aligned.get("pricing-cached") == [0.2, 0.2], f"cached must align, got {aligned}"
    assert aligned.get("pricing-output") == [20.0, 20.0], f"output must align, got {aligned}"


@test("POST /api/v1/compare does not mint a benchmark name as a number (v1.8.8 r2)")
def _():
    # Run-17 gap 3: "Work360 benchmark score: Qwen4-Max 52.69" minted
    # "360" as a task-count claim (twice, across entities) and the real
    # 52.69 never minted. A digit run glued to a preceding letter is
    # part of an identifier, not a value.
    claims = extract_generic_quantitative_claims(
        "Work360 benchmark score: Qwen4-Max 52.69.")
    vals = [c["numeric_value"] for c in claims]
    assert 360.0 not in vals, f"Work360 must never mint 360, got {vals}"
    assert 52.69 in vals, f"the real score 52.69 must mint, got {vals}"


@test("POST /api/v1/compare mints a glued percentage and an index-points score (v1.8.8 r2)")
def _():
    # Run-17 gaps 4/6: "89.0%with tools" never minted (the % suffix
    # plus a following letter skipped it), and "Artificial Analysis
    # gives it 43 points" never minted ("points" was not a score cue).
    claims = extract_generic_quantitative_claims(
        "Chartography 89.0%with tools and OSWorld 81.8%partial scores.")
    keys = {(c["entity_key"], c["numeric_value"]) for c in claims}
    assert ("generic|percentage|score|bench=chartography||unstated", 89.0) in keys, \
        f"Chartography 89.0% must mint, got {keys}"
    assert ("generic|percentage|score|bench=osworld||unstated", 81.8) in keys, \
        f"OSWorld 81.8% must mint, got {keys}"
    claims = extract_generic_quantitative_claims(
        "Artificial Analysis gives it 43 points, nine more than its predecessor.")
    keys = {(c["entity_key"], c["numeric_value"]) for c in claims}
    assert ("generic|task_count|score|bench=artificial-analysis||unstated", 43.0) in keys, \
        f"AA Index 43 points must mint, got {keys}"


@test("POST /api/v1/compare reads a slash date into the shared-date set (v1.8.8 r2)")
def _():
    # Run-17 gap 11: the primary's "2026/11/15" never minted, so it
    # never joined the shared-date set.
    assert [c["date_key"] for c in extract_date_claims("Dated 2026/11/15.")] == ["2026-11-15"], \
        "slash date must mint as 2026-11-15"
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "The post is dated 2026/11/15 and announces the model.", "source_name": "A"},
            {"text": "Published November 15, 2026: the model is out.", "source_name": "B"},
        ]
    })
    d = r.get_json()
    shared_keys = [c.get("date_key") for c in (d.get("shared_dates") or [])]
    assert "2026-11-15" in shared_keys, \
        f"slash date must join the shared dates, got {d.get('shared_dates')}"


@test("POST /api/v1/compare does not mint a feed-header line as a claim (v1.8.8 r2)")
def _():
    # Run-17 gap 8: the InfoWorld "Source: ..." header line minted its
    # "62%" as a bench=infoworld claim (double-counted with the body).
    claims = extract_generic_quantitative_claims(
        "Source: InfoWorld — 'Qwen4-Max review' 62% of tasks.\n"
        "The model scores 62% of tasks.")
    keys = {(c["entity_key"], c["numeric_value"]) for c in claims}
    assert not any("bench=infoworld" in k for k, _v in keys), \
        f"header line must never mint a claim, got {keys}"
    assert any(v == 62.0 for _k, v in keys) and len(keys) == 1, \
        f"the body 62% must still mint exactly once, got {keys}"


@test("POST /api/v1/compare mints a bare-number table-row score (v1.8.8 r2)")
def _():
    # Run-17 gap 4: "- DeepSWE v1.1 (agentic coding): 75.7" minted
    # nothing — the row parser only read "Model Version %" pairs.
    # A bare number after the row's colon IS the row's score.
    row = "- DeepSWE v1.1 (agentic coding): 75.7"
    claims = extract_generic_quantitative_claims(row)
    keys = {(c["entity_key"], c["numeric_value"]) for c in claims}
    assert ("generic|measurement|score|bench=deepswe||unstated", 75.7) in keys, \
        f"bare row score must mint, got {keys}"
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": row, "source_name": "A"},
            {"text": row, "source_name": "B"},
        ]
    })
    d = r.get_json()
    assert not d.get("conflicts"), f"identical bare rows must not conflict, got {d.get('conflicts')}"
    aligned = [c["entity_key"] for c in d.get("alignment", [])]
    assert "generic|measurement|score|bench=deepswe||unstated" in aligned, \
        f"bare row score must align, got {aligned}"


@test("POST /api/v1/analyze keeps a declared primary on its own domain (v1.8.8 r2)")
def _():
    # Run-17 gap 2: the Qwen primary feed — a third-person summary of
    # the vendor's own blog post, hosted on qwen.ai — was downgraded
    # three runs in a row for lack of first-person announcement voice.
    # The feed-declared publisher matching the source URL's own domain
    # corroborates the declaration; the summary's voice is the
    # pipeline's, not the source's. Same vetoes as the announcement
    # exemption: a third-party attribution cue still voids it.
    text = ("Source: Qwen Team — 'Qwen4-Max: Your New Teammate Has Arrived' "
            "(Qwen Blog, dated 2026/11/15). The post introduces Qwen4-Max, "
            "a new flagship model with stronger performance across nearly "
            "every capability evaluated.")
    r = client.post("/api/v1/analyze", json={
        "text": text,
        "source_type": "primary",
        "source_name": "Qwen Team — Qwen4-Max",
        "source_url": "https://qwen.ai/blog/qwen4-max/",
    })
    d = r.get_json()
    assert d["source_type"] == "primary", \
        f"own-domain primary must keep its type, got {d['source_type']}"
    assert not any("downgraded" in f for f in d.get("flags", [])), \
        f"no downgrade flag allowed, got {d.get('flags')}"
    # Negative: the same declaration on a third-party host that still
    # reads as a news article (a /blog/ URL) still downgrades.
    r = client.post("/api/v1/analyze", json={
        "text": text,
        "source_type": "primary",
        "source_name": "Qwen Team — Qwen4-Max",
        "source_url": "https://example.com/blog/qwen4-max/",
    })
    d = r.get_json()
    assert d["source_type"] != "primary", \
        f"third-party host must not keep the primary type, got {d['source_type']}"


@test("POST /api/v1/analyze does not misread a host or a mention as a type (v1.8.8 r2)")
def _():
    # Run-17 gap 9: "x.com" matched inside "wixx.com" and called a
    # Reuters wire 'social'; a negated body mention ("not from the
    # official press release") called an analysis 'official'.
    assert _infer_source_type_from_signals(
        "Alibaba unveiled a new flagship model.", "https://wixx.com/2026/11/15/x/") != "social", \
        "wixx.com must not read as x.com"
    body = "x " * 400 + ("not from the official press release, and the "
                         "announcement does not mention it either.")
    assert _infer_source_type_from_signals(
        body, "https://www.askthisguy.com/en/blog/x/") != "official", \
        "a body mention of a press release must not infer official"
    # Positive controls: a real social host and a real press-release
    # header still fire.
    assert _infer_source_type_from_signals(
        "Breaking news.", "https://x.com/someone/status/1") == "social"
    assert _infer_source_type_from_signals(
        "PRESS RELEASE: Acme announces a new model today.",
        "https://example.com/news/") == "official"


@test("POST /api/v1/analyze a benchmark named after a number does not own it (v1.8.8 r2)")
def _():
    # Run-17 gap 3, follow-on: the sentence-level single-benchmark
    # fallback used the WHOLE sentence, so a SWE-bench score was
    # mislabeled Terminal-Bench ("scores 77.2 against 76.2; on
    # Terminal-Bench 2.0, 59.3") and a scale marker ("out of 100")
    # was mislabeled OSWorld. Only benchmarks named BEFORE the
    # number count. (The 59.3/52.5 themselves stay unminted in both
    # builds — bare numbers in a cue-less clause — a known residual,
    # not a regression.)
    claims = extract_generic_quantitative_claims(
        "On SWE-bench Verified, the dense model scores 77.2 against "
        "76.2; on Terminal-Bench 2.0, 59.3 against 52.5.")
    by_val = {c["numeric_value"]: c["entity_key"] for c in claims}
    assert by_val.get(76.2, "").find("terminal-bench") == -1, \
        "the SWE-bench 76.2 must not carry a Terminal-Bench key"
    assert 59.3 not in by_val and 52.5 not in by_val, \
        "the bare Terminal-Bench figures stay unminted (known residual)"
    # The documented use case still works: the benchmark sits
    # before the number.
    claims = extract_generic_quantitative_claims(
        "In its internal CVE Rediscovery evaluation, the model "
        "scored 78.3%.")
    by_val = {c["numeric_value"]: c["entity_key"] for c in claims}
    assert "cve" in by_val.get(78.3, "").lower() or \
        "rediscovery" in by_val.get(78.3, "").lower(), \
        "a benchmark named before the number must still attach"


@test("POST /api/v1/analyze a benchmark lead-in owns its clause's percentages (v1.8.8 r2)")
def _():
    # Run-17 gap 4, follow-on: "- On Last Secure Code Bench,
    # function-level pass rate rose from 22.0% to 31.8%" minted
    # nothing — the benchmark sat in the preceding clause, and the
    # clause-scoped score rule never saw it. A lead-in (On/For/In +
    # benchmark, no verb) owns the percentages in the clause it
    # introduces. The rule is deliberately one clause deep: the
    # all-three-granularities figures later in the same sentence
    # (8%, 14.7%) stay unminted — a lead-in that owned the whole
    # sentence would also own unrelated later metrics ("On HLE,
    # scored 51.7%, and improved 12% on efficiency").
    claims = extract_generic_quantitative_claims(
        "- On Last Secure Code Bench, function-level pass rate rose "
        "from 22.0% to 31.8%, and the rate of passing all three "
        "granularities rose from 8% to 14.7%.")
    by_val = {c["numeric_value"]: c["entity_key"] for c in claims}
    for v in (22.0, 31.8):
        assert "last-secure-code-bench" in by_val.get(v, ""), \
            f"{v} must mint as a Last Secure Code Bench score"
    assert 8.0 not in by_val and 14.7 not in by_val, \
        "the two-clauses-back figures stay unminted by design"
    # Guard: a complete earlier statement never owns a later
    # percentage.
    claims = extract_generic_quantitative_claims(
        "HLE was 51.7%. The model also improved 12% on efficiency.")
    assert not any(c["numeric_value"] == 12.0 and
                   "hle" in c["entity_key"] for c in claims), \
        "a complete HLE statement must not own the later 12%"


@test("POST /api/v1/compare a bench-less score never joins a benchmark its sentence never names (v1.8.8 r2)")
def _():
    # Run-17 gap 3, follow-on: the merge attached the bench-less
    # SWE-bench 76.2 to the nearest benched group by value alone —
    # DeepSWE 75.7 (a false alignment; before the extraction fix
    # it was a false Terminal-Bench conflict). Two halves: the
    # phrase list now names SWE-bench at extraction, and the merge
    # refuses an attach when the claim's sentence names other
    # benchmarks but not the target.
    claims = extract_generic_quantitative_claims(
        "On SWE-bench Verified, the dense model scores 77.2 against "
        "76.2; on Terminal-Bench 2.0, 59.3 against 52.5.")
    by_val = {c["numeric_value"]: c["entity_key"] for c in claims}
    assert "swe-bench" in by_val.get(76.2, ""), \
        "the SWE-bench 76.2 must carry its own bench, not another's"
    assert "swe-bench" in by_val.get(77.2, ""), \
        "the SWE-bench 77.2 must carry its own bench"
    # Merge guard, direct: a bench-less score whose span names
    # Terminal-Bench must not attach to a DeepSWE group.
    from signal_compare import _span_bench_slugs
    slugs = _span_bench_slugs(
        "On SWE-bench Verified, the dense model scores 77.2 against "
        "76.2; on Terminal-Bench 2.0, 59.3 against 52.5.")
    assert "deepswe" not in slugs and "terminal-bench" in slugs, \
        "the span names Terminal-Bench, never DeepSWE"
    assert not _span_bench_slugs("The model scored 78.3%."), \
        "a sentence naming no benchmark yields an empty set"


# ── v1.8.9: footer sponsorship disclosure (run-20 gap 11b) ────────────────

_SPONSOR_BODY = (
    "The new model scores 61 on the benchmark and completes most tasks "
    "in under 20 minutes. Reviewers praised its speed and its lower cost "
    "per task, and noted the long context window. The release includes "
    "new safety evaluations and a public changelog. Analysts expect the "
    "model to compete with other frontier systems on agentic tasks. "
    "Early benchmarks show strong results on coding and research. "
    "The company says more capability reports will follow next month. "
    "Independent testers confirmed the headline scores within a few points. "
    "Pricing starts at two dollars per million input tokens. "
)


@test("POST /api/v1/analyze records a footer sponsorship disclosure (v1.8.9)")
def _():
    r = client.post("/api/v1/analyze", json={
        "text": _SPONSOR_BODY + " Sponsored by Acme Corp. All rights reserved."})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    disc = d.get("sponsorship_disclosure")
    assert disc, "a footer 'Sponsored by' line must be recorded"
    assert disc["present"] is True
    assert disc["kind"] == "sponsored"
    assert disc["org"] == "Acme Corp", f"org: {disc['org']!r}"
    assert "Sponsored by Acme Corp" in disc["snippet"]
    assert any("Sponsorship disclosure" in f for f in d["flags"]), \
        "the disclosure must surface as a flag"
    assert "Sponsorship disclosure" in d["triage_summary"], \
        "the disclosure must ride the triage summary"


@test("POST /api/v1/analyze records a footer paid-partnership disclosure (v1.8.9)")
def _():
    r = client.post("/api/v1/analyze", json={
        "text": _SPONSOR_BODY + " Paid partnership with Beta Ltd."})
    d = r.get_json()
    disc = d.get("sponsorship_disclosure")
    assert disc and disc["present"] is True
    assert disc["kind"] == "paid_partnership"
    assert disc["org"] == "Beta Ltd"


@test("POST /api/v1/analyze records a footer affiliate disclosure (v1.8.9)")
def _():
    r = client.post("/api/v1/analyze", json={
        "text": _SPONSOR_BODY + " This post contains affiliate links. "
        "We may earn a commission on purchases."})
    d = r.get_json()
    disc = d.get("sponsorship_disclosure")
    assert disc and disc["present"] is True
    assert disc["kind"] == "affiliate"


@test("POST /api/v1/analyze does not record a mid-text sponsored mention as a footer disclosure (v1.8.9)")
def _():
    # "sponsored" in the body is not a footer disclosure — only the
    # footer zone counts. The body here mentions it up front; the
    # footer is clean.
    r = client.post("/api/v1/analyze", json={
        "text": "This sponsored segment discusses the model. " + _SPONSOR_BODY
        + " All rights reserved. Contact the editor."})
    d = r.get_json()
    assert "sponsorship_disclosure" not in d, \
        "a body mention must not be recorded as a footer disclosure"


@test("POST /api/v1/analyze omits sponsorship_disclosure for a clean text (v1.8.9)")
def _():
    r = client.post("/api/v1/analyze", json={"text": _SPONSOR_BODY})
    d = r.get_json()
    assert "sponsorship_disclosure" not in d


# ── v1.8.9 follow-on: footer bias bump for paid_partnership / funded ──

@test("POST /api/v1/analyze gives the +0.12 bias bump for a footer paid-partnership disclosure (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/analyze", json={
        "text": _SPONSOR_BODY + " Paid partnership with Beta Ltd."})
    d = r.get_json()
    disc = d.get("sponsorship_disclosure")
    assert disc and disc["kind"] == "paid_partnership"
    assert d["scores"]["bias_risk"] == 0.19, \
        "0.07 (bias word 'paid partnership') + 0.12 (footer bump)"


@test("POST /api/v1/analyze gives the +0.12 bias bump for a footer funded-by disclosure (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/analyze", json={
        "text": _SPONSOR_BODY + " Funded by the Delta Fund."})
    d = r.get_json()
    disc = d.get("sponsorship_disclosure")
    assert disc and disc["kind"] == "funded"
    assert d["scores"]["bias_risk"] == 0.12, \
        "'funded by' is not a bias word, so the bump alone is 0.12"


@test("POST /api/v1/analyze gives no bias bump for an unrelated footer (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/analyze", json={
        "text": _SPONSOR_BODY + " All rights reserved. Contact the editor."})
    d = r.get_json()
    assert "sponsorship_disclosure" not in d
    assert d["scores"]["bias_risk"] == 0.0


@test("POST /api/v1/analyze gives no bias bump for a body paid-partnership mention (v1.8.9 follow-on)")
def _():
    # "paid partnership" in the BODY is ordinary reporting about
    # someone else's deal, not this piece disclosing its own
    # financial involvement — the bias word fires (0.07) but the
    # footer-gated bump must not.
    r = client.post("/api/v1/analyze", json={
        "text": "The paid partnership between Acme and Beta was announced. "
        + _SPONSOR_BODY + " All rights reserved."})
    d = r.get_json()
    assert "sponsorship_disclosure" not in d
    assert d["scores"]["bias_risk"] == 0.07, \
        "bias word only — no footer disclosure, no bump"


@test("POST /api/v1/analyze gives no bias bump for a body funded-by mention (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/analyze", json={
        "text": "The study was funded by the National Science Foundation. "
        + _SPONSOR_BODY + " All rights reserved."})
    d = r.get_json()
    assert "sponsorship_disclosure" not in d
    assert d["scores"]["bias_risk"] == 0.0


@test("POST /api/v1/analyze gives no bias bump for an unpaid partnership footer (v1.8.9 follow-on)")
def _():
    # "in partnership with" is editorial collaboration, not payment —
    # Taylor's yes covered paid partnership + funded by only.
    r = client.post("/api/v1/analyze", json={
        "text": _SPONSOR_BODY + " Produced in partnership with Gamma Org."})
    d = r.get_json()
    disc = d.get("sponsorship_disclosure")
    assert disc and disc["kind"] == "partnership", \
        "the disclosure is still recorded"
    assert d["scores"]["bias_risk"] == 0.0, \
        "an unpaid partnership earns no bump"


@test("POST /api/v1/analyze gives no bias bump for an affiliate footer (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/analyze", json={
        "text": _SPONSOR_BODY + " This post contains affiliate links. "
        "We may earn a commission on purchases."})
    d = r.get_json()
    disc = d.get("sponsorship_disclosure")
    assert disc and disc["kind"] == "affiliate"
    assert d["scores"]["bias_risk"] == 0.07, \
        "bias word 'affiliate' only — no +0.12 bump for affiliate kind"


# ── v1.8.9: multiplier / speed-ratio, separability, revision (run-20 gaps 11/14/10) ──

@test("POST /api/v1/compare checks a claimed multiplier against measured durations (v1.8.9)")
def _():
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Google also said Neon completes most tasks about three "
                 "times as quickly as Carbon at costs of about 60% lower.",
         "source_name": "SiliconANGLE"},
        {"text": "Gemini 4 Neon Extra High: 61, $6.44 per task, 21.5 minutes. "
                 "Gemini 4 Neon High: 60, $3.74 per task, 14.8 minutes. "
                 "Gemini 4 Carbon High: 60, $7.76 per task, 37.1 minutes.",
         "source_name": "Kingy"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    checks = d.get("speed_ratio_checks", [])
    assert len(checks) == 1, f"expected one speed-ratio check, got {checks}"
    check = checks[0]
    assert check["multiplier"] == 3.0
    assert check["multiplier_direction"] == "faster"
    assert check["verdict"] == "within_tolerance", \
        "only the matched high tier determines the headline check"
    assert check["info_weight"] == "low" and check["signal_role"] == "bias", \
        "a headline multiplier is low-weight bias signal, not a fact"
    by_tier = {(p["subject_tier"], p["baseline_tier"]): p for p in check["pairings"]}
    assert by_tier[("extra-high", "high")]["verdict"] == "outside_tolerance"
    assert by_tier[("high", "high")]["verdict"] == "within_tolerance"
    assert not any(a["entity_key"].startswith("generic|multiplier")
                   for a in d.get("alignment", []) + d.get("conflicts", [])), \
        "a multiplier must never align or conflict as a fact"


@test("POST /api/v1/compare flags ranks inside a stated margin as not separable (v1.8.9)")
def _():
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Vals AI's November 18 results put Neon Extra High second at "
                 "70.87% on Vals Index v2.2, at about $7.40 per test. "
                 "Fable 5.2 scored 71.31%, Sonnet 5.5 scored 70.85% and Astra "
                 "scored 67.66%. Its published uncertainty for Neon is plus or "
                 "minus 0.94 percentage points, much wider than the gap to "
                 "either Fable or Sonnet.",
         "source_name": "Kingy"},
        {"text": "On Vals Index v2.2, Neon scored 70.9% and Fable 5.2 scored 71.3%.",
         "source_name": "Second"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    seps = [s for s in d.get("separability", []) if s["benchmark"] == "vals-index"]
    assert len(seps) == 1, f"expected one Vals separability entry, got {d.get('separability')}"
    entry = seps[0]
    assert entry["verdict"] == "mixed"
    neon = [s for s in entry["scores"]
            if s["subject"] == "gemini-4-neon" and s["value"] == 70.87]
    assert neon and neon[0]["rank"] == 2, "the stated rank rides on the score"
    pair = [p for p in entry["pairs"]
            if p["higher"]["subject"] == "fable-5-2"
            and p["lower"]["subject"] == "gemini-4-neon"
            and p["lower"]["value"] == 70.87]
    assert pair, "the Fable-vs-Neon pair must be present"
    assert pair[0]["gap"] == 0.44 and pair[0]["separable"] is False, \
        "a 0.44pp gap inside ±0.94pp is not separable"


@test("POST /api/v1/compare surfaces a board revision instead of aligning it (v1.8.9)")
def _():
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Neon ranks second on Arena's Agent Arena Index with a score "
                 "of 1336 across more than 4,200 agent sessions.",
         "source_name": "Primary"},
        {"text": "Arena's November 18 board moved Neon High to third overall "
                 "at 1,333, three points behind Opus 5.5 Max.",
         "source_name": "Kingy"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    notes = d.get("revision_notes", [])
    assert len(notes) == 1, f"expected one revision note, got {notes}"
    assert "1336" in notes[0]["note"] and "1333" in notes[0]["note"], \
        "the note must surface the revision, not hide it"
    assert not any(a["entity_key"].endswith("bench=agent-arena||gemini-4-neon")
                   for a in d.get("alignment", [])), \
        "a revision must never align as agreement"


# ── v1.8.9 follow-on: general headline-position rule ──

@test("POST /api/v1/compare routes headline figures to headline_weighting, never to facts (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Source: SiliconANGLE — \"Google's latest midtier model Gemini 4 "
                 "Neon is three times faster than Carbon and wins on some metrics\" "
                 "by Mike Wheatley (published Nov 18, 2026).\n\n"
                 "Google also said Neon completes most tasks about three times as "
                 "quickly as Carbon at costs of about 60% lower.",
         "source_name": "SiliconANGLE — Google's latest midtier model Gemini 4 "
                        "Neon is three times faster than Carbon and wins on some metrics"},
        {"text": "Gemini 4 Neon Extra High: 61, $6.44 per task, 21.5 minutes. "
                 "Gemini 4 Neon High: 60, $3.74 per task, 14.8 minutes. "
                 "Gemini 4 Carbon High: 60, $7.76 per task, 37.1 minutes.",
         "source_name": "Kingy"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    hw = d.get("headline_weighting", [])
    assert len(hw) == 1, f"expected one headline entry, got {hw}"
    assert "SiliconANGLE" in hw[0]["source"]
    assert hw[0]["headline"].startswith("Google's latest midtier model")
    figs = hw[0]["figures"]
    assert len(figs) == 1, f"expected one headline figure, got {figs}"
    fig = figs[0]
    assert fig["kind"] == "multiplier" and fig["value"] == 3.0
    assert fig["baseline"] == "gemini-4-carbon"
    assert fig["from_headline"] is True
    assert fig["info_weight"] == "low" and fig["signal_role"] == "bias"
    # The headline multiplier must not double the speed-ratio check:
    # body claim preferred, exactly one check.
    checks = d.get("speed_ratio_checks", [])
    assert len(checks) == 1, f"expected one speed-ratio check, got {checks}"
    assert checks[0]["verdict"] == "within_tolerance"
    assert checks[0]["from_headline"] is False, \
        "the body multiplier is the checked claim, not the headline one"
    assert not any(a["entity_key"].startswith("generic|multiplier")
                   or a["entity_key"].startswith("generic|headline")
                   for a in d.get("alignment", []) + d.get("conflicts", [])
                   + d.get("unique_claims", [])), \
        "a headline figure must never align, conflict, or count as a unique"
    assert "headline_weighting" in d["triage_summary"]


@test("POST /api/v1/compare records a headlined benchmark score without demoting the body claim (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Source: AlphaSignal — \"Google DeepMind's Gemini 4 Neon Targets "
                 "Complex Agent Workflows at 61 Intelligence Score\" (published "
                 "Nov 18, 2026).\n\n"
                 "Artificial Analysis, a third-party model evaluator, gives Neon an "
                 "Intelligence Index score of 61 at its highest xhigh thinking level.",
         "source_name": "AlphaSignal — Google DeepMind's Gemini 4 Neon Targets "
                        "Complex Agent Workflows at 61 Intelligence Score"},
        {"text": "Source: Google (The Keyword) — \"Introducing Gemini 4 Neon\" by "
                 "Koray Kavukcuoglu (published Nov 18, 2026).\n\n"
                 "Neon ranks ahead of comparable models and #3 overall with a score "
                 "of 61 on the Artificial Analysis Intelligence Index at its xhigh "
                 "thinking level.",
         "source_name": "Google — Introducing Gemini 4 Neon (Nov 18, 2026)"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    hw = d.get("headline_weighting", [])
    assert len(hw) == 1, f"expected one headline entry, got {hw}"
    assert "AlphaSignal" in hw[0]["source"]
    figs = hw[0]["figures"]
    assert len(figs) == 1, f"expected one headline figure, got {figs}"
    assert figs[0]["kind"] == "score" and figs[0]["value"] == 61.0
    assert figs[0]["subject"] == "gemini-4-neon"
    # The body 61 still aligns as a fact — the doc is not punished for
    # headlining its own lead fact.
    aligned_61 = [a for a in d.get("alignment", [])
                  if 61.0 in [v.get("value") for v in a.get("values", [])]]
    assert aligned_61, f"the body 61 must still align, got {d.get('alignment')}"
    assert not any(a["entity_key"].startswith("generic|headline")
                   for a in d.get("alignment", []) + d.get("conflicts", [])), \
        "a headline figure must never align or conflict as a fact"


@test("POST /api/v1/compare still checks a headline-only multiplier, tagged from_headline (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Source: SiliconANGLE — \"Neon is three times faster than Carbon\" "
                 "by Mike Wheatley (published Nov 18, 2026).\n\n"
                 "Neon launched on November 18, 2026. It costs $3 per million "
                 "input tokens.",
         "source_name": "SiliconANGLE — Neon is three times faster than Carbon"},
        {"text": "Gemini 4 Neon Extra High: 61, $6.44 per task, 21.5 minutes. "
                 "Gemini 4 Neon High: 60, $3.74 per task, 14.8 minutes. "
                 "Gemini 4 Carbon High: 60, $7.76 per task, 37.1 minutes.",
         "source_name": "Kingy"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    checks = d.get("speed_ratio_checks", [])
    assert len(checks) == 1, f"expected one speed-ratio check, got {checks}"
    assert checks[0]["from_headline"] is True, \
        "no body multiplier exists, so the headline one is the checked claim"
    assert checks[0]["verdict"] == "within_tolerance"
    hw = d.get("headline_weighting", [])
    assert len(hw) == 1 and hw[0]["figures"][0]["kind"] == "multiplier"


@test("POST /api/v1/compare detects the headline from source_name when the text carries no Source line (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Google also said Neon completes most tasks about three times as "
                 "quickly as Carbon at costs of about 60% lower.",
         "source_name": "SiliconANGLE — Neon is three times faster than Carbon"},
        {"text": "Gemini 4 Neon Extra High: 61, $6.44 per task, 21.5 minutes. "
                 "Gemini 4 Carbon High: 60, $7.76 per task, 37.1 minutes.",
         "source_name": "Kingy"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    hw = d.get("headline_weighting", [])
    assert len(hw) == 1, f"expected one headline entry, got {hw}"
    assert hw[0]["headline"] == "Neon is three times faster than Carbon"
    assert hw[0]["figures"][0]["kind"] == "multiplier"


@test("POST /api/v1/compare mints nothing from a Source line whose number has no benchmark cue (v1.8.9 follow-on, gap-8)")
def _():
    # Run-17 gap 8: a header number ("62%") minted as a claim. The
    # headline pass only reads numbers with a benchmark cue nearby,
    # and the Source line stays dropped as a sentence source — so a
    # bare header figure surfaces nowhere.
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Source: InfoWorld — \"62% of users prefer the new model\" by "
                 "X (published Nov 18, 2026).\n\n"
                 "The model launched today. It costs $3 per million input tokens.",
         "source_name": "InfoWorld — 62% of users prefer the new model"},
        {"text": "The model launched today. It costs $3 per million input tokens.",
         "source_name": "Second"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert d.get("headline_weighting", []) == [], \
        "a header number with no benchmark cue is not a headline figure"
    for entry in d.get("alignment", []) + d.get("conflicts", []):
        vals = [v.get("value") for v in entry.get("values", [])]
        assert 62.0 not in vals, \
            f"the Source-line 62 must mint nowhere, found in {entry}"
    for entry in d.get("unique_claims", []):
        assert entry.get("numeric_value") != 62.0, \
            f"the Source-line 62 must mint nowhere, found unique {entry}"


@test("POST /api/v1/compare keeps headline scores out of the separability report (v1.8.9 follow-on)")
def _():
    r = client.post("/api/v1/compare", json={"documents": [
        {"text": "Source: Kingy — \"Neon scores 70.9% on Vals Index\" (published "
                 "Nov 18, 2026).\n\n"
                 "Vals AI's November 18 results put Neon Extra High second at "
                 "70.87% on Vals Index v2.2, at about $7.40 per test. "
                 "Fable 5.2 scored 71.31%, Sonnet 5.5 scored 70.85% and Astra "
                 "scored 67.66%. Its published uncertainty for Neon is plus or "
                 "minus 0.94 percentage points, much wider than the gap to "
                 "either Fable or Sonnet.",
         "source_name": "Kingy — Neon scores 70.9% on Vals Index"},
        {"text": "On Vals Index v2.2, Neon scored 70.9% and Fable 5.2 scored 71.3%.",
         "source_name": "Second"},
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    hw = d.get("headline_weighting", [])
    assert len(hw) == 1, f"expected one headline entry, got {hw}"
    assert hw[0]["figures"][0]["kind"] == "score"
    assert hw[0]["figures"][0]["value"] == 70.9
    seps = [s for s in d.get("separability", []) if s["benchmark"] == "vals-index"]
    assert len(seps) == 1
    headline_title = "Neon scores 70.9% on Vals Index"
    assert not any(s.get("span") == headline_title
                   for s in seps[0]["scores"]), \
        "a headline score must never join a separability group"


# ── v1.8.9 correction pass, Fix 1 + Fix 3 (2026-11-20) ─────────────────────
# Fix 1: a before→after pair (baseline → with-memory) is a temporal
# comparison, not a conflict. Fix 3: restore the v1.8.8 unresolved
# indicators (needs_source_clarification / has_unresolved_comparisons).

def _packet_compare(day, feeds):
    """Compare one tests/packets/<day> packet exactly as the final-review
    runner did: Fetched segments stripped, compare docs carry
    {text, source_name, source_url} only."""
    import re as _re
    docs = []
    for fname, sname, surl in feeds:
        text = (_PACKETS_DIR / day / fname).read_text()
        text = "\n".join(
            _re.sub(r"\s*Fetched [^\n]*", "", ln)
            for ln in text.splitlines() if not ln.strip().startswith("Fetched "))
        docs.append({"text": text, "source_name": sname, "source_url": surl})
    r = client.post("/api/v1/compare", json={"documents": docs})
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


_PACKETS_DIR = SCRIPT_DIR / "tests" / "packets"

_NOV9_FEEDS = [
    ("2026-11-09-feed-primary.txt",
     "Anthropic — The components of agent memory (Lance Martin, 2026-11-06)",
     "https://www.anthropic.com/research/components-of-agent-memory"),
    ("2026-11-09-feed-explainx.txt",
     "explainx.ai — Agent Memory: Anthropic's Dreaming Tests, 13% to 48%",
     "https://explainx.ai/blog/anthropic-components-of-agent-memory-dreaming-cl-bench-pokemon-wiki-stainless-2026"),
    ("2026-11-09-feed-alfw.txt",
     "alfw.ai — Anthropic's Dreaming Repairs Notes; Wiki Scores 48%",
     "https://alfw.ai/article/anthropic-agent-memory-dreaming-wiki-48-percent/"),
    ("2026-11-09-feed-agentspulse.txt",
     "AgentsPulse — Claude Dreaming: How Anthropic's Dreams Fixes Agent Memory",
     "https://agentspulse.github.io/tutorials/claude-dreaming/"),
]

_NOV8_FEEDS = [
    ("2026-11-08-feed-primary.txt", "Google ADK Python release v2.14.0 (GitHub)",
     "https://github.com/google/adk-python/releases/tag/v2.14.0"),
    ("2026-11-08-feed-emergent.txt", "Emergent — Google ADK 2.14 Adds Teams Integration and Multi-Run Evals",
     "https://emergent.sh/news/google-adk-2-14-adds-teams-integration-multi"),
    ("2026-11-08-feed-aiweekly.txt", "AI Weekly — Google ADK 2.14.0 Graduates Core Modules, Adds Teams Runner",
     "https://aiweekly.co/alerts/google-adk-2140-graduates-core-modules-adds-teams-runner"),
    ("2026-11-08-feed-forkast.txt", "Forkast — Google ADK v2.14.0 replaces shell-command input for skill scripts",
     "https://forkast.news/google-adk-v2-14-0-replaces-shell-command-input-for-skill-scripts/"),
]

_NOV7_FEEDS = [
    ("2026-11-07-feed-primary.txt",
     "OpenAI — GPT-6 Cyber: Advancing the frontier of aligned cybersecurity agents (Nov 5, 2026)",
     "https://openai.com/index/introducing-gpt-6-cyber/"),
    ("2026-11-07-feed-linux.txt",
     "OpenAI — A thousand fewer paths to root (Nov 4, 2026)",
     "https://openai.com/index/a-thousand-fewer-paths-to-root/"),
    ("2026-11-07-feed-securityweek.txt",
     "SecurityWeek — OpenAI Says New GPT-6 Cyber Model Hunts Flaws Without Crossing the Line",
     "https://www.securityweek.com/openai-says-new-gpt-6-cyber-model-hunts-flaws-without-crossing-the-line/"),
    ("2026-11-07-feed-tsn.txt",
     "TSN Media — GPT-6 Cyber: OpenAI's Aligned Cybersecurity Model",
     "https://tsnmedia.org/gpt-6-cyber/"),
]

_NOV10_FEEDS = [
    ("2026-11-10-feed-primary.txt",
     "Anthropic — Introducing Claude Fable 5.2 and Claude Mythos 5.2 (2026-11-09)",
     "https://www.anthropic.com/claude-fable-and-mythos-5-2"),
    ("2026-11-10-feed-techtimes.txt",
     "TechTimes — Anthropic Cuts Claude Fable 5.2 Price 20%, Says Safeguards Block 55% Fewer Requests",
     "https://www.techtimes.com/articles/330367/20261110/anthropic-cuts-claude-fable-52-price-20-says-safeguards-block-55-fewer-requests.htm"),
    ("2026-11-10-feed-infoworld.txt",
     "InfoWorld — Anthropic unveils Claude Fable 5.2, Mythos 5.2 with coding gains and lower prices",
     "https://www.infoworld.com/article/4246681/anthropic-unveils-claude-fable-5-2-mythos-5-2-with-coding-gains-and-lower-prices.html"),
    ("2026-11-10-feed-medium.txt",
     "Medium — Claude Fable 5.2 review: what changed from Fable 5.1, what it costs, and who can use Mythos 5.2",
     "https://medium.com/@mcthkayaa/claude-fable-5-2-review-what-changed-from-fable-5-1-what-it-costs-and-who-can-use-mythos-5-2-888180931c37"),
]


@test("POST /api/v1/compare reads a before/after pair as a temporal comparison, not a conflict (v1.8.9 correction, Fix 1)")
def _():
    # Nov 9, verbatim sentences: both describe the SAME Sonnet 4.6
    # trajectory — 0.330 at baseline, 0.641 with memory plus refined
    # instruction. The endpoints must not conflict with each other.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": ("Prompting Sonnet 4.6 to read memory first and test theories nearly "
                  "doubled its score, from 0.330 at baseline to 0.641 with memory plus "
                  "refined instruction."),
         "source_name": "explainx"},
        {"text": ("Sonnet 4.6 barely moved (0.330 to 0.364) until prompted to read its "
                  "memory and test its theories, which took it to 0.641."),
         "source_name": "agentspulse"},
    ]}).get_json()
    assert not d.get("conflicts"), \
        f"before/after endpoints must never conflict, got {d.get('conflicts')}"
    # Temporal is not an alignment of equal values, so with nothing
    # else compared the status is not_established — not "aligned".
    assert d.get("agreement_status") == "not_established", d.get("agreement_status")
    temporal = [t for t in d.get("temporal_comparisons", [])
                if t["entity_key"] == "generic|measurement|memory|||sonnet-4-6"]
    assert len(temporal) == 1, \
        f"the pair must be surfaced as one temporal comparison, got {d.get('temporal_comparisons')}"
    assert temporal[0]["before"]["value"] == 0.33, temporal[0]
    assert temporal[0]["after"]["value"] == 0.641, temporal[0]
    # Not silently dropped: the entry carries the spans it was read from.
    assert temporal[0].get("evidence_spans"), f"temporal entry must carry evidence spans, got {temporal[0]}"


@test("POST /api/v1/compare returns the Nov 9 packet to aligned with a temporal comparison (v1.8.9 correction, Fix 1)")
def _():
    # Final review (RC2): conflicts_present, 34 shared / 33 aligned /
    # 1 conflict / 42 uniques — the one conflict is the false Sonnet 4.6
    # memory pair. After the fix: aligned, 0 conflicts, same shared /
    # aligned / unique counts, and the pair surfaced as temporal.
    d = _packet_compare("2026-11-09", _NOV9_FEEDS)
    assert d.get("agreement_status") == "aligned", d.get("agreement_status")
    assert not d.get("conflicts"), f"Nov 9 must have 0 conflicts, got {d.get('conflicts')}"
    assert len(d.get("shared_claims", [])) == 34, len(d.get("shared_claims", []))
    assert len(d.get("alignment", [])) == 33, len(d.get("alignment", []))
    assert len(d.get("unique_claims", [])) == 42, len(d.get("unique_claims", []))
    temporal = [t for t in d.get("temporal_comparisons", [])
                if t["entity_key"] == "generic|measurement|memory|||sonnet-4-6"]
    assert len(temporal) == 1, f"Nov 9 temporal pair missing, got {d.get('temporal_comparisons')}"
    assert (temporal[0]["before"]["value"], temporal[0]["after"]["value"]) == (0.33, 0.641), temporal[0]
    # The genuine Nov 9 cost conflict (128 vs 283) stays fixed: no
    # conflict anywhere in the packet, and no cost group conflicts.
    assert not [c for c in d.get("conflicts", []) if "cost" in c["entity_key"]], d.get("conflicts")
    # Double-count surfaced, not silent: the 0.330 baseline is also the
    # Figure-3 alt-text score aligned under the score metric.
    assert temporal[0].get("also_aligned_as") == ["generic|measurement|score|||sonnet-4-6"], \
        f"temporal entry must name the other metric the baseline aligns under, got {temporal[0]}"


@test("POST /api/v1/compare still conflicts on a genuine disagreement with no before/after span (v1.8.9 correction, Fix 1)")
def _():
    # Positive control for the temporal guard: two plain statements of
    # different values, neither span containing both endpoints plus a
    # transition cue, must still conflict.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "Sonnet 4.6 scored 0.641 with memory on the database task.",
         "source_name": "A"},
        {"text": "Sonnet 4.6 scored 0.400 with memory on the database task.",
         "source_name": "B"},
    ]}).get_json()
    conf = [c for c in d.get("conflicts", [])
            if c["entity_key"] == "generic|measurement|score|||sonnet-4-6"]
    assert len(conf) == 1, \
        f"a genuine disagreement must still conflict, got {d.get('conflicts')} / {d.get('temporal_comparisons')}"
    assert not d.get("temporal_comparisons"), \
        f"a genuine disagreement must not be read as temporal, got {d.get('temporal_comparisons')}"


@test("POST /api/v1/compare restores has_unresolved_comparisons and needs_source_clarification (v1.8.9 correction, Fix 3)")
def _():
    # Live v1.8.8 semantics: has_unresolved_comparisons is true when
    # needs_source_clarification or not_comparable is non-empty.
    # Nov 8 is clean (5 shared / 5 aligned / 0 not-comparable);
    # Nov 10 is not (pricing identities split).
    d8 = _packet_compare("2026-11-08", _NOV8_FEEDS)
    assert d8.get("agreement_status") == "aligned", d8.get("agreement_status")
    assert d8.get("has_unresolved_comparisons") is False, d8
    assert d8.get("needs_source_clarification") == [], d8.get("needs_source_clarification")
    assert "AGREEMENT: aligned — no unresolved comparisons" in d8["capsule"], d8["capsule"]
    d10 = _packet_compare("2026-11-10", _NOV10_FEEDS)
    assert d10.get("agreement_status") == "aligned", d10.get("agreement_status")
    assert d10.get("has_unresolved_comparisons") is True, d10
    assert isinstance(d10.get("needs_source_clarification"), list), d10
    assert d10.get("not_comparable"), \
        f"Nov 10 pricing identities must split into not_comparable, got {d10.get('not_comparable')}"
    assert "UNRESOLVED:" in d10["triage_summary"], d10["triage_summary"]
    assert "UNRESOLVED:" in d10["capsule"], d10["capsule"]
    # The Nov 10 TechTimes $10/$50 case is surfaced, never silently
    # dropped and never a conflict: the $10/$50 figures appear either
    # aligned under the prior era, split across prior-era
    # not_comparable identities, abstained in unmatched_claims, or
    # listed as bare-tier uniques. (Which bucket holds them moved
    # while Worker B's concurrent attribution fix reattributed the
    # prior-era prices, so this asserts surfacing, not one bucket.)
    assert not d10.get("conflicts"), f"Nov 10 must have 0 conflicts, got {d10.get('conflicts')}"
    surfaced_10, surfaced_50 = False, False
    for a in d10.get("alignment", []):
        if a["entity_key"] in ("pricing-input", "pricing-output"):
            for v in a.get("values", []):
                if "TechTimes" in v.get("source", "") and v["value"] == 10.0:
                    surfaced_10 = True
                if "TechTimes" in v.get("source", "") and v["value"] == 50.0:
                    surfaced_50 = True
    for n in d10.get("not_comparable", []):
        for i in n["identities"]:
            if i["era"] == "prior":
                for v in i["values"]:
                    if "TechTimes" in v.get("source", "") and v["value"] == 10.0:
                        surfaced_10 = True
                    if "TechTimes" in v.get("source", "") and v["value"] == 50.0:
                        surfaced_50 = True
    for u in d10.get("unmatched_claims", []):
        if "TechTimes" in u.get("source", "") and u.get("value") == 10.0:
            surfaced_10 = True
        if "TechTimes" in u.get("source", "") and u.get("value") == 50.0:
            surfaced_50 = True
    for u in d10.get("unique_claims", []):
        if u.get("found_in", "").startswith("TechTimes") and u.get("value") == 10.0:
            surfaced_10 = True
        if u.get("found_in", "").startswith("TechTimes") and u.get("value") == 50.0:
            surfaced_50 = True
    assert surfaced_10, \
        f"TechTimes $10 must be surfaced (aligned prior / not_comparable / unmatched / unique), got alignment={d10.get('alignment')} not_comparable={d10.get('not_comparable')} unmatched={d10.get('unmatched_claims')}"
    assert surfaced_50, \
        f"TechTimes $50 must be surfaced (aligned prior / not_comparable / unmatched / unique), got alignment={d10.get('alignment')} not_comparable={d10.get('not_comparable')} unmatched={d10.get('unmatched_claims')}"


@test("POST /api/v1/compare lists a discrepancy under an incomplete identity for clarification, never as a conflict (v1.8.9 correction, Fix 3)")
def _():
    # Unstated-subject generic disagreement: the identity is incomplete
    # (no model named on either side), so the differing values are
    # neither a conflict nor an alignment.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "The new model uses 35% fewer tokens than its predecessor.",
         "source_name": "A"},
        {"text": "The new model uses 20% fewer tokens than its predecessor.",
         "source_name": "B"},
    ]}).get_json()
    assert not d.get("conflicts"), f"unstated subjects must never conflict, got {d.get('conflicts')}"
    clar = [c for c in d.get("needs_source_clarification", []) if "tokens" in c["entity_key"]]
    assert clar and "subject" in clar[0]["missing"], \
        f"unstated-subject disagreement must need clarification on subject, got {d.get('needs_source_clarification')}"
    assert d.get("has_unresolved_comparisons") is True, d
    # One source reporting two values under one identity is ambiguity
    # inside that source, not a cross-source conflict.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": ("Opus 5.5 now costs $4 per million input tokens. "
                  "Opus 5.5 now costs $5 per million input tokens."),
         "source_name": "A"},
        {"text": "Opus 5.5 now costs $4 per million input tokens.",
         "source_name": "B"},
    ]}).get_json()
    assert not d.get("conflicts"), f"intra-source values must not conflict, got {d.get('conflicts')}"
    clar = [c for c in d.get("needs_source_clarification", [])
            if c["entity_key"] == "pricing-input"]
    assert clar and "one source reports several values" in clar[0]["missing"], \
        f"one source's two values must need clarification, got {d.get('needs_source_clarification')}"
    assert d.get("has_unresolved_comparisons") is True, d


# ── v1.8.9 correction pass, Worker B: pricing / attribution (2026-11-20) ──
# Fixes 1–5: prior/current-era pricing alignment, safeguards subject,
# Nov 7 ledger alignments, unmatched claims + unit-mismatch restore.

def _aligned_values(d, key):
    """(value -> [source names]) for every alignment entry with key."""
    out = {}
    for a in d.get("alignment", []):
        if a["entity_key"] == key:
            for v in a.get("values", []):
                out.setdefault(v["value"], []).append(v.get("source", ""))
    return out


@test("POST /api/v1/compare restores the Nov 7 ledger alignments (v1.8.9 correction, Fix 4)")
def _():
    # Final review (RC2): 12 shared / 12 aligned / 36 uniques — four
    # ledger figures extracted but left as uniques. Live v1.8.8: 16
    # shared / 16 aligned / 26 uniques. After the fix: live's numbers.
    d = _packet_compare("2026-11-07", _NOV7_FEEDS)
    assert d.get("agreement_status") == "aligned", d.get("agreement_status")
    assert not d.get("conflicts"), f"Nov 7 must have 0 conflicts, got {d.get('conflicts')}"
    assert len(d.get("shared_claims", [])) == 16, len(d.get("shared_claims", []))
    assert len(d.get("alignment", [])) == 16, len(d.get("alignment", []))
    assert len(d.get("unique_claims", [])) == 26, len(d.get("unique_claims", []))
    av = _aligned_values(d, "generic|percentage|out-of-scope avoidance|||gpt-6-cyber")
    assert sorted(av.get(99.93, [])) and len(av.get(99.93, [])) == 4, \
        f"avoidance 99.93 must align 4-way (primary x2, SecurityWeek, TSN), got {av}"
    assert any("TSN" in s for s in av.get(99.93, [])), av
    av = _aligned_values(d, "generic|percentage|recall|||gpt-6-cyber")
    assert len(av.get(40.3, [])) == 2, f"recall 40.3 must align primary + TSN, got {av}"
    av = _aligned_values(d, "generic|percentage|completion|||gpt-6-cyber")
    assert len(av.get(95.0, [])) == 2, f"completion 95 must align primary + TSN, got {av}"
    av = _aligned_values(d, "generic|percentage|refusal|||gpt-6-astra")
    assert len(av.get(100.0, [])) == 2, f"refusal 100 must align primary + TSN, got {av}"


@test("POST /api/v1/compare attributes 'the new model achieved X%, up from Y% for <Model>' to the earlier subject, not the later model (v1.8.9 correction, Nov 7 48.5)")
def _():
    # SecurityWeek Nov 7: "the new model achieved 48.5%, up from 32.1%
    # for GPT-5.6 Cyber" — the 48.5% is the new model's (GPT-6 Cyber),
    # the 32.1% is GPT-5.6 Cyber's. The later "for <Model>" owner must
    # not be claimed as the earlier figure's subject by the exactly-one
    # fallback; the figure mints unstated and the compare merge
    # attaches it to the one value-matching stated group.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": ("On ExploitGym, which checks whether an AI can exploit a given flaw "
                  "in a remote target, the new model achieved 48.5%, up from 32.1% "
                  "for GPT-5.6 Cyber."),
         "source_name": "securityweek"},
        {"text": ("GPT-6 Cyber scores 48.5% on ExploitGym, up from 32.1% for "
                  "GPT-5.6 Cyber."),
         "source_name": "primary"},
    ]}).get_json()
    assert not d.get("conflicts"), f"48.5/32.1 must never conflict, got {d.get('conflicts')}"
    av = _aligned_values(d, "generic|percentage|score|bench=exploitgym||gpt-6-cyber")
    assert len(av.get(48.5, [])) == 2, \
        f"48.5 must align x2 under gpt-6-cyber, got {av}"
    assert not _aligned_values(d, "generic|percentage|score|bench=exploitgym||gpt-5-6-cyber").get(48.5), \
        "the 48.5 must not sit under gpt-5-6-cyber (the compared-against model)"
    av32 = _aligned_values(d, "generic|percentage|score|bench=exploitgym||gpt-5-6-cyber")
    assert len(av32.get(32.1, [])) == 2, \
        f"32.1 must still align x2 under gpt-5-6-cyber, got {av32}"


@test("POST /api/v1/compare leaves an unidentified 'new model' figure unstated instead of guessing its subject (v1.8.9 correction, Nov 7 48.5)")
def _():
    # Same sentence shape, but no other document states the 48.5%
    # under any subject — there is nothing to attach it to, so it
    # must stand as an unstated unique, never forced onto GPT-5.6
    # Cyber (the compared-against model) or any other named model.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": ("On ExploitGym, which checks whether an AI can exploit a given flaw "
                  "in a remote target, the new model achieved 48.5%, up from 32.1% "
                  "for GPT-5.6 Cyber."),
         "source_name": "securityweek"},
        {"text": ("GPT-5.6 Cyber scored 32.1% on ExploitGym."),
         "source_name": "baseline"},
    ]}).get_json()
    assert not d.get("conflicts"), f"48.5/32.1 must never conflict, got {d.get('conflicts')}"
    un = [u for u in d.get("unique_claims", []) if u.get("value") == 48.5]
    assert len(un) == 1, f"48.5 must stand as one unique, got {un}"
    assert un[0].get("entity_key") == "generic|percentage|score|bench=exploitgym||unstated", \
        f"48.5 must stay unstated, got {un[0].get('entity_key')}"
    av = _aligned_values(d, "generic|percentage|score|bench=exploitgym||gpt-5-6-cyber")
    assert len(av.get(32.1, [])) == 2, \
        f"32.1 must still align x2 under gpt-5-6-cyber, got {av}"


@test("POST /api/v1/compare aligns prior and current Fable pricing without mixing eras (v1.8.9 correction, Fixes 1+2)")
def _():
    # Live v1.8.8: $10 x3 and $50 x3 (prior, Fable 5.1); $8 x4 and $40 x4
    # (current, Fable 5.2, primary included). RC2 never aligned the
    # prior prices and partitioned the primary's $8/$40 under mythos-5-1.
    d = _packet_compare("2026-11-10", _NOV10_FEEDS)
    assert not d.get("conflicts"), f"Nov 10 must have 0 conflicts, got {d.get('conflicts')}"
    av = _aligned_values(d, "pricing-input")
    assert sorted(av.get(10.0, [])) and len(av.get(10.0, [])) == 3, \
        f"prior $10 must align x3, got {av}"
    assert sorted(av.get(8.0, [])) and len(av.get(8.0, [])) == 4, \
        f"current $8 must align x4 including the primary, got {av}"
    assert any("Anthropic" in s for s in av.get(8.0, [])), \
        f"the primary must join the $8 alignment, got {av}"
    av = _aligned_values(d, "pricing-output")
    assert len(av.get(50.0, [])) == 3, f"prior $50 must align x3, got {av}"
    assert len(av.get(40.0, [])) == 4, f"current $40 must align x4, got {av}"
    assert any("Anthropic" in s for s in av.get(40.0, [])), \
        f"the primary must join the $40 alignment, got {av}"
    # Eras never mix: no aligned pricing group holds both eras' values.
    for a in d.get("alignment", []):
        if a["entity_key"] in ("pricing-input", "pricing-output"):
            vals = {v["value"] for v in a.get("values", [])}
            assert not ({8.0, 10.0} <= vals or {40.0, 50.0} <= vals), \
                f"eras mixed in one alignment, got {a}"
    # ...and the not_comparable identities keep the eras apart too.
    nc = {n["entity_key"]: [(i["era"], i["subject"]) for i in n["identities"]]
          for n in d.get("not_comparable", [])}
    assert ("prior", "fable-5-1") in nc.get("pricing-input", []), nc
    assert ("unknown", "fable-5-2") in nc.get("pricing-input", []), nc


@test("POST /api/v1/compare files the 60% fewer-blocks figure under Fable 5.2 (v1.8.9 correction, Fix 3)")
def _():
    # "55% fewer blocks with Fable 5.2 compared with Fable 5.1 — about
    # 60% fewer in cybersecurity": the measured thing is Fable 5.2,
    # Fable 5.1 is the baseline. Live files it under fable-5-2; RC2
    # filed it under fable-5-1.
    d = _packet_compare("2026-11-10", _NOV10_FEEDS)
    av = _aligned_values(d, "generic|percentage|blocks|rel||fable-5-2")
    assert len(av.get(60.0, [])) == 3, \
        f"blocks 60.0 must align x3 under fable-5-2, got {av}"
    assert any("Anthropic" in s for s in av.get(60.0, [])), av
    assert not _aligned_values(d, "generic|percentage|blocks|rel||fable-5-1").get(60.0), \
        "the 60% must not sit under fable-5-1 (the baseline)"


@test("POST /api/v1/compare leaves Nov 10 with zero unmatched claims and restores the cost unit mismatch (v1.8.9 correction, Fix 5)")
def _():
    # RC2: 4 unmatched (TechTimes $10/$50, TechTimes $1.00, Medium
    # $6.721) and no unit_mismatches. Live: 0 unmatched, one
    # unit_mismatch ($75 billion vs $14.83, count vs $).
    d = _packet_compare("2026-11-10", _NOV10_FEEDS)
    assert d.get("unmatched_claims") == [], \
        f"Nov 10 must have 0 unmatched claims, got {d.get('unmatched_claims')}"
    mm = [m for m in d.get("unit_mismatches", [])
          if m["entity_key"] == "generic|cost|cost|||unstated"]
    assert len(mm) == 1, f"cost unit mismatch missing, got {d.get('unit_mismatches')}"
    assert sorted(v["value"] for v in mm[0]["values"]) == [14.83, 75000000000.0], mm[0]
    assert sorted(mm[0]["units"]) == ["$", "count"], mm[0]
    # The TechTimes Astra prices are owned, not abstained: bare-tier
    # uniques under the "pricing" key, never unmatched.
    bare = {(u.get("value")) for u in d.get("unique_claims", [])
            if u.get("entity_key") == "pricing"
            and (u.get("found_in") or "").startswith("TechTimes")}
    assert {10.0, 50.0} <= bare, f"Astra $10/$50 must stand as bare-pricing uniques, got {bare}"


@test("POST /api/v1/compare gives an against-pair's second side to the compared-against model (v1.8.9 correction, Fix 1)")
def _():
    # "cached tokens for Fable 5.2 cost $0.20 per million against $1.00
    # for Astra": the $1.00 is Astra's cached price, not Fable 5.2's —
    # the old nearest-model rule claimed it for Fable 5.2 (the nearest
    # token before the amount), so it never met Astra's own $1.00.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": ("Cached tokens for Fable 5.2 cost $0.20 per million against "
                  "$1.00 for Astra."),
         "source_name": "techtimes"},
        {"text": ("Astra's cached tokens cost $1.00 per million."),
         "source_name": "price-list"},
    ]}).get_json()
    assert not d.get("conflicts"), f"against-pair prices must never conflict, got {d.get('conflicts')}"
    assert len(_aligned_values(d, "pricing-cached").get(1.0, [])) == 2, \
        f"Astra $1.00 cached must align, got {_aligned_values(d, 'pricing-cached')}"


@test("POST /api/v1/compare keeps a possessive baseline as the figure's owner (v1.8.9 correction, Fix 3)")
def _():
    # "compared with Argon's 50": Argon is the comparison baseline AND
    # the 50 is Argon's own score — the possessive makes it the owner,
    # so the baseline skip must not strip it to a bare "argon" slug or
    # leave it unstated. (Guards the baseline helper, not the RC2 bug:
    # pristine also attributed this one; the naive skip broke it.)
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": ("Hydrogen scores 51 on the Artificial Analysis Intelligence Index "
                  "at its default high setting and 54 at xhigh, compared with "
                  "Argon's 50."),
         "source_name": "alphasignal"},
        {"text": ("Argon scored 50 on the Artificial Analysis Intelligence Index."),
         "source_name": "index"},
    ]}).get_json()
    assert not d.get("conflicts"), f"Argon 50 must never conflict, got {d.get('conflicts')}"
    av = _aligned_values(d, "generic|task_count|score|bench=artificial-analysis||gemini-4-argon")
    assert len(av.get(50.0, [])) == 2, \
        f"Argon's 50 must align under gemini-4-argon, got {av}"


# ── runner ─────────────────────────────────────────────────────────────────────

def run() -> int:
    failed = 0
    print(f"{DIM}Signal Sieve API test suite{RESET}\n")
    for name, fn in TESTS:
        try:
            fn()
            print(f"{GREEN}PASS{RESET}  {name}")
        except Exception as exc:
            print(f"{RED}FAIL{RESET}  {name}")
            print(f"  {DIM}{exc}{RESET}")
            failed += 1
    total = len(TESTS)
    print(f"\n{total - failed}/{total} passed")
    if failed:
        print(f"{RED}{failed} failed{RESET}")
    return failed


if __name__ == "__main__":
    sys.exit(run())
