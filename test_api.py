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
from signal_sieve import _hosted_release_artifact
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
    assert not any(v == 13.1 for _k, v in unique), "Terminal-Bench is unmapped — 13.1% must stay invisible"


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
    assert "generic|speed|nodes=30;turns=200||unstated" in aligned, \
        f"expected speed aligned, got {list(aligned)}"
    assert aligned["generic|speed|nodes=30;turns=200||unstated"] == [23.0, 23.0]
    assert aligned.get("generic|memory|nodes=30;turns=200||unstated") == [9.0, 9.0], \
        f"expected memory aligned, got {list(aligned)}"
    assert aligned.get("generic|modules|||unstated") == [26.0, 26.0]
    assert aligned.get("generic|features|||unstated") == [16.0, 16.0]
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
    assert aligned.get("generic|out-of-scope avoidance|||gpt-6-cyber") == [99.93, 99.93], \
        f"expected GPT-6 99.93 aligned, got {aligned}"
    assert aligned.get("generic|out-of-scope avoidance|||gpt-5-6-cyber") == [94.49, 94.49], \
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
    assert aligned.get("generic|modules|||unstated") == [26.0, 26.0], \
        f"expected modules 26 aligned, got {aligned}"
    assert aligned.get("generic|features|||unstated") == [16.0, 16.0], \
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
    assert aligned.get("generic|completion|||unstated") == [95.0, 95.0], \
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
                if c["entity_key"] == "generic|completion|||unstated"], \
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
