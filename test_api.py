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
    # subject with the primary's stated one, so it aligns 2-way.
    # v1.8.8 r3: the primary's subject is Fable 5.2, not Fable 5.1 —
    # "55% fewer blocks with Fable 5.2 compared with Fable 5.1" names
    # 5.1 as the comparison BASELINE, and a baseline is never the
    # subject (Arch audit 2026-11-17). This test used to pin the
    # baseline-as-subject mis-binding.
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


@test("POST /api/v1/compare flags two undated prices that disagree for source clarification (v1.8.8 r3)")
def _():
    # Two reports that both state no era still compare with each
    # other — that is the November 11 replay's behaviour, where all
    # four $2/$10 reports state no era and align.
    # v1.8.8 r3 (Arch audit 2026-11-17, core invariant): a CONFLICT
    # needs the reference era on every side. Two undated prices that
    # disagree may simply be from different dates, so they are listed
    # under needs_source_clarification (missing: era) — never a
    # conflict, never an alignment. Until r3 this test asserted a
    # conflict. The dated pair below ("today" vs "now") is the
    # positive control that still conflicts.
    r = client.post("/api/v1/compare", json={
        "documents": [
            {"text": "Carbon input costs $2 per million tokens.",
             "source_name": "A"},
            {"text": "Carbon input costs $3 per million tokens.",
             "source_name": "B"},
        ],
    })
    d = r.get_json()
    assert not d.get("conflicts"), \
        f"two undated prices must never conflict, got {d.get('conflicts')}"
    clar = [c for c in d.get("needs_source_clarification", [])
            if c["entity_key"] == "pricing-input"]
    assert clar and clar[0]["missing"] == ["era"], \
        f"undated $2 vs $3 must need clarification on era only, got {d.get('needs_source_clarification')}"
    assert {v["subject"] for v in clar[0]["values"]} == {"gemini-4-carbon"}, \
        f"both sides must carry the stated subject, got {clar[0]['values']}"
    assert not [a for a in d.get("alignment", []) if a["entity_key"] == "pricing-input"], \
        f"undated $2 vs $3 must not align, got {d.get('alignment')}"
    assert "need source clarification" in d["triage_summary"], d["triage_summary"]
    assert "NEEDS SOURCE CLARIFICATION" in d["capsule"], d["capsule"]
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
    # v1.8.8 r3: the conflict carries the identity it was compared as.
    assert d["conflicts"][0].get("identity") == {"subject": "gemini-4-carbon", "era": "current"}, \
        f"conflict identity must be Carbon / current, got {d['conflicts'][0].get('identity')}"


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


# ── v1.8.8 r3: conflict integrity (Arch audit 2026-11-17) ─────────────────────

_NOV16_DIR = SCRIPT_DIR / "tests" / "packets" / "2026-11-16"


def _nov16_docs():
    docs = []
    for f in sorted(_NOV16_DIR.glob("*.txt")):
        docs.append({"text": f.read_text(encoding="utf-8"),
                     "source_name": f.stem.rsplit("-feed-", 1)[-1]})
    return docs


def _packet_docs(name):
    d = SCRIPT_DIR / "tests" / "packets" / name
    docs = []
    for f in sorted(d.glob("*.txt")):
        docs.append({"text": f.read_text(encoding="utf-8"),
                     "source_name": f.stem.rsplit("-feed-", 1)[-1]})
    return docs


@test("POST /api/v1/compare reports zero conflicts on the Nov 16 five-feed packet (v1.8.8 r3)")
def _():
    # Arch audit 2026-11-17: r2 reported six conflicts on this packet,
    # all mis-scoped (HLE across models, three pooled price lists, a
    # "62 percent" read as 62 tasks, 35% vs 50% tokens). None is a
    # same-identity contradiction, so none may be a conflict.
    docs = _nov16_docs()
    assert len(docs) == 5, f"expected the five Nov 16 feeds, got {[d['source_name'] for d in docs]}"
    r = client.post("/api/v1/compare", json={"documents": docs})
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    d = r.get_json()
    assert d.get("conflicts") == [], \
        f"no misleading conflicts allowed, got {[(c['entity_key'], [v['value'] for v in c['values']]) for c in d['conflicts']]}"
    aligned = {}
    for a in d.get("alignment", []):
        aligned.setdefault(a["entity_key"], []).append(
            (a.get("identity", {}).get("subject", ""),
             sorted({v["value"] for v in a["values"]}),
             sorted({v["source"] for v in a["values"]})))
    # The genuine same-model price agreements survive, scoped by model.
    assert ("qwen-4-max", [1.5], ["askthisguy", "digitaltrends"]) in aligned.get("pricing-input", []), aligned
    assert ("qwen-4-max", [4.5], ["askthisguy", "digitaltrends"]) in aligned.get("pricing-output", []), aligned
    assert ("opus-5-5", [20.0], ["digitaltrends", "infoworld"]) in aligned.get("pricing-output", []), aligned
    # Every alignment shows the identity it was compared as. (An
    # alignment may have an unstated subject — the $4.43 per-task
    # figure all three sources repeat names no model — but the gap is
    # visible on the entry, and such a group can never conflict.)
    for a in d["alignment"]:
        if a["entity_key"].startswith("pricing") or a["entity_key"].startswith("generic|"):
            assert "identity" in a and "subject" in a["identity"], a
    # "62 percent reduction in task costs" is a percentage, never a task count.
    for bucket in ("alignment", "unique_claims", "needs_source_clarification"):
        for c in d.get(bucket, []):
            if c["entity_key"].startswith("generic|task_count|tasks"):
                vals = [v["value"] for v in c.get("values", [])] or [c.get("value")]
                assert 62.0 not in vals, f"62 percent minted as a task count in {bucket}: {c}"
    # Whatever stays unresolved is listed with its gaps and spans, never hidden.
    for c in d.get("needs_source_clarification", []):
        assert c["missing"], c
        assert all(v.get("context") for v in c["values"]), c


@test("POST /api/v1/compare binds chart series positionally on the Nov 16 packet (v1.8.8 r4)")
def _():
    # r3 recommendation 1: "for A, B and C: Bench v1, v2 and v3" binds
    # value i to model i. The Nov 16 clarification closes — the HLE
    # values now carry their owners — and the group becomes a visible
    # not_comparable record (different models, different scores), never
    # a conflict and never an alignment.
    docs = _nov16_docs()
    d = client.post("/api/v1/compare", json={"documents": docs}).get_json()
    assert d.get("conflicts") == [], d.get("conflicts")
    assert d.get("needs_source_clarification") == [], \
        f"the last Nov 16 clarification must close, got {d.get('needs_source_clarification')}"
    hle = [n for n in d.get("not_comparable", []) if n["entity_key"] == "hle"]
    assert len(hle) == 1, f"one HLE not_comparable record expected, got {d.get('not_comparable')}"
    owners = {i["subject"]: sorted(v["value"] for v in i["values"]) for i in hle[0]["identities"]}
    assert owners == {
        "qwen-3-6-27b": [24.0],
        "qwen-3-8-27b": [30.8],
        "opus-4-6-max": [40.0],
        "": [51.7],
    }, owners
    # The 30.8 and 40.0 never minted before r4; the 53.4/61.7 SWE-bench
    # and 89.2/91.3 GPQA values mint too. All are unique claims — no
    # second document reports any chart value for any model — so no
    # 12th alignment exists in this data; the honest Nov 16 outcome
    # is 11 aligned / 0 conflicts / 3 not-comparable / 0 clarification.
    uniques = {(u["entity_key"], u["value"], u["found_in"])
               for u in d.get("unique_claims", [])}
    for want in (("swe-bench", 61.7, "askthisguy"), ("swe-bench", 53.4, "askthisguy"),
                 ("gpqa", 89.2, "askthisguy"), ("gpqa", 91.3, "askthisguy")):
        assert want in uniques, f"chart value {want} must mint as a unique claim"


@test("extract_numeric_claims binds 'for A, B and C:' chart series positionally (v1.8.8 r4)")
def _():
    line = ("Bar chart of scores published by Qwen, out of 100, for Qwen 3.6 27B, "
            "Qwen 3.8 27B and Opus 4.6 Max: SWE-bench Pro 53.5, 61.7 and 53.4; "
            "Terminal Bench 2.1 63.4, 73.0 and 78.2; HLE 24.0, 30.8 and 40.0.")
    claims = extract_numeric_claims(line)
    got = {(c["entity_key"], c["numeric_value"], c["subject"], c["variant"]) for c in claims}
    assert ("swe-bench", 53.5, "qwen-3-6-27b", "pro") in got, got
    assert ("swe-bench", 61.7, "qwen-3-8-27b", "pro") in got, got
    assert ("swe-bench", 53.4, "opus-4-6-max", "pro") in got, got
    assert ("hle", 24.0, "qwen-3-6-27b", "") in got, got
    assert ("hle", 30.8, "qwen-3-8-27b", "") in got, got
    assert ("hle", 40.0, "opus-4-6-max", "") in got, got
    # Terminal Bench is unmapped: nothing mints for it, and the "2.1"
    # version token is never read as a score.
    assert not any(c["numeric_value"] in (2.1, 63.4, 73.0, 78.2) for c in claims), got
    # A prose "for" line is not a chart: no positional binding fires.
    prose = extract_numeric_claims("For best results, GPQA: 61.3%.")
    assert not any(c["subject"] for c in prose if c["entity_key"] == "gpqa"), prose
    # A series whose value count does not match the model count fails
    # closed: only the inline pass's first value mints, subject-less.
    mismatch = extract_numeric_claims(
        "for Qwen 3.6 27B and Qwen 3.8 27B: HLE 24.0, 30.8 and 40.0.")
    hle = [c for c in mismatch if c["entity_key"] == "hle"]
    assert len(hle) == 1 and hle[0]["numeric_value"] == 24.0 and hle[0]["subject"] == "", hle


@test("POST /api/v1/compare reports has_unresolved_comparisons on Nov 16 and false on a clean packet (v1.8.8 r4)")
def _():
    # Arch audit 2026-11-17, risk 1: "aligned" must never read as a
    # clean bill of health while comparisons sit unresolved. The flag
    # is additive; the agreement_status enum is unchanged.
    d = client.post("/api/v1/compare", json={"documents": _nov16_docs()}).get_json()
    assert d.get("agreement_status") == "aligned", d
    assert d.get("has_unresolved_comparisons") is True, d
    assert len(d.get("not_comparable", [])) == 3, d.get("not_comparable")
    assert "UNRESOLVED: 0 discrepancy(ies) need source clarification, 3 claim key(s) are not comparable" in d["triage_summary"], d["triage_summary"]
    assert "UNRESOLVED: 3 (0 clarification, 3 not comparable)" in d["capsule"], d["capsule"]
    assert "AGREEMENT: aligned — with 3 unresolved comparison(s)" in d["capsule"], d["capsule"]
    # A fully clean packet: Nov 07, 16 aligned / 0 conflicts /
    # 0 not-comparable / 0 clarification.
    d7 = client.post("/api/v1/compare", json={"documents": _packet_docs("2026-11-07")}).get_json()
    assert d7.get("agreement_status") == "aligned", d7
    assert d7.get("has_unresolved_comparisons") is False, d7
    assert "AGREEMENT: aligned — no unresolved comparisons" in d7["capsule"], d7["capsule"]
    assert "UNRESOLVED: 0 (0 clarification, 0 not comparable)" in d7["capsule"], d7["capsule"]


@test("POST /api/v1/compare sets has_unresolved_comparisons only for clarification / not-comparable (v1.8.8 r4)")
def _():
    # Undated Carbon $2 vs $3: a safe abstention — clarification, flag
    # true, and nothing was actually compared, so the status is
    # not_established rather than aligned.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "Carbon input costs $2 per million tokens.", "source_name": "A"},
        {"text": "Carbon input costs $3 per million tokens.", "source_name": "B"},
    ]}).get_json()
    assert d.get("has_unresolved_comparisons") is True, d
    assert d.get("agreement_status") == "not_established", d
    assert "UNRESOLVED" in d["triage_summary"], d["triage_summary"]
    assert "UNRESOLVED: 1 (1 clarification, 0 not comparable)" in d["capsule"], d["capsule"]
    # Two agreeing docs: clean, flag false.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "Carbon input costs $2 per million tokens today.", "source_name": "A"},
        {"text": "Carbon input costs $2 per million tokens today.", "source_name": "B"},
    ]}).get_json()
    assert d.get("has_unresolved_comparisons") is False, d
    assert d.get("agreement_status") == "aligned", d
    # A dated conflict IS a comparison — resolved, not "unresolved".
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "Carbon input costs $2 per million tokens today.", "source_name": "A"},
        {"text": "Carbon input costs $3 per million tokens now.", "source_name": "B"},
    ]}).get_json()
    assert d.get("has_unresolved_comparisons") is False, d
    assert d.get("agreement_status") == "conflicts_present", d


@test("POST /api/v1/compare keeps a genuine same-model price conflict inside a multi-feed packet (v1.8.8 r3)")
def _():
    # Positive control against a blanket suppressor: the Nov 16 packet
    # plus one feed that states a DIFFERENT current Qwen4-Max price.
    # Same model, same tier, same stated era, same unit — that is a
    # real contradiction and must still be a conflict.
    docs = _nov16_docs() + [
        {"text": ("Qwen4-Max now costs $1.80 per million input tokens and $5.40 "
                  "per million output tokens."), "source_name": "contrarian"},
    ]
    d = client.post("/api/v1/compare", json={"documents": docs}).get_json()
    conf = [c for c in d.get("conflicts", []) if c["entity_key"] == "pricing-input"]
    assert len(conf) == 1, f"the dated Qwen4-Max input price dispute must conflict, got {d.get('conflicts')}"
    assert conf[0]["identity"] == {"subject": "qwen-4-max", "era": "current"}, conf[0]
    assert sorted(v["value"] for v in conf[0]["values"]) == [1.5, 1.8], conf[0]
    out = [c for c in d.get("conflicts", []) if c["entity_key"] == "pricing-output"]
    assert len(out) == 1 and sorted(v["value"] for v in out[0]["values"]) == [4.5, 5.4], \
        f"the dated Qwen4-Max output price dispute must conflict, got {d.get('conflicts')}"


@test("POST /api/v1/compare keeps a genuine same-model benchmark conflict (v1.8.8 r3)")
def _():
    # Positive control: same model, same benchmark, no stated variant
    # or version on either side — a real disagreement stays a conflict.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "On GPQA, Opus 5.5 scored 87.0%.", "source_name": "A"},
        {"text": "On GPQA, Opus 5.5 scored 83.5%.", "source_name": "B"},
    ]}).get_json()
    conf = [c for c in d.get("conflicts", []) if c["entity_key"] == "gpqa"]
    assert len(conf) == 1, f"same-model GPQA dispute must conflict, got {d.get('conflicts')} / {d.get('needs_source_clarification')}"
    assert conf[0]["identity"]["subject"] == "opus-5-5", conf[0]
    # The same two scores with the model named on one side only are a
    # clarification, not a conflict.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "On GPQA, Opus 5.5 scored 87.0%.", "source_name": "A"},
        {"text": "GPQA: 83.5%.", "source_name": "B"},
    ]}).get_json()
    assert not d.get("conflicts"), f"subject on one side only must not conflict, got {d.get('conflicts')}"
    assert not [a for a in d.get("alignment", []) if a["entity_key"] == "gpqa"], d.get("alignment")
    # Not silently two uniques either: the split is a visible
    # not_comparable record naming both identities.
    nc = [n for n in d.get("not_comparable", []) if n["entity_key"] == "gpqa"]
    assert nc and {i["subject"] for i in nc[0]["identities"]} == {"opus-5-5", ""}, \
        f"stated vs unstated model must be recorded not_comparable, got {d.get('not_comparable')}"
    # Unstated on BOTH sides and disagreeing: clarification, not conflict.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "GPQA: 87.0%.", "source_name": "A"},
        {"text": "GPQA: 83.5%.", "source_name": "B"},
    ]}).get_json()
    assert not d.get("conflicts"), f"unstated models must never conflict, got {d.get('conflicts')}"
    clar = [c for c in d.get("needs_source_clarification", []) if c["entity_key"] == "gpqa"]
    assert clar and "subject" in clar[0]["missing"], d.get("needs_source_clarification")


@test("POST /api/v1/compare keeps a genuine same-subject generic conflict and gates an unstated one (v1.8.8 r3)")
def _():
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "Qwen4-Max uses 35% fewer tokens than its predecessor.", "source_name": "A"},
        {"text": "Qwen4-Max uses 20% fewer tokens than its predecessor.", "source_name": "B"},
    ]}).get_json()
    conf = [c for c in d.get("conflicts", []) if "tokens" in c["entity_key"]]
    assert len(conf) == 1 and conf[0]["identity"]["subject"] == "qwen-4-max", \
        f"same-subject token figures that disagree must conflict, got {d.get('conflicts')}"
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "The new model uses 35% fewer tokens than its predecessor.", "source_name": "A"},
        {"text": "The new model uses 20% fewer tokens than its predecessor.", "source_name": "B"},
    ]}).get_json()
    assert not d.get("conflicts"), f"unstated subjects must never conflict, got {d.get('conflicts')}"
    assert [c for c in d.get("needs_source_clarification", []) if "tokens" in c["entity_key"]], \
        f"unstated-subject disagreement must be listed for clarification, got {d.get('needs_source_clarification')}"


@test("POST /api/v1/compare never turns one source's two values into a cross-source conflict (v1.8.8 r3)")
def _():
    # A states $4 and (elsewhere) $5 for the same model, tier and era;
    # B states $4. A agrees with B on $4 — A's extra $5 is ambiguity
    # inside A, listed for clarification, not a conflict with B.
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": "Opus 5.5 now costs $4 per million input tokens. Opus 5.5 now costs $5 per million input tokens.",
         "source_name": "A"},
        {"text": "Opus 5.5 now costs $4 per million input tokens.", "source_name": "B"},
    ]}).get_json()
    assert not d.get("conflicts"), f"intra-source values must not conflict, got {d.get('conflicts')}"
    clar = [c for c in d.get("needs_source_clarification", []) if c["entity_key"] == "pricing-input"]
    assert clar and "one source reports several values" in clar[0]["missing"], d.get("needs_source_clarification")


@test("extraction coverage does not count dates, URLs, times or identifiers as candidates (v1.8.8 r3)")
def _():
    # Arch audit 2026-11-17: Reuters' 0/10 mint ratio was a measure
    # problem — its "numbers" were a timestamp, a URL date, "Nov 15",
    # "3D" and "Qwen4". A document with no numeric propositions is not
    # a recall defect.
    wire = ("Published: 2026-11-16T04:29:29+00:00. Fetched via "
            "https://wixx.com/2026/11/15/alibaba-unveils/\n"
            "Nov 15 (Reuters) - Alibaba unveiled Qwen4-Max, spanning 3D modeling and "
            "game development, and said it outperforms Qwen3.8-Max.")
    d = client.post("/api/v1/compare", json={"documents": [
        {"text": wire, "source_name": "wire"},
        {"text": "Qwen4-Max is priced at $1.50 per million input tokens.", "source_name": "B"},
    ]}).get_json()
    cov = {c["source"]: c for c in d["extraction_coverage"]}
    assert cov["wire"]["numbers_seen"] == 0, f"no candidate numbers in the wire text, got {cov['wire']}"
    assert "wire" not in d["triage_summary"].split("Extraction coverage")[-1].split("See extraction_coverage")[0] \
        or "Extraction coverage" not in d["triage_summary"], d["triage_summary"]
    assert all(0.0 <= c["coverage"] <= 1.0 for c in d["extraction_coverage"])


@test("POST /api/v1/compare never binds a comparison baseline as the subject (v1.8.8 r3)")
def _():
    # "outperforms its predecessor, Qwen3.8-Max ... 35% fewer tokens":
    # the 35% is Qwen4-Max's. "Outperforms Qwen3.8-Max with ~50% fewer
    # tokens" names no subject at all. Binding the baseline made the
    # two read as one stated identity.
    claims = extract_generic_quantitative_claims(
        "Qwen4-Max outperforms its predecessor, Qwen3.8-Max, overall while "
        "using 35% fewer tokens at High effort.", "A", [])
    subj = {c["numeric_value"]: c["subject"] for c in claims}
    assert subj.get(35.0) == "qwen-4-max", f"35% belongs to Qwen4-Max, got {subj}"
    claims = extract_generic_quantitative_claims(
        "Outperforms Qwen3.8-Max with up to ~50% fewer tokens.", "B", [])
    subj = {c["numeric_value"]: c["subject"] for c in claims}
    assert subj.get(50.0) == "", f"a baseline is never the subject, got {subj}"


@test("extract_numeric_claims binds standard-tier prices to the model they are quoted for (v1.8.8 r3)")
def _():
    text = ("It costs $1.50 per million input tokens and $4.50 per million output tokens in "
            "Singapore, against $2 and $6 for Qwen 3.8 Max. In comparison, Claude Opus 5.5 "
            "costs $4 and $20 for input and output tokens, respectively. Google's Gemini 4 "
            "Argon and Carbon both currently cost $2 and $10, respectively, for input and "
            "output tokens. GPT-6.1 Terra charges more for output tokens at $5 per million.")
    got = sorted((c["entity_key"], c["numeric_value"], c["subject"], c["era"])
                 for c in extract_numeric_claims(text))
    want = sorted([
        ("pricing-input", 1.5, "", "unknown"),          # "It" — never guessed
        ("pricing-output", 4.5, "", "unknown"),
        ("pricing-input", 2.0, "qwen-3-8-max", "unknown"),
        ("pricing-output", 6.0, "qwen-3-8-max", "unknown"),
        ("pricing-input", 4.0, "opus-5-5", "unknown"),
        ("pricing-output", 20.0, "opus-5-5", "unknown"),
        ("pricing-input", 2.0, "gemini-4-argon", "current"),
        ("pricing-input", 2.0, "gemini-4-carbon", "current"),
        ("pricing-output", 10.0, "gemini-4-argon", "current"),
        ("pricing-output", 10.0, "gemini-4-carbon", "current"),
        ("pricing-output", 5.0, "gpt-6-1-terra", "unknown"),
    ])
    assert got == want, f"\n got  {got}\n want {want}"
    # A model pair without "both/each" may be a respectively-series:
    # unresolved, never guessed.
    got = {(c["numeric_value"], c["subject"]) for c in extract_numeric_claims(
        "Gemini 4 Argon and Carbon cost $2 and $3 per million input tokens.")}
    assert got == {(2.0, ""), (3.0, "")}, got


@test("price owners skip comparison baselines and stay fail-closed when ambiguous (v1.8.8 r3)")
def _():
    def owners(text):
        return [(c["entity_key"], c["numeric_value"], c["subject"], c["era"])
                for c in extract_numeric_claims(text)]
    # "twice as much as Opus 5.5 ($8 / $40 against $4 / $20)": the
    # first pair is Fable 5.2's, the second Opus 5.5's.
    assert owners("Fable 5.2 costs twice as much as Opus 5.5 ($8 / $40 against $4 / $20 per million).") == [
        ("pricing-input", 8.0, "fable-5-2", "unknown"), ("pricing-output", 40.0, "fable-5-2", "unknown"),
        ("pricing-input", 4.0, "opus-5-5", "unknown"), ("pricing-output", 20.0, "opus-5-5", "unknown")]
    # A baseline coordinated with a baseline is a baseline: never
    # "Mythos 5.1" for the new price. The new pair is not marked
    # "both", so the owner stays unresolved rather than guessed.
    got = owners("Fable 5.2 and Mythos 5.2 are faster than Claude Fable 5.1 and Claude Mythos 5.1, "
                 "and are available at a lower price: $8 per million input tokens.")
    assert got == [("pricing-input", 8.0, "", "unknown")], got
    # An "against" belongs to the price right after it: the batch
    # pair after a closed cache-read pair is not a prior price.
    got = owners("Fable 5.2 is cheaper than Fable 5.1: cache reads ($0.20 against $0.25) "
                 "and batch ($4 / $20 against $5 / $25) are all 20% lower.")
    assert ("pricing-batch-input", 4.0, "fable-5-1", "prior") not in got, got
    assert ("pricing-batch-input", 5.0, "fable-5-1", "prior") in got, got
    assert ("pricing-cached", 0.2, "fable-5-2", "unknown") in got, got
    # "cheaper than Fable 5.2 at $4": after "than", "at" does not make
    # the baseline the owner — the $4 is the subject's (Opus 5.5's).
    got = owners("Opus 5.5 is cheaper than Fable 5.2 at $4 per million input tokens.")
    assert got == [("pricing-input", 4.0, "opus-5-5", "unknown")], got
    # ...whereas after "against"/"ahead of", "at" is the baseline's own.
    got = owners("Fable 5.2 costs $8 per million input tokens, ahead of Opus 5.5 at "
                 "$4 per million input tokens.")
    assert got == [("pricing-input", 8.0, "fable-5-2", "unknown"),
                   ("pricing-input", 4.0, "opus-5-5", "unknown")], got


@test("fresh phrasings: conflicts only on a complete identity, both directions (v1.8.8 r3)")
def _():
    # Ten two-document cases written for r3, not copied from any
    # packet. r2 reported a conflict on all five "none" cases;
    # the three "conflict" cases are positive controls that must keep
    # firing so a blanket suppressor cannot pass.
    cases = [
        ("align", "Qwen4-Max costs $1.50 per million input tokens.",
                  "Qwen 4 Max costs $1.50 per million input tokens."),
        ("none", "Qwen4-Max now costs $1.50 per million input tokens.",
                 "Qwen3.8-Max now costs $2 per million input tokens."),
        ("none", "The new Qwen model costs $1.50 per million input tokens today.",
                 "Qwen4-Max costs $2 per million input tokens today."),
        ("conflict", "Qwen4-Max costs $1.50 per million input tokens today.",
                     "Qwen4-Max currently costs $2 per million input tokens."),
        ("none", "Qwen4-Max scored 51.7% on Humanity's Last Exam.",
                 "On HLE, Qwen 3.6 27B scored 24.0%."),
        ("conflict", "Qwen4-Max scored 51.7% on Humanity's Last Exam.",
                     "On HLE, Qwen4-Max scored 48.2%."),
        ("none", "Qwen4-Max beats Qwen3.8-Max with 35% fewer tokens.",
                 "Qwen3.8-Max used 50% fewer tokens than Qwen 3.5."),
        ("conflict", "Qwen4-Max cut average task costs by 62%.",
                     "Qwen4-Max cut average task costs by 45%."),
        ("not-conflict", "Qwen4-Max cut average task costs by 62%.",
                         "Qwen4-Max cut average task costs by around 60%."),
        ("none", "Alibaba considers this a 62 percent reduction in task costs.",
                 "Work360 covers 721 specialized fields of tasks."),
    ]
    for want, a, b in cases:
        d = client.post("/api/v1/compare", json={"documents": [
            {"text": a, "source_name": "A"}, {"text": b, "source_name": "B"}]}).get_json()
        got = "conflict" if d["conflicts"] else ("align" if d["alignment"] else "none")
        ok = got == want or (want == "not-conflict" and got != "conflict")
        assert ok, f"want {want}, got {got}: {a!r} / {b!r} -> {d['conflicts'] or d['alignment']}"


@test("model tokens keep tier and size in the identity (v1.8.8 r3)")
def _():
    from signal_compare import _MODEL_TOKEN_RE, _model_slug
    def slugs(t):
        return [_model_slug(m.group(1)) for m in _MODEL_TOKEN_RE.finditer(t)]
    assert slugs("Qwen 3.8 Max and Qwen 3.8 27B") == ["qwen-3-8-max", "qwen-3-8-27b"], \
        "different Qwen 3.8 models must not share one slug"
    assert slugs("Qwen4-Max") == slugs("Qwen 4 Max") == ["qwen-4-max"]
    assert slugs("GPT6.1-Sol") == slugs("GPT-6.1 Sol") == ["gpt-6-1-sol"]
    assert slugs("GPT-6.1 Terra") == ["gpt-6-1-terra"]
    assert slugs("Pro users on the Max plan") == [], "a tier word alone is not a model"


@test("generic extraction reads spelled-out percent and percentage ranges (v1.8.8 r3)")
def _():
    claims = extract_generic_quantitative_claims(
        "Alibaba considers this a 62 percent reduction in task costs.", "A", [])
    c62 = [c for c in claims if c["numeric_value"] == 62.0]
    assert c62 and c62[0]["unit"] == "percent" and "task_count" not in c62[0]["entity_key"], \
        f"62 percent must be a percentage, got {claims}"
    claims = extract_generic_quantitative_claims(
        "Qwen4-Max uses 24 to 35% fewer tokens than its predecessor.", "A", [])
    units = {c["numeric_value"]: (c["unit"], c["subject"]) for c in claims}
    assert units.get(24.0) == ("percent", "qwen-4-max") and units.get(35.0) == ("percent", "qwen-4-max"), \
        f"both range endpoints are Qwen4-Max percentages, got {units}"
    claims = extract_generic_quantitative_claims(
        "The gap widened by 16 percentage points.", "A", [])
    assert all(c["unit"] != "percent" or c["numeric_value"] != 16.0 or "point" in c["entity_key"]
               for c in claims), f"percentage points are not a plain percent, got {claims}"


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
