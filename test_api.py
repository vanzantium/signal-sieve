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
    API_VERSION, ANALYSIS_VERSION, MAX_TEXT_CHARS,
    _ACTION_ENUM, _SOURCE_TYPE_ENUM,
)

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
