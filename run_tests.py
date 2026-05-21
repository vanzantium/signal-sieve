#!/usr/bin/env python3
"""
run_tests.py -- fixture-based test runner for signal_sieve.

Each fixture in ./tests/*.txt is paired with declarative expectations below.
A check is one of:
  "action_in"                : recommended_action is one of these values
  "warning_in"               : custody_warning_level is one of these values
  "score_at_most"            : scores[key] <= threshold
  "score_at_least"           : scores[key] >= threshold
  "flag_contains"            : at least one flag contains this substring (case-insensitive)
  "evidence"                 : {"anchored_at_least": N} or {"unanchored_at_least": N}
  "genre_in"                 : source_genre is one of these values (Phase 1.2)
  "source_type_in"           : source_type is one of these values (Phase 1.2)
  "source_type_confidence_max": source_type_confidence <= threshold (Phase 1.2)
  "signal_brief_min"         : {brief_key: min_count} (Phase 1.2)
  "evidence_shape"           : {field: expected_value} (Phase 1.2)

A test passes only when ALL checks pass.

Exit code is the number of failed tests (0 on full pass).
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from signal_sieve import analyze  # noqa: E402

TESTS_DIR = SCRIPT_DIR / "tests"

# fixture filename -> {"source_type": str, "checks": dict}
EXPECTATIONS = {
    # ── original 7 fixtures (unchanged) ────────────────────────────────────
    "hype_social.txt": {
        "source_type": "social",
        "checks": {
            "action_in": ["seek_receipts", "drop", "reject"],
            "warning_in": ["weak", "watch"],
            "score_at_least": {"pressure": 0.50, "manipulation_risk": 0.40},
            "score_at_most":  {"overall_confidence": 0.30},
            "flag_contains":  ["pressure"],
        },
    },
    "primary_clean.txt": {
        "source_type": "primary",
        "checks": {
            "action_in":     ["verify_primary"],
            "warning_in":    ["clean"],
            "score_at_least": {"overall_confidence": 0.65, "source_custody": 0.70, "evidence": 0.30},
            "score_at_most":  {"pressure": 0.30, "manipulation_risk": 0.35},
            "evidence":       {"anchored_at_least": 3},
        },
    },
    "primary_laundered.txt": {
        "source_type": "primary",
        "checks": {
            "action_in":     ["drop", "reject", "seek_receipts"],
            "score_at_most":  {"overall_confidence": 0.45},
            "score_at_least": {"certainty_bias": 0.50, "certainty_to_evidence_gap": 0.20},
            "flag_contains":  ["certainty"],
        },
    },
    "anonymous_claim.txt": {
        "source_type": "secondary",
        "checks": {
            "action_in":    ["seek_receipts", "treat_as_lead", "drop"],
            "warning_in":   ["weak", "watch", "mismatch"],
            "score_at_most": {"overall_confidence": 0.50, "source_custody": 0.65},
            "flag_contains": ["custody"],
        },
    },
    "short_specific_claim.txt": {
        "source_type": "primary",
        "checks": {
            "score_at_most": {"overall_confidence": 0.55},
            "flag_contains": ["short"],
        },
    },
    "attack_language.txt": {
        "source_type": "unknown",
        "checks": {
            "action_in":     ["reject"],
            "flag_contains":  ["attack"],
            "score_at_least": {"manipulation_risk": 0.50},
            "score_at_most":  {"overall_confidence": 0.30},
        },
    },
    "hedged_opinion.txt": {
        "source_type": "opinion",
        "checks": {
            "action_in":    ["treat_as_lead", "seek_receipts"],
            "score_at_most": {"certainty_bias": 0.55, "pressure": 0.40, "certainty_to_evidence_gap": 0.30},
        },
    },

    # ── Phase 1.2 fixtures ─────────────────────────────────────────────────
    "financial_market_snapshot.txt": {
        "source_type": "auto",
        "checks": {
            "action_in":     ["treat_as_lead", "seek_receipts"],
            "genre_in":      ["financial_market_snapshot"],
            "source_type_in": ["secondary_market_article", "secondary"],
            "signal_brief_min": {"key_signals": 2, "missing_receipts": 2},
            "evidence_shape":   {"local_numeric_anchors": "high"},
        },
    },
    "press_release_official.txt": {
        "source_type": "auto",
        "checks": {
            "action_in":  ["verify_primary", "treat_as_lead"],
            "genre_in":   ["official_release"],
            "source_type_in": ["official", "official_release"],
        },
    },
    "opinion_with_numbers.txt": {
        "source_type": "auto",
        "checks": {
            "action_in":  ["treat_as_lead", "seek_receipts"],
            "genre_in":   ["opinion_article", "generic_article"],
            "score_at_most": {"certainty_bias": 0.70},
        },
    },

    # ── Phase 1.4 fixtures ─────────────────────────────────────────────────
    "first_hand_witness.txt": {
        # First-person account of an undisclosed corporate announcement.
        # Should score high on source_custody (first_hand weight), low pressure.
        # st_confidence may be < 0.50 (text pattern reads partially official)
        # → custody_warning_level may be mismatch/watch, which is acceptable.
        "source_type": "first_hand",
        "checks": {
            "action_in":     ["verify_primary", "treat_as_lead", "seek_receipts"],
            "warning_in":    ["clean", "watch", "mismatch"],
            "score_at_least": {"source_custody": 0.60},
            "score_at_most":  {"pressure": 0.15, "manipulation_risk": 0.30},
        },
    },
    "tertiary_wiki.txt": {
        # Wikipedia-style background summary — low specificity, no strong attribution chain.
        # Should score low confidence, low pressure, no strong custody.
        "source_type": "auto",
        "checks": {
            "action_in":    ["treat_as_lead", "seek_receipts"],
            "score_at_most": {"overall_confidence": 0.35, "pressure": 0.05},
        },
    },
}

GREEN  = "\033[32m" if sys.stdout.isatty() else ""
RED    = "\033[31m" if sys.stdout.isatty() else ""
YELLOW = "\033[33m" if sys.stdout.isatty() else ""
DIM    = "\033[2m"  if sys.stdout.isatty() else ""
RESET  = "\033[0m"  if sys.stdout.isatty() else ""


def check_one(result: dict, spec: dict) -> list[str]:
    fails: list[str] = []
    scores = result["scores"]
    checks = spec["checks"]

    if "action_in" in checks and result["recommended_action"] not in checks["action_in"]:
        fails.append(
            f"action: got {result['recommended_action']!r}, expected one of {checks['action_in']}"
        )

    if "warning_in" in checks and result["custody_warning_level"] not in checks["warning_in"]:
        fails.append(
            f"warning_level: got {result['custody_warning_level']!r}, "
            f"expected one of {checks['warning_in']}"
        )

    for key, threshold in checks.get("score_at_least", {}).items():
        if scores.get(key, 0) < threshold:
            fails.append(
                f"score {key}: got {scores.get(key, 0):.3f}, expected >= {threshold}"
            )

    for key, threshold in checks.get("score_at_most", {}).items():
        if scores.get(key, 0) > threshold:
            fails.append(
                f"score {key}: got {scores.get(key, 0):.3f}, expected <= {threshold}"
            )

    for needle in checks.get("flag_contains", []):
        if not any(needle.lower() in f.lower() for f in result["flags"]):
            fails.append(
                f"flags: no flag contains {needle!r}. Actual flags: {result['flags']}"
            )

    ev = checks.get("evidence", {})
    if "anchored_at_least" in ev and result["evidence_breakdown"]["anchored"] < ev["anchored_at_least"]:
        fails.append(
            f"evidence anchored: got {result['evidence_breakdown']['anchored']}, "
            f"expected >= {ev['anchored_at_least']}"
        )
    if "unanchored_at_least" in ev and result["evidence_breakdown"]["unanchored"] < ev["unanchored_at_least"]:
        fails.append(
            f"evidence unanchored: got {result['evidence_breakdown']['unanchored']}, "
            f"expected >= {ev['unanchored_at_least']}"
        )

    # ── Phase 1.2 checks ──────────────────────────────────────────────────
    if "genre_in" in checks:
        got_genre = result.get("source_genre", "")
        if got_genre not in checks["genre_in"]:
            fails.append(
                f"source_genre: got {got_genre!r}, expected one of {checks['genre_in']}"
            )

    if "source_type_in" in checks and result["source_type"] not in checks["source_type_in"]:
        fails.append(
            f"source_type: got {result['source_type']!r}, "
            f"expected one of {checks['source_type_in']}"
        )

    if "source_type_confidence_max" in checks:
        got_conf = result.get("source_type_confidence", 1.0)
        if got_conf > checks["source_type_confidence_max"]:
            fails.append(
                f"source_type_confidence: got {got_conf:.3f}, "
                f"expected <= {checks['source_type_confidence_max']}"
            )

    sb = result.get("signal_brief", {})
    for brief_key, min_count in checks.get("signal_brief_min", {}).items():
        actual = len(sb.get(brief_key, []))
        if actual < min_count:
            fails.append(
                f"signal_brief.{brief_key}: got {actual} items, expected >= {min_count}"
            )

    es = result.get("evidence_shape", {})
    for field, expected in checks.get("evidence_shape", {}).items():
        got = es.get(field)
        if got != expected:
            fails.append(
                f"evidence_shape.{field}: got {got!r}, expected {expected!r}"
            )

    return fails


def run() -> int:
    if not TESTS_DIR.exists():
        print(f"{RED}tests/ directory not found at {TESTS_DIR}{RESET}")
        return 1

    failed = 0
    total  = 0
    print(f"{DIM}signal_sieve test corpus{RESET}\n")

    for fname, spec in EXPECTATIONS.items():
        total += 1
        path = TESTS_DIR / fname
        if not path.exists():
            print(f"{YELLOW}MISSING{RESET} {fname}")
            failed += 1
            continue

        result = analyze(
            path.read_text(encoding="utf-8"),
            source_type=spec["source_type"],
            source_name=f"fixture: {fname}",
        )
        fails = check_one(result, spec)
        color  = GREEN if not fails else RED
        status = "PASS"  if not fails else "FAIL"
        genre_tag = f" genre={result.get('source_genre','?')}" if result.get("source_genre") else ""
        print(
            f"{color}{status}{RESET}    {fname:36s} -> {result['recommended_action']:14s} "
            f"[{result['custody_warning_level']}] oc={result['scores']['overall_confidence']:.2f}"
            f"{genre_tag}"
        )
        if fails:
            failed += 1
            for f in fails:
                print(f"  {DIM}- {f}{RESET}")
            print(f"  {DIM}triage: {result['triage_summary']}{RESET}")

    print(f"\n{total - failed}/{total} passed")
    if failed:
        print(f"{RED}{failed} failed{RESET}")
    return failed


if __name__ == "__main__":
    sys.exit(run())
