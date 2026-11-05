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
  "flag_absent"              : no flag contains this substring (case-insensitive)
  "questions_absent"         : no question contains this substring (case-insensitive)
  "questions_contains"       : at least one question contains this substring (v1.8.1)
  "genre_in"                 : source_genre is one of these values (Phase 1.2)
  "source_type_in"           : source_type is one of these values (Phase 1.2)
  "source_type_confidence_max": source_type_confidence <= threshold (Phase 1.2)
  "signal_brief_min"         : {brief_key: min_count} (Phase 1.2)
  "evidence_shape"           : {field: expected_value} (Phase 1.2)
  "custody_breakdown"        : {field: expected_value} (v1.8.1)

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
            # Arch design note 2026-11-01: a clean primary now earns
            # use_as_anchor rather than verify_primary.
            "action_in":     ["use_as_anchor"],
            "warning_in":    ["clean"],
            "score_at_least": {"overall_confidence": 0.65, "source_custody": 0.70, "evidence": 0.30},
            "score_at_most":  {"pressure": 0.30, "manipulation_risk": 0.35},
            "evidence":       {"anchored_at_least": 3},
            # Sieve gap 2026-11-02: a verified primary feed must not get
            # the generic "find a primary" question.
            "questions_absent": ["closest primary source"],
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

    # ── Phase 1.6 fixtures — financial genre splitting ─────────────────────
    "financial_schwab_market_update.txt": {
        # Brokerage daily snapshot: many tickers, numeric density, OTC caveats.
        # Opening sentence says indexes "retreated early" but reported figures
        # are positive — should trigger timeframe-mismatch in missing_receipts.
        "source_type": "auto",
        "checks": {
            "action_in":      ["treat_as_lead", "seek_receipts"],
            "genre_in":       ["financial_market_snapshot"],
            "source_type_in": ["secondary_market_article", "secondary"],
            "signal_brief_min": {"key_signals": 2, "missing_receipts": 2},
            "evidence_shape":   {"local_numeric_anchors": "high"},
        },
    },
    "financial_trowe_macro_update.txt": {
        # Multi-region macro weekly: CPI/PPI/PMI + Europe/Japan/China/EM sections.
        # Should classify as global_macro_market_update, not financial_market_snapshot.
        "source_type": "auto",
        "checks": {
            "action_in":      ["treat_as_lead", "seek_receipts"],
            "genre_in":       ["global_macro_market_update"],
            "source_type_in": ["secondary_market_article", "secondary"],
            "signal_brief_min": {"key_signals": 3, "missing_receipts": 2},
        },
    },
    "financial_blackrock_strategy.txt": {
        # Asset-manager allocation thesis: overweight/underweight language,
        # tactical/strategic positioning, forward-looking projections.
        # Should classify as investment_strategy_commentary and trigger house-view flag.
        "source_type": "auto",
        "checks": {
            "action_in":    ["treat_as_lead", "seek_receipts"],
            "genre_in":     ["investment_strategy_commentary"],
            "flag_contains": ["house view"],
            "score_at_most": {"pressure": 0.25},
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
    "primary_abstract_auto.txt": {
        # Textbook trial abstract in AUTO mode (no declared source type).
        # Regression fixture for the 2026-10-22 calibration fix: before the
        # fix, verify_primary was unreachable without a declared type even
        # with custody ~0.9, zero pressure and full anchoring (n, CI, p, DOI,
        # trial registration). Pinned issue #1.
        # Arch design note 2026-11-01: this textbook-grade primary now
        # earns use_as_anchor rather than verify_primary.
        "source_type": "auto",
        "checks": {
            "action_in":     ["use_as_anchor"],
            "warning_in":    ["clean"],
            "score_at_least": {"source_custody": 0.85, "evidence": 0.30},
            "score_at_most":  {"pressure": 0.30, "manipulation_risk": 0.35},
            "evidence":       {"anchored_at_least": 3},
            # Sieve gap 2026-11-02: custody-adaptive questions.
            "questions_absent": ["closest primary source"],
        },
    },

    # ── Sieve gap 2026-11-02: first-party announcement on a blog URL ─────
    "primary_blog_announcement.txt": {
        # A first-person announcement hosted on a /blog/ URL, declared
        # primary. The URL token alone must not downgrade it to secondary:
        # the publisher speaking about its own policy IS a primary source.
        # The exemption holds the TYPE at primary (and the adaptive
        # questions follow); the ACTION stays treat_as_lead because the
        # language guardrail holds a thin announcement at lead — type and
        # action are separate axes, and the guardrail is doing its job.
        "source_type": "primary",
        "source_url": "https://example.com/blog/2026/10/shared-ai-training-control",
        "checks": {
            "action_in":     ["treat_as_lead"],
            "source_type_in": ["primary"],
            "flag_absent":   ["conflicts with"],
            "questions_absent": ["closest primary source"],
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
    "x_news_receipted_post.txt": {
        # X/Twitter breaking-news style post with timestamp and off-platform receipt.
        # Should be useful as a lead, but not upgraded to verify_primary.
        "source_type": "auto",
        "source_url": "https://x.com/example/status/1234567890",
        "checks": {
            "action_in": ["treat_as_lead"],
            "genre_in": ["x_news_post"],
            "source_type_in": ["social"],
            "score_at_most": {"overall_confidence": 0.62},
            "flag_contains": ["X news post"],
        },
    },
    "x_news_rumor_post.txt": {
        # Rumor-chain + engagement-bait post should demand receipts.
        "source_type": "auto",
        "source_url": "https://x.com/example/status/9999999999",
        "checks": {
            "action_in": ["seek_receipts", "drop", "reject"],
            "genre_in": ["x_news_post"],
            "source_type_in": ["social"],
            "score_at_most": {"overall_confidence": 0.42},
            "flag_contains": ["X news post"],
        },
    },

    # ── v1.8.0 fixtures (Arch review 2026-11-04) ───────────────────────────
    "primary_blog_contraction.txt": {
        # Same first-party announcement as primary_blog_announcement, but
        # contracted: "Today, we're announcing our plans..." The v1.7.0
        # announcement pattern required whitespace after "we" and missed
        # every contracted announcement — the Nov 4 Google blog downgrade.
        # Ownership + origin, not URL vocabulary, decides the exemption.
        "source_type": "primary",
        "source_url": "https://example.com/blog/2026/10/shared-ai-training-control",
        "checks": {
            "action_in":     ["treat_as_lead"],
            "source_type_in": ["primary"],
            "flag_absent":   ["conflicts with"],
            "questions_absent": ["closest primary source"],
        },
    },
    "primary_discussing_others.txt": {
        # First-party channel discussing SOMEBODY ELSE'S research, declared
        # primary. Arch: a Google blog discussing another party's research
        # is not primary evidence for that research. Third-party attribution
        # ("according to", "a study by", "researchers found") vetoes the
        # exemption, so the blog URL's article genre downgrades it.
        "source_type": "primary",
        "source_url": "https://example.com/blog/2026/10/research-roundup",
        "checks": {
            "source_type_in": ["secondary"],
            "flag_contains": ["first-party exemption did not apply"],
        },
    },
    "inferred_primary_no_evidence.txt": {
        # Dealroom case (Nov 3 gap 2): an ordinary article whose text
        # mentions methodology/sample size, declared "article" (form).
        # The old inference reported PRIMARY with a contradictory
        # "trusted at low confidence" flag. The gate drops the inference
        # to secondary and flags it with Arch's custody-uncertain wording.
        "source_type": "article",
        "checks": {
            "source_type_in": ["secondary"],
            "flag_contains": ["Source custody uncertain"],
        },
    },
    "x_post_genuine_text.txt": {
        # A genuine X post fed as bare text (no URL): short, @handle,
        # #tag, breaking cue. Post STRUCTURE is the evidence — this must
        # still classify as x_news_post after the mention-only fix.
        "source_type": "auto",
        "checks": {
            "action_in": ["treat_as_lead"],
            "genre_in": ["x_news_post"],
            "source_type_in": ["social"],
            "flag_contains": ["X news post"],
        },
    },
    "article_mentioning_social.txt": {
        # TechPolicy.Press case (Nov 4 addendum): a long-form analysis
        # that mentions social media ("follow standardized crawler
        # instructions") and carries news cues ("Reports"). Mentioning a
        # platform is not post structure — it must NOT become x_news_post.
        "source_type": "auto",
        "checks": {
            "genre_in": ["generic_article"],
            "flag_absent": ["X news post"],
        },
    },
    "primary_announcement_with_rumor.txt": {
        # NEGATIVE anchor-gate test (Arch: gates preserved unchanged):
        # a declared primary with announcement voice BUT anonymous/rumor
        # sourcing language. The anchor predicate's provenance bar
        # (st_confidence >= 0.70, no rumor language) must keep this out
        # of use_as_anchor whatever else changes in v1.8.0.
        "source_type": "primary",
        "checks": {
            "action_in": ["treat_as_lead", "verify_primary", "seek_receipts", "drop", "reject"],
            "flag_contains": ["custody"],
        },
    },

    # ── v1.8.1 fixtures (Arch review 2026-11-05) ───────────────────────────
    "primary_hf_model_card.txt": {
        # Nov 5 gap 1: Baidu's own model card hosted on Hugging Face,
        # declared primary. A vendor's own release artifact does not
        # stop being first-party because another platform hosts it:
        # the org slug (baidu) + authorship ("Baidu ERNIE Team") +
        # release metadata (model card, license, parameters) establish
        # ownership per Arch's hierarchy. The old inference said
        # secondary ("results reported alongside...") and the
        # contradiction flag discounted custody to 0.40.
        "source_type": "primary",
        "source_url": "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base",
        "checks": {
            "source_type_in": ["primary"],
            "flag_absent":   ["contradicts"],
            "questions_absent": ["closest primary source", "evidence would establish"],
            "signal_brief_min": {"key_signals": 1},
            "custody_breakdown": {"primary_source_present": True},
        },
    },
    "primary_claims_custody_uncertain.txt": {
        # Nov 5 gap 4 + gap 1 sentinel: declared primary, but the text
        # reports on somebody else's release ("according to reports",
        # "sources say") and carries no ownership evidence. The
        # two-stage question system must ask what evidence would
        # establish authorship — never "where is the original source?"
        # for a doc claiming to BE it. And because the classifier made
        # a consequential custody change with no supporting cue, the
        # signal brief must say so (the no-cue sentinel) instead of
        # returning an empty key_signals list.
        "source_type": "primary",
        "checks": {
            "questions_contains": ["evidence would establish"],
            "questions_absent": ["closest primary source"],
            "flag_contains": ["contradicts"],
            "signal_brief_min": {"key_signals": 1},
        },
    },
    "primary_hf_with_rumor.txt": {
        # NEGATIVE anchor-gate test (Arch: the hosted-artifact
        # exemption must never bypass the complete anchor predicate):
        # a hosted release artifact WITH anonymous/rumor sourcing
        # language. Ownership evidence does not cure a provenance
        # failure — this must stay out of use_as_anchor.
        "source_type": "primary",
        "source_url": "https://huggingface.co/baidu/ERNIE-5.5-Flash-Base",
        "checks": {
            "action_in": ["treat_as_lead", "verify_primary", "seek_receipts", "drop", "reject"],
            "flag_contains": ["custody"],
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

    for needle in checks.get("flag_absent", []):
        if any(needle.lower() in f.lower() for f in result["flags"]):
            fails.append(
                f"flags: unexpected flag contains {needle!r}. Actual flags: {result['flags']}"
            )

    for needle in checks.get("questions_absent", []):
        if any(needle.lower() in q.lower() for q in result.get("questions_to_ask", [])):
            fails.append(
                f"questions: unexpected question contains {needle!r}. "
                f"Actual questions: {result.get('questions_to_ask', [])}"
            )

    # v1.8.1 check: at least one question contains this substring
    # (two-stage custody-aware questions — Arch review 2026-11-05).
    for needle in checks.get("questions_contains", []):
        if not any(needle.lower() in q.lower() for q in result.get("questions_to_ask", [])):
            fails.append(
                f"questions: no question contains {needle!r}. "
                f"Actual questions: {result.get('questions_to_ask', [])}"
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

    # v1.8.1 check: custody_breakdown field equality (same shape as
    # evidence_shape above).
    cb = result.get("custody_breakdown", {})
    for field, expected in checks.get("custody_breakdown", {}).items():
        got = cb.get(field)
        if got != expected:
            fails.append(
                f"custody_breakdown.{field}: got {got!r}, expected {expected!r}"
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
            source_url=spec.get("source_url", ""),
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
