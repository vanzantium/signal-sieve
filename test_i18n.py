#!/usr/bin/env python3
"""Contract checks for declared-language and structural-only guardrails."""

from __future__ import annotations

import sys

from signal_sieve import analyze
from signal_sieve_i18n import analyze as analyze_language


def run() -> int:
    checks = []

    junk = analyze_language(
        "SHOCKING breakthrough! This miracle cure is proven to work. "
        "They don't want you to know. Wake up, sheeple!",
        language="en",
    )
    checks.append(("English attack/certainty junk is rejected", junk["recommended_action"] == "reject"))

    unsupported = analyze(
        "Selon l'étude, voir doi:10.1038/s41586-023-00001.",
        language="fr",
        source_type="primary",
    )
    checks.append((
        "Unsupported language cannot earn verify_primary",
        unsupported["recommended_action"] != "verify_primary"
        and unsupported["language"]["mode"] == "structural_only",
    ))

    spanish = analyze(
        "Fuentes anónimas dicen que está comprobado, pero no presentan datos.",
        language="es",
    )
    checks.append((
        "Spanish starter pack is explicit and active",
        spanish["language"]["lexicon"] == "es"
        and spanish["language"]["calibration"] == "starter_partial"
        and bool(spanish["language_note"]),
    ))
    checks.append((
        "Language downgrade updates the public triage summary",
        spanish["recommended_action"] == "seek_receipts"
        and spanish["triage_summary"].startswith("Language-aware guardrail selected"),
    ))

    poison = analyze(
        "According to doi:10.1038/test, researchers measured a 47% reduction.\n"
        "SHOCKING miracle proven! They don't want you to know. Wake up, sheeple!",
        language="en",
        source_type="primary",
    )
    worst = poison["language_assessment"].get("paragraph_worst", {})
    checks.append((
        "Risky paragraph caps a clean wrapper",
        poison["recommended_action"] in {"seek_receipts", "drop", "reject"}
        and worst.get("capped_document_verdict") is True,
    ))

    invalid = analyze("A short claim with no receipts.", language=42)
    checks.append((
        "Invalid language type falls back safely",
        invalid["language"]["mode"] == "structural_only"
        and invalid["language"]["requested_language"] == "invalid",
    ))

    failures = 0
    print("Signal Sieve language contract\n")
    for label, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
        failures += not passed
    print(f"\n{len(checks) - failures}/{len(checks)} passed")
    return int(failures)


if __name__ == "__main__":
    sys.exit(run())
