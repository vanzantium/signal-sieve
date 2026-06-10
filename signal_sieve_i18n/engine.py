"""Declared-language lexical risk assessment for Signal Sieve."""

from __future__ import annotations

import importlib
from typing import Dict, Optional

from .lexicons import AVAILABLE
from .structural_core import anchor_check, detect_script, extract_structural


_PACK_CACHE: Dict[str, object] = {}
_RANK = {"verify_primary": 0, "treat_as_lead": 1, "seek_receipts": 2, "drop": 3, "reject": 4}


def normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None or language == "":
        return None
    if not isinstance(language, str):
        return "invalid"
    return language.strip().lower().split("-")[0].split("_")[0]


def _load_pack(code: str):
    if code not in _PACK_CACHE:
        module = importlib.import_module(f".lexicons.{code}", package=__package__)
        _PACK_CACHE[code] = module.PACK
    return _PACK_CACHE[code]


def _route(text: str, declared: Optional[str]) -> Dict:
    script = detect_script(text)
    code = normalize_language(declared)
    if code in AVAILABLE:
        pack = _load_pack(code)
        route = {
            "script": script,
            "lexicon": code,
            "mode": "full",
            "calibration": pack.calibration,
        }
        if script == "mixed":
            route["warning"] = (
                f"mixed_script: declared '{code}' lexicon applied to mixed-script text."
            )
        elif script not in ("unknown", pack.script):
            route["warning"] = (
                f"script_mismatch: declared '{code}' uses '{pack.script}', "
                f"but the text appears '{script}'."
            )
        return route
    if code:
        return {
            "script": script,
            "lexicon": None,
            "mode": "structural_only",
            "requested_language": code,
            "available_languages": sorted(AVAILABLE),
            "note": (
                f"Unsupported language '{code}'. Verdict uses structural features only "
                "and cannot earn verify_primary."
            ),
        }
    return {
        "script": script,
        "lexicon": None,
        "mode": "structural_only",
        "available_languages": sorted(AVAILABLE),
        "note": "No language declared. Declare language for lexical scoring.",
    }


def _score(text: str, pack) -> Dict[str, float]:
    structural = extract_structural(text)
    if pack is None:
        return {
            "anchor_density": structural.anchor_density,
            "has_strong_custody": float(structural.has_strong_custody),
            "certainty_bias": 0.0,
            "hype": 0.0,
            "attack": 0.0,
            "hedge": 0.0,
            "anonymous_sourcing": 0.0,
            "evidence_anchored": 0.0,
            "evidence_unanchored": 0.0,
            "certainty_to_evidence_gap": 0.0,
            "lexical_available": 0.0,
        }
    certainty = pack.find("certainty", text)
    hedge = pack.find("hedge", text)
    hype = pack.find("hype", text)
    attack = pack.find("attack", text)
    evidence = pack.find("evidence", text)
    anonymous = pack.find("anonymous", text)
    anchored, unanchored = anchor_check(
        text,
        evidence,
        extra_hooks=pack.find_months(text),
        exclude_spans=certainty,
    )
    sentence_count = max(1, structural.n_sentences)
    certainty_rate = len(certainty) / sentence_count
    evidence_strength = anchored + structural.n_identifiers + structural.n_urls
    gap = max(0.0, certainty_rate - evidence_strength / sentence_count)
    return {
        "anchor_density": structural.anchor_density,
        "has_strong_custody": float(structural.has_strong_custody),
        "certainty_bias": round(min(1.0, certainty_rate), 3),
        "hype": round(min(1.0, len(hype) / sentence_count), 3),
        "attack": round(min(1.0, len(attack) / sentence_count), 3),
        "hedge": round(min(1.0, len(hedge) / sentence_count), 3),
        "anonymous_sourcing": round(min(1.0, len(anonymous) / sentence_count), 3),
        "evidence_anchored": float(anchored),
        "evidence_unanchored": float(unanchored),
        "certainty_to_evidence_gap": round(min(1.0, gap), 3),
        "lexical_available": 1.0,
    }


def _verdict(scores: Dict[str, float], mode: str) -> str:
    if scores["attack"] > 0.15 and scores["certainty_bias"] > 0.2:
        action = "reject"
    elif scores["certainty_to_evidence_gap"] > 0.4:
        action = "seek_receipts"
    elif scores["hype"] > 0.2 and scores["evidence_anchored"] == 0:
        action = "drop"
    elif scores["anonymous_sourcing"] > 0.15 and scores["evidence_anchored"] == 0:
        action = "seek_receipts"
    elif scores["has_strong_custody"] and scores["certainty_to_evidence_gap"] < 0.15:
        action = "verify_primary"
    else:
        action = "treat_as_lead"
    if action == "verify_primary" and scores["anonymous_sourcing"] > 0.15:
        action = "seek_receipts"
    if mode == "structural_only" and action == "verify_primary":
        action = "treat_as_lead"
    return action


def analyze(text: str, language: Optional[str] = None) -> Dict:
    route = _route(text, language)
    pack = _load_pack(route["lexicon"]) if route["lexicon"] else None
    scores = _score(text, pack)
    action = _verdict(scores, route["mode"])

    worst = None
    capped = False
    paragraphs = [paragraph for paragraph in text.splitlines() if paragraph.strip()]
    if len(paragraphs) > 1 and pack is not None:
        paragraph_results = []
        for paragraph in paragraphs:
            paragraph_scores = _score(paragraph, pack)
            paragraph_results.append(
                (
                    _verdict(paragraph_scores, route["mode"]),
                    paragraph_scores["certainty_to_evidence_gap"],
                    paragraph[:120],
                )
            )
        worst = max(paragraph_results, key=lambda item: (_RANK[item[0]], item[1]))
        if worst[0] in ("reject", "drop") and _RANK[action] < _RANK["seek_receipts"]:
            action = "seek_receipts"
            capped = True

    note = "Heuristic language/structure verdict, not truth."
    if route["mode"] == "structural_only":
        note += " Lexical risk was unread, so strong custody cannot earn verify_primary."
    if route.get("calibration") == "starter_partial":
        note += " This language pack is a starter lexicon and is not corpus-calibrated."
    if route.get("warning"):
        note += " " + route["warning"]
    if capped:
        note += " A reject/drop-grade paragraph capped the document at seek_receipts."

    result = {
        "recommended_action": action,
        "language": route,
        "scores": scores,
        "ai_instruction": note,
    }
    if worst:
        result["paragraph_worst"] = {
            "action": worst[0],
            "gap": worst[1],
            "preview": worst[2],
            "capped_document_verdict": capped,
        }
    return result
