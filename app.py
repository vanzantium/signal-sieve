from __future__ import annotations

import json
import os
from flask import Flask, jsonify, render_template, request

from signal_sieve import analyze

app = Flask(__name__)

SOURCE_TYPES = [
    "auto",
    "first_hand",
    "primary",
    "official",
    "expert",
    "secondary",
    "tertiary",
    "anonymous",
    "social",
    "opinion",
    "unknown",
]


def _score(result: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(result.get("scores", {}).get(key, default))
    except (TypeError, ValueError):
        return default


@app.get("/")
def index():
    return render_template(
        "index.html",
        source_types=SOURCE_TYPES,
        result=None,
        input_text="",
        selected_source_type="auto",
        source_name="",
        source_url="",
        error=None,
    )


@app.post("/")
def run_sieve():
    text = request.form.get("text", "")
    source_type = request.form.get("source_type", "auto")
    source_name = request.form.get("source_name", "unknown")
    source_url = request.form.get("source_url", "")

    if not text.strip():
        return render_template(
            "index.html",
            source_types=SOURCE_TYPES,
            result=None,
            input_text=text,
            selected_source_type=source_type,
            source_name=source_name,
            source_url=source_url,
            error="Paste some text first.",
        )

    result = analyze(
        text,
        source_type=source_type,
        source_name=source_name or "unknown",
        source_url=source_url or "",
    )

    return render_template(
        "index.html",
        source_types=SOURCE_TYPES,
        result=result,
        result_json=json.dumps(result, indent=2, ensure_ascii=False),
        input_text=text,
        selected_source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        error=None,
        score=_score,
    )


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing non-empty 'text' field"}), 400

    result = analyze(
        text,
        source_type=payload.get("source_type", "auto"),
        source_name=payload.get("source_name", "unknown"),
        source_url=payload.get("source_url", ""),
    )
    return jsonify(result)


@app.get("/health")
def health():
    return {"ok": True, "service": "signal-sieve"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
