from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

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


# ── URL fetcher ───────────────────────────────────────────────────────────────

# Obvious single-chunk nav/UI junk (exact-match, case-insensitive)
_JUNK_CHUNK_RE = re.compile(
    r"^(?:skip\s+to\s+(?:main\s+)?content|menu|search|home|subscribe|log\s?in|"
    r"sign\s+in|share|tweet|facebook|linkedin|pinterest|email this|print|"
    r"advertisement|sponsored|related articles?|you may also like|"
    r"read more|see more|load more|comments?|reply|close|dismiss|"
    r"toggle\s+\w+|back\s+to\s+top|cookie\s+\w+|accept\s+all|"
    r"privacy\s+policy|terms\s+of\s+(?:use|service))$",
    re.I,
)


def _is_junk_chunk(text: str) -> bool:
    if _JUNK_CHUNK_RE.match(text):
        return True
    # Very short ALL-CAPS items (nav labels, e.g. "HOME", "ABOUT US")
    if len(text.split()) <= 3 and text.upper() == text and not re.search(r"\d", text):
        return True
    return False


class _TextExtractor(HTMLParser):
    """Strip HTML to plain text, skipping boilerplate/chrome tags."""
    _SKIP = {"script", "style", "nav", "footer", "head",
             "noscript", "iframe", "aside", "svg", "form", "header"}

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._depth:
            t = data.strip()
            if t and not _is_junk_chunk(t):
                self.chunks.append(t)


def _extract_article_html(raw: str) -> str:
    """Attempt to isolate article/main body HTML before full-page parse.

    Tries (in order): <article>, <main>, then a div with a known content class/id.
    Returns original raw if no container found or extracted block is too short.
    """
    lower = raw.lower()

    for tag in ("article", "main"):
        m = re.search(rf"<{tag}(?:\s[^>]*)?>", lower)
        if m:
            end_marker = f"</{tag}>"
            end_pos = lower.rfind(end_marker)
            if end_pos > m.start() + 200:
                block = raw[m.start(): end_pos + len(end_marker)]
                # Only use if it has meaningful content
                if len(block.split()) >= 80:
                    return block

    # Content-bearing div
    m = re.search(
        r'<div[^>]+(?:class|id)=["\'][^"\']*'
        r"(?:article[-_]?body|entry[-_]?content|post[-_]?content|"
        r"story[-_]?body|field[-_]?body|article__body|post__content|"
        r"main[-_]?content|content[-_]?area)"
        r'[^"\']*["\'][^>]*>',
        raw, re.I,
    )
    if m:
        # Can't reliably find end without full parse; take a generous slice
        return raw[m.start(): m.start() + 60_000]

    return raw


def fetch_url_text(url: str) -> tuple[str, str]:
    """Fetch URL and return (plain_text, error_message).

    Returns up to 12 000 characters of extracted text.
    Limitation: JavaScript-rendered pages (React/SPA) may return little text.
    Works well on static pages, blogs, Wikipedia, press releases, arxiv, SEC.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SignalSieve/1.0 (+https://github.com/vanzantium/signal-sieve)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read(600_000).decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return "", f"Could not fetch URL: {exc.reason}"
    except Exception as exc:
        return "", f"Fetch error: {exc}"

    if "html" in ct or raw.lstrip().startswith("<"):
        # Prefer article/main body extraction to avoid nav/header junk
        html_to_parse = _extract_article_html(raw)
        parser = _TextExtractor()
        parser.feed(html_to_parse)
        text = " ".join(parser.chunks)
        # If we got very little from the article block, fall back to full page
        if len(text.split()) < 80 and html_to_parse is not raw:
            full_parser = _TextExtractor()
            full_parser.feed(raw)
            text = " ".join(full_parser.chunks)
    else:
        text = raw

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 50:
        return "", "Too little text extracted — page may require JavaScript to render."

    return text[:12_000], ""


# ── helpers ───────────────────────────────────────────────────────────────────

def _score(result: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(result.get("scores", {}).get(key, default))
    except (TypeError, ValueError):
        return default


def _render(*, result=None, input_text="", fetch_url="", selected_source_type="auto",
            source_name="", source_url="", error=None, url_error=None):
    kw = dict(
        source_types=SOURCE_TYPES,
        result=result,
        input_text=input_text,
        fetch_url=fetch_url,
        selected_source_type=selected_source_type,
        source_name=source_name,
        source_url=source_url,
        error=error,
        url_error=url_error,
        score=_score,
    )
    if result is not None:
        kw["result_json"] = json.dumps(result, indent=2, ensure_ascii=False)
    return render_template("index.html", **kw)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return _render()


@app.post("/")
def run_sieve():
    fetch_url_val = request.form.get("fetch_url", "").strip()
    text          = request.form.get("text", "")
    source_type   = request.form.get("source_type", "auto")
    source_name   = request.form.get("source_name", "unknown")
    source_url    = request.form.get("source_url", "")
    url_error     = None

    # If a fetch URL was provided, attempt to extract its text
    if fetch_url_val:
        fetched, url_error = fetch_url_text(fetch_url_val)
        if fetched:
            text = fetched
            if not source_url:
                source_url = fetch_url_val
        # If fetch failed, fall through to whatever is in the textarea

    if not text.strip():
        msg = url_error or "Paste some text or enter a URL to fetch."
        return _render(
            input_text=text,
            fetch_url=fetch_url_val,
            selected_source_type=source_type,
            source_name=source_name,
            source_url=source_url,
            error=msg,
        )

    result = analyze(
        text,
        source_type=source_type,
        source_name=source_name or "unknown",
        source_url=source_url or "",
    )

    return _render(
        result=result,
        input_text=text,
        fetch_url=fetch_url_val,
        selected_source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        url_error=url_error,  # show fetch warning even if analysis ran
    )


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}

    # Support optional url field in API too
    text = payload.get("text", "")
    url  = payload.get("url", "")

    if url and not text:
        text, err = fetch_url_text(url)
        if err:
            return jsonify({"error": err}), 400

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Provide a non-empty 'text' or a fetchable 'url'."}), 400

    result = analyze(
        text,
        source_type=payload.get("source_type", "auto"),
        source_name=payload.get("source_name", "unknown"),
        source_url=payload.get("source_url", url or ""),
    )
    return jsonify(result)


@app.get("/health")
def health():
    return {"ok": True, "service": "signal-sieve"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
