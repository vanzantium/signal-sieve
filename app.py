from __future__ import annotations

import datetime
import json
import os
import re
import time
import uuid
import urllib.error
import urllib.request
from html.parser import HTMLParser

from flask import Flask, jsonify, render_template, request

from signal_sieve import analyze

app = Flask(__name__)

# ── API constants ─────────────────────────────────────────────────────────────
API_VERSION       = "v1"
ANALYSIS_VERSION  = "v1.3.0"
MAX_TEXT_CHARS        = 10_000
MAX_SOURCE_URL_CHARS  = 2_000
MAX_SOURCE_NAME_CHARS = 200
_START_TIME = time.time()

_ACTION_ENUM = [
    "verify_primary", "treat_as_lead", "treat_as_background",
    "seek_receipts", "drop", "reject",
]
_SOURCE_TYPE_ENUM = [
    "auto", "primary", "official", "official_release", "regulatory_filing",
    "expert", "first_hand", "secondary", "secondary_market_article",
    "news_article", "opinion_article", "social_post", "tertiary",
    "anonymous", "unknown",
]
_GENRE_ENUM = [
    "financial_market_snapshot", "official_release", "regulatory_filing",
    "news_article", "opinion_article", "social_post", "generic_article",
]

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


class _MetaExtractor(HTMLParser):
    """Extract title, author, and publisher from HTML meta/OG tags and JSON-LD."""

    def __init__(self) -> None:
        super().__init__()
        self.title: str = ""
        self.author: str = ""
        self.publisher: str = ""
        self._in_title = False
        self._in_ld = False
        self._ld_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        ad = {k.lower(): (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = ad.get("name", "").lower()
            prop = ad.get("property", "").lower()
            content = ad.get("content", "").strip()
            if not content:
                return
            # Author
            if not self.author and name in ("author", "article:author", "byl"):
                self.author = content
            elif not self.author and prop in ("article:author",):
                self.author = content
            # Publisher / site name
            if not self.publisher and prop == "og:site_name":
                self.publisher = content
            elif not self.publisher and name == "publisher":
                self.publisher = content
        elif tag == "script" and ad.get("type", "").lower() == "application/ld+json":
            self._in_ld = True
            self._ld_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        elif tag.lower() == "script" and self._in_ld:
            self._in_ld = False
            self._parse_ld("".join(self._ld_buf))

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()
        elif self._in_ld:
            self._ld_buf.append(data)

    def _parse_ld(self, ld_text: str) -> None:
        try:
            data = json.loads(ld_text)
        except Exception:
            return
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return
        self._absorb_ld(data)
        # Also check @graph array
        for item in data.get("@graph", []):
            if isinstance(item, dict):
                self._absorb_ld(item)

    def _absorb_ld(self, data: dict) -> None:
        if not self.author:
            raw = data.get("author") or data.get("creator")
            if isinstance(raw, list):
                raw = raw[0] if raw else None
            if isinstance(raw, dict):
                self.author = raw.get("name", "").strip()
            elif isinstance(raw, str):
                self.author = raw.strip()
        if not self.publisher:
            raw = data.get("publisher") or data.get("sourceOrganization")
            if isinstance(raw, dict):
                self.publisher = raw.get("name", "").strip()
            elif isinstance(raw, str):
                self.publisher = raw.strip()

    def meta(self) -> dict:
        """Return non-empty extracted fields."""
        return {k: v for k, v in {
            "title":     self.title,
            "author":    self.author,
            "publisher": self.publisher,
        }.items() if v}


def fetch_url_text(url: str) -> tuple[str, str, dict]:
    """Fetch URL and return (plain_text, error_message, page_meta).

    page_meta may contain: title, author, publisher (all optional).
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
        return "", f"Could not fetch URL: {exc.reason}", {}
    except Exception as exc:
        return "", f"Fetch error: {exc}", {}

    page_meta: dict = {}

    if "html" in ct or raw.lstrip().startswith("<"):
        # Extract metadata (title/author/publisher) from the full raw HTML
        meta_parser = _MetaExtractor()
        meta_parser.feed(raw[:120_000])
        page_meta = meta_parser.meta()

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
        return "", "Too little text extracted — page may require JavaScript to render.", page_meta

    return text[:12_000], "", page_meta


# ── API v1 helpers ────────────────────────────────────────────────────────────

_SAFE_USE = (
    "Receipt triage only. Signal Sieve is a heuristic receipt clerk, not a "
    "fact-checker. overall_confidence counts signal-quality markers detected "
    "in the text — it is NOT the probability that claims are true."
)


def _make_request_id() -> str:
    date = datetime.datetime.utcnow().strftime("%Y%m%d")
    uid  = uuid.uuid4().hex[:8]
    return f"req_{date}_{uid}"


def _api_envelope(request_id: str) -> dict:
    return {
        "kind":              "heuristic_signal_triage",
        "verdict_type":      "signal_shape_assessment",
        "safe_use":          _SAFE_USE,
        "api_version":       API_VERSION,
        "analysis_version":  ANALYSIS_VERSION,
        "request_id":        request_id,
    }


def _api_error(status: int, error: str, message: str, request_id: str, **extra):
    body = {"error": error, "message": message, "request_id": request_id, **extra}
    resp = jsonify(body)
    resp.status_code = status
    resp.headers["X-Request-Id"] = request_id
    return resp


def _check_api_key() -> "str | None":
    """Return None if auth passes, error string if it fails."""
    required = os.environ.get("SIGNAL_SIEVE_API_KEY", "").strip()
    if not required:
        return None                                   # public mode — no key needed
    provided = request.headers.get("X-API-Key", "")
    if provided != required:
        return "Missing or invalid API key."
    return None


def _validate_analyze_payload(payload: dict, request_id: str):
    """Return (text, source_type, source_name, source_url, error_response).

    error_response is non-None when validation fails.
    """
    text        = payload.get("text", "")
    source_url  = (payload.get("source_url",  "") or "")[:MAX_SOURCE_URL_CHARS]
    source_name = (payload.get("source_name", "") or "")[:MAX_SOURCE_NAME_CHARS]
    source_type = payload.get("source_type", "auto")

    if not isinstance(text, str) or not text.strip():
        return None, None, None, None, _api_error(
            400, "bad_request", "Missing required field: text", request_id
        )
    if len(text) > MAX_TEXT_CHARS:
        return None, None, None, None, _api_error(
            413, "text_too_long", "Text exceeds maximum length.",
            request_id, max_chars=MAX_TEXT_CHARS,
        )
    return text, source_type, source_name or "unknown", source_url, None


# ── OpenAPI spec ───────────────────────────────────────────────────────────────

_OPENAPI_SPEC: dict = {
    "openapi": "3.0.3",
    "info": {
        "title":       "Signal Sieve API",
        "version":     API_VERSION,
        "description": (
            "Heuristic receipt clerk for signal/noise triage. "
            "Checks whether text has usable signal-quality markers: named sources, "
            "numeric anchors, source custody, attribution, caveats, pressure language. "
            "NOT a fact-checker. overall_confidence measures signal markers, not truth."
        ),
        "contact": {"url": "https://github.com/vanzantium/signal-sieve"},
    },
    "servers": [
        {"url": "https://signal-sieve.onrender.com", "description": "Production"},
    ],
    "components": {
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey", "in": "header", "name": "X-API-Key",
                "description": "Required only when SIGNAL_SIEVE_API_KEY env var is set on the server.",
            }
        },
        "schemas": {
            "AnalyzeRequest": {
                "type": "object", "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string", "maxLength": MAX_TEXT_CHARS,
                        "description": "Text to analyze. Max 10 000 characters.",
                        "example": "Scientists at MIT confirmed a 40% reduction in...",
                    },
                    "source_url": {
                        "type": "string", "maxLength": MAX_SOURCE_URL_CHARS,
                        "description": "Source URL used as context metadata only — not fetched by v1.",
                        "example": "https://example.com/article",
                    },
                    "source_type": {
                        "type": "string", "enum": _SOURCE_TYPE_ENUM, "default": "auto",
                        "description": "Declared source type. 'auto' infers from text and URL.",
                    },
                    "source_name": {
                        "type": "string", "maxLength": MAX_SOURCE_NAME_CHARS,
                        "description": "Publisher or author label.",
                        "example": "Rob Dale / Business of Cannabis",
                    },
                    "include_signal_brief": {
                        "type": "boolean", "default": True,
                        "description": "Include the signal_brief extraction in the response.",
                    },
                },
            },
            "AnalyzeResponse": {
                "type": "object",
                "properties": {
                    "kind":              {"type": "string", "enum": ["heuristic_signal_triage"]},
                    "verdict_type":      {"type": "string", "enum": ["signal_shape_assessment"]},
                    "safe_use":          {"type": "string"},
                    "api_version":       {"type": "string", "example": API_VERSION},
                    "analysis_version":  {"type": "string", "example": ANALYSIS_VERSION},
                    "request_id":        {"type": "string", "example": "req_20260521_a1b2c3d4"},
                    "recommended_action": {
                        "type": "string", "enum": _ACTION_ENUM,
                        "description": (
                            "verify_primary — strong custody shape; still verify key claims. "
                            "treat_as_lead — usable signals; needs receipts. "
                            "treat_as_background — contextual/interpretive framing only. "
                            "seek_receipts — claims outpace evidence. "
                            "drop — noise/hype/low signal. "
                            "reject — attack language or dangerous framing."
                        ),
                    },
                    "overall_confidence": {
                        "type": "number", "minimum": 0, "maximum": 1,
                        "description": (
                            "Heuristic score 0–1: count of signal-quality markers detected. "
                            "NOT the probability that claims are true. "
                            "Alias: receipt_readiness_score."
                        ),
                    },
                    "receipt_readiness_score": {
                        "type": "number", "minimum": 0, "maximum": 1,
                        "description": "Preferred alias for overall_confidence.",
                    },
                    "confidence_band": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 2, "maxItems": 2,
                        "description": (
                            "Heuristic range [low, high]. "
                            "NOT a statistical confidence interval. "
                            "Expresses scoring sensitivity to extraction noise, "
                            "genre mismatch, or missing source metadata."
                        ),
                    },
                    "source_type":   {"type": "string", "enum": _SOURCE_TYPE_ENUM},
                    "source_genre":  {"type": "string", "enum": _GENRE_ENUM},
                    "evidence_shape": {
                        "type": "object",
                        "description": "Structural inventory: numeric anchors, named sources, receipts.",
                    },
                    "signal_brief": {
                        "type": "object",
                        "description": (
                            "Structured extraction with keys: top_movers, key_signals, "
                            "source_caveats, interpretation_or_framing, missing_receipts, "
                            "do_not_pass_forward_as, follow_up_sources."
                        ),
                    },
                    "missing_receipts": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Top-level shortcut to signal_brief.missing_receipts.",
                    },
                    "flags": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Triggered heuristic warning flags.",
                    },
                    "scores": {
                        "type": "object",
                        "description": "Full heuristic score breakdown (all sub-scores).",
                    },
                    "questions_to_ask": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Suggested verification questions.",
                    },
                },
            },
            "ErrorResponse": {
                "type": "object", "required": ["error", "message", "request_id"],
                "properties": {
                    "error": {
                        "type": "string",
                        "enum": [
                            "bad_request", "unauthorized", "text_too_long",
                            "rate_limited", "internal_error",
                        ],
                    },
                    "message":             {"type": "string"},
                    "request_id":          {"type": "string"},
                    "max_chars":           {"type": "integer", "description": "Present on text_too_long."},
                    "retry_after_seconds": {"type": "integer", "description": "Present on rate_limited."},
                },
            },
        },
    },
    "paths": {
        "/health": {
            "get": {
                "summary": "Health check", "operationId": "health",
                "responses": {
                    "200": {
                        "description": "Service healthy",
                        "content": {"application/json": {"schema": {
                            "type": "object",
                            "properties": {
                                "status":           {"type": "string", "example": "ok"},
                                "service":          {"type": "string"},
                                "api_version":      {"type": "string"},
                                "analysis_version": {"type": "string"},
                                "uptime_seconds":   {"type": "integer"},
                            },
                        }}},
                    }
                },
            }
        },
        "/api/v1/analyze": {
            "post": {
                "summary": "Analyze text — structured JSON response",
                "operationId": "analyzeV1",
                "security": [{"ApiKeyAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/AnalyzeRequest"},
                        "examples": {
                            "basic": {
                                "summary": "Minimal request",
                                "value": {"text": "According to SEC filings...", "source_type": "auto"},
                            },
                            "full": {
                                "summary": "Full request",
                                "value": {
                                    "text": "Company X today announced...",
                                    "source_url": "https://example.com/pr",
                                    "source_type": "official_release",
                                    "source_name": "Acme Corp IR",
                                    "include_signal_brief": True,
                                },
                            },
                        },
                    }},
                },
                "responses": {
                    "200": {
                        "description": "Analysis complete",
                        "headers": {"X-Request-Id": {"schema": {"type": "string"}}},
                        "content": {"application/json": {
                            "schema": {"$ref": "#/components/schemas/AnalyzeResponse"}
                        }},
                    },
                    "400": {"description": "Bad request (missing text)",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    "401": {"description": "Unauthorized — API key required"},
                    "413": {"description": "Text too long (> 10 000 chars)"},
                    "429": {"description": "Rate limited"},
                    "500": {"description": "Internal error"},
                },
            }
        },
        "/api/v1/analyze.txt": {
            "post": {
                "summary": "Analyze text — plain-text report with stable section headers",
                "operationId": "analyzeV1Txt",
                "security": [{"ApiKeyAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/AnalyzeRequest"}
                    }},
                },
                "responses": {
                    "200": {
                        "description": "Plain-text report",
                        "headers": {"X-Request-Id": {"schema": {"type": "string"}}},
                        "content": {"text/plain": {"schema": {"type": "string"}}},
                    },
                    "400": {"description": "Bad request"},
                    "401": {"description": "Unauthorized"},
                    "413": {"description": "Text too long"},
                },
            }
        },
    },
}


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
        fetched, url_error, page_meta = fetch_url_text(fetch_url_val)
        if fetched:
            text = fetched
            if not source_url:
                source_url = fetch_url_val
            # Auto-fill source name from page metadata when user left it blank
            if source_name in ("", "unknown") and page_meta:
                parts = [p for p in (page_meta.get("author"), page_meta.get("publisher")) if p]
                if parts:
                    source_name = " / ".join(parts)
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
        text, err, page_meta = fetch_url_text(url)
        if err:
            return jsonify({"error": err}), 400
        # Auto-fill source_name from page metadata if caller didn't supply one
        if not payload.get("source_name") and page_meta:
            parts = [p for p in (page_meta.get("author"), page_meta.get("publisher")) if p]
            if parts:
                payload["source_name"] = " / ".join(parts)

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
    return jsonify({
        "status":           "ok",
        "service":          "signal-sieve",
        "api_version":      API_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "uptime_seconds":   int(time.time() - _START_TIME),
    })


@app.get("/openapi.json")
def openapi_spec():
    return jsonify(_OPENAPI_SPEC)


# ── API v1 routes ─────────────────────────────────────────────────────────────

@app.post("/api/v1/analyze")
def api_v1_analyze():
    request_id = _make_request_id()

    auth_err = _check_api_key()
    if auth_err:
        return _api_error(401, "unauthorized", auth_err, request_id)

    payload = request.get_json(silent=True) or {}
    text, source_type, source_name, source_url, err = _validate_analyze_payload(
        payload, request_id
    )
    if err:
        return err

    try:
        result = analyze(
            text,
            source_type=source_type,
            source_name=source_name,
            source_url=source_url,
        )
    except Exception as exc:
        return _api_error(500, "internal_error",
                          "Signal Sieve failed while processing this input.", request_id)

    include_brief = payload.get("include_signal_brief", True)
    band = result.get("confidence_band", {})

    response = {
        **_api_envelope(request_id),
        "recommended_action":      result["recommended_action"],
        "overall_confidence":      result["scores"]["overall_confidence"],
        "receipt_readiness_score": result["scores"]["overall_confidence"],
        "confidence_band":         [band.get("low", 0), band.get("high", 0)],
        "custody_warning_level":   result["custody_warning_level"],
        "custody_label":           result.get("custody_label", ""),
        "triage_summary":          result["triage_summary"],
        "source_type":             result["source_type"],
        "source_genre":            result.get("source_genre", ""),
        "source_name":             result.get("source_name", ""),
        "source_url":              result.get("source_url", ""),
        "scores":                  result["scores"],
        "flags":                   result["flags"],
        "evidence_breakdown":      result["evidence_breakdown"],
        "questions_to_ask":        result["questions_to_ask"],
        "ai_instruction":          result.get("ai_instruction", ""),
    }

    if result.get("evidence_shape"):
        response["evidence_shape"] = result["evidence_shape"]

    if include_brief and result.get("signal_brief"):
        response["signal_brief"]      = result["signal_brief"]
        response["missing_receipts"]  = result["signal_brief"].get("missing_receipts", [])
    else:
        response["missing_receipts"] = []

    resp = jsonify(response)
    resp.headers["X-Request-Id"] = request_id
    return resp


@app.post("/api/v1/analyze.txt")
def api_v1_analyze_txt():
    request_id = _make_request_id()

    auth_err = _check_api_key()
    if auth_err:
        return _api_error(401, "unauthorized", auth_err, request_id)

    payload = request.get_json(silent=True) or {}
    text, source_type, source_name, source_url, err = _validate_analyze_payload(
        payload, request_id
    )
    if err:
        return err

    try:
        result = analyze(
            text,
            source_type=source_type,
            source_name=source_name,
            source_url=source_url,
        )
    except Exception as exc:
        return _api_error(500, "internal_error",
                          "Signal Sieve failed while processing this input.", request_id)

    sb = result.get("signal_brief", {})
    es = result.get("evidence_shape", {})
    HR = "---"

    lines = [
        "SIGNAL SIEVE REPORT",
        HR,
        f"ACTION: {result['recommended_action'].replace('_', ' ').upper()}",
        "VERDICT TYPE: signal_shape_assessment",
        "SAFE USE: Receipt triage only. Not fact-checking.",
        f"REQUEST ID: {request_id}",
        f"ANALYSIS VERSION: {ANALYSIS_VERSION}",
        HR,
        "SOURCE",
        f"TYPE: {result['source_type']}",
        f"GENRE: {result.get('source_genre', 'unknown')}",
        f"CUSTODY: {result['custody_warning_level']}",
        f"NAME: {result.get('source_name', 'unknown')}",
    ]
    if result.get("source_url"):
        lines.append(f"URL: {result['source_url']}")

    if es:
        providers = ", ".join(es.get("named_data_providers", [])) or "none"
        venues    = ", ".join(es.get("markets_venues_mentioned", [])) or "none"
        lines += [
            HR, "EVIDENCE SHAPE",
            f"LOCAL NUMBERS: {es.get('local_numeric_anchors', 'unknown')}",
            f"NAMED DATA PROVIDERS: {providers}",
            f"MARKETS / VENUES: {venues}",
            f"PRIMARY DATA LINKED: {'yes' if es.get('primary_links_present') else 'no'}",
            f"EXTERNAL RECEIPTS: {es.get('external_receipts', 'unknown')}",
            f"CAVEATS PRESENT: {'yes' if es.get('source_caveats_present') else 'no'}",
        ]

    if sb.get("top_movers"):
        lines += [HR, "TOP MOVERS"]
        lines.extend(f"  {m}" for m in sb["top_movers"])

    for section_key, header in [
        ("key_signals",              "KEY SIGNALS"),
        ("source_caveats",           "SOURCE CAVEATS"),
        ("interpretation_or_framing","INTERPRETATION / FRAMING"),
        ("missing_receipts",         "MISSING RECEIPTS"),
        ("do_not_pass_forward_as",   "DO NOT PASS FORWARD AS"),
        ("follow_up_sources",        "FOLLOW-UP SOURCES"),
    ]:
        items = sb.get(section_key, [])
        if items:
            lines += [HR, header]
            lines.extend(f"- {i}" for i in items)

    if result["flags"]:
        lines += [HR, "FLAGS"]
        lines.extend(f"- {f}" for f in result["flags"])

    if result["questions_to_ask"]:
        lines += [HR, "QUESTIONS TO ASK"]
        lines.extend(f"- {q}" for q in result["questions_to_ask"])

    lines += [HR, result.get("ai_instruction", "")]

    body = "\n".join(lines)
    resp = app.make_response(body)
    resp.headers["Content-Type"]  = "text/plain; charset=utf-8"
    resp.headers["X-Request-Id"]  = request_id
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
