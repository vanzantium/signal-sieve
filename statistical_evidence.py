"""Passage-scoped release provenance and bounded extraction of economic measures."""
import calendar
import json
import re
from pathlib import Path

RELEASES = tuple(json.loads((Path(__file__).parent / "evidence/releases.json").read_text()))
RELEASES += tuple(json.loads(p.read_text()) for p in sorted((Path(__file__).parent / "evidence/reviewed").glob("*.json")))
MONTH = "(?:" + "|".join(calendar.month_name[1:]) + ")"
NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
MOVE = r"increased|rose|decreased|fell|declined"
PERIOD = rf"(?: in (?P<period>{MONTH}(?: 20\d\d)?))?"
MEASURE = re.compile(
    rf"\b(?:(?P<payroll>(?:total )?nonfarm payroll(?: employment|s)) (?P<pdir>{MOVE}) by (?P<jobs>{NUMBER})(?!\d|[,.]\d)(?!\s*(?:million|billion|thousand|%|percent))(?: jobs)?"
    rf"|(?P<index>consumer price index(?: for all urban consumers(?: \(cpi-u\))?)?|producer price index(?: for final demand)?|food index) (?P<idir>{MOVE}) (?:by )?(?P<change>{NUMBER})\s*(?:percent\b(?!\s+points?)|%(?!\s+points?))"
    rf"|(?P<rate>unemployment rate) (?:changed little at|stood at|was|fell to|rose to|edged (?:down|up) to) (?P<level>{NUMBER})\s*(?:percent\b(?!\s+points?)|%(?!\s+points?)))"
    + r"(?P<adjusted> on a seasonally adjusted basis)?" + PERIOD, re.I)
PRIOR = re.compile(rf", after (?P<dir>rising|falling|declining) (?P<value>{NUMBER}) percent in (?P<period>{MONTH}(?: 20\d\d)?)\b", re.I)
HEADER = re.compile(r"^\s*(?:source|url|title|author|date|length|coverage|fetched|published|updated)\s*:|^\s*(?:primary source|coverage|secondary source)\s*[—–]", re.I)
NON_ASSERTION = re.compile(r"\b(?:if|unless|suppose|hypothetical|forecast|projected|expected|denied|incorrect|falsely|not true)\b", re.I)
REPORTER = re.compile(r"\b(?:according to|reported by|sources said|the agency said|Bureau of Labor Statistics said|Reuters reports|Reuters reported|researchers found|researchers at|a study by|sources say|previously reported|according to|study conducted by|reported that|news coverage|summary of|summarizes)\b", re.I)

def statistics(text):
    rows = []
    # Keep offsets stable when an HTML or PDF excerpt wraps a sentence.
    clean = "".join((" " * len(line.rstrip("\r\n")) + line[len(line.rstrip("\r\n")):]) if HEADER.match(line) else line
                    for line in text.splitlines(keepends=True))
    for line in re.finditer(r"[^\n]+(?:\n(?![ \t]*\n)[^\n]+)*", clean):
        body = re.sub(r"\s", " ", line.group())
        if HEADER.match(body):
            continue
        previous = None
        for m in MEASURE.finditer(body):
            key = "jobs" if m["jobs"] else "change" if m["change"] else "level"
            measure = "nonfarm payroll employment" if m["jobs"] else (m["index"] or "unemployment rate").lower()
            direction = m["pdir"] or m["idir"] or ""
            prefix = body[:m.start()]
            clause = re.split(r"(?<=[.!?])\s+", prefix)[-1]
            # A conjunct can inherit an explicit period only within this narrow paired statement.
            period = m["period"]
            if not period and previous and re.fullmatch(r",? and the ", body[previous.end():m.start()], re.I):
                period = previous["period"]
            def emit(match, group, movement, period, subject):
                value = float(match[group].replace(",", ""))
                tail = body[match.end():]
                is_question = re.match(r"[^.!?]*\?", tail)
                rows.append(dict(measure=subject, value=value,
                                 unit="jobs" if key == "jobs" else "percent",
                                 kind="level" if key == "level" else "change",
                                 direction=("down" if movement.lower() in ("falling","declining","fell","decreased","declined") else "up") if movement else None,
                                 period=period, adjustment=("seasonally adjusted" if (match is m and m["adjusted"]) or re.match(r",\s*seasonally adjusted\b",tail,re.I) else None), span=text[line.start()+match.start():line.start()+match.end()], start=line.start()+match.start(), end=line.start()+match.end(),
                                 assertion="review" if NON_ASSERTION.search(clause) or is_question else "as_written"))
            emit(m, key, direction, period, measure)
            prior = PRIOR.match(body, m.end()) if m["index"] else None
            if prior:
                emit(prior, "value", prior["dir"], prior["period"], measure)
            previous = m
    return rows

def _normalized(text):
    return " ".join(text.split())

def public_release(record):
    fields = ("publisher", "release_date", "release_id", "url", "html_sha256",
              "passage", "revision_of", "revision_scope")
    return {key: record[key] for key in fields if key in record}


def passage_evidence(text, source_url, declared, stats, source_name="", unresolved_genre=False):
    supplied = source_url.strip().split("#", 1)[0]
    body = _normalized(text)
    if not body:
        return None
    visible_body = _normalized("\n".join(line for line in text.splitlines() if not HEADER.match(line)))
    matches = []
    versions = [r for r in RELEASES if r["url"] == supplied]
    fingerprint = lambda t: re.sub(NUMBER, "#", t.rstrip("."))
    selected = next((r for r in versions if body.rstrip(".") == r["passage"].rstrip(".")), None)
    if selected is None:
        comparable = [r for r in versions if fingerprint(body) == fingerprint(r["passage"])]
        selected = comparable[0] if len(comparable) == 1 else None
    for r in RELEASES:
        passage = r["passage"]
        if re.search(r"(?<!\w)" + re.escape(passage) + r"(?!\w)", visible_body):
            matches.append(public_release(r))
    full = bool(selected and body.rstrip(".") == selected["passage"].rstrip("."))
    if full and not matches:
        matches.append(public_release(selected))
    apparent_coverage = declared in ("secondary","tertiary","social","opinion") or bool(REPORTER.search(body))
    if source_name.strip().lower() in ("reuters", "associated press", "ap news") and not (selected and source_name.casefold() == selected["publisher"].casefold()):
        apparent_coverage = True
    named = source_name.strip().casefold()
    aliases = {"", "unknown"}
    if selected:
        aliases.add(selected["publisher"].casefold())
        if selected["publisher"] == "U.S. Bureau of Labor Statistics":
            aliases.update(("bls", "bureau of labor statistics"))
    identity_consistent = named in aliases or re.fullmatch(r"doc_?\d+", named)
    if full and identity_consistent and not apparent_coverage and declared in ("primary","official","auto","unknown",""):
        status, resolution = "matched_release_excerpt", "primary"
    elif selected and not full and fingerprint(body) == fingerprint(selected["passage"]):
        status, resolution = "numeric_discrepancy", "secondary" if apparent_coverage else "unknown"
    elif matches:
        status, resolution = "matched_passage_in_document", "secondary" if apparent_coverage else "unknown"
    elif apparent_coverage:
        return None
    elif versions or ((stats or unresolved_genre) and declared in ("primary","official") and supplied):
        status, resolution = "passage_match_pending", "unknown"
    elif stats and declared in ("primary","official"):
        status, resolution = "source_location_missing", "unknown"
    else:
        return None
    revisions = [public_release(r) for r in RELEASES if r.get("revision_of") in {m["release_id"] for m in matches}]
    differences = []
    if status == "numeric_discrepancy":
        for submitted, reference in zip(re.finditer(NUMBER, body), re.finditer(NUMBER, selected["passage"])):
            if submitted.group() != reference.group():
                differences.append(dict(submitted=submitted.group(), publisher=reference.group()))
    return dict(quotation_use=("attribute_exact_words_to_dated_release" if status == "matched_release_excerpt" else "review"),
                differing_numbers=differences, later_revisions=revisions, status=status, document_type=resolution, supplied_url=supplied,
                scope="passage only", matched_releases=matches,
                comparison_release=public_release(selected) if status == "numeric_discrepancy" else None,
                document_publisher=source_name or None)

LABELS = {
    "matched_release_excerpt": "Excerpt matched to the dated publisher release.",
    "numeric_discrepancy": "A number differs from the dated publisher passage. Review both versions.",
    "matched_passage_in_document": "Publisher wording located inside the supplied document. The match applies to that passage.",
    "source_location_missing": "First-party origin claimed. A source location or reviewed passage is needed.",
    "passage_match_pending": "First-party source supplied. Passage match pending.",
}
