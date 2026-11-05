# Signal Sieve

**A heuristic pre-belief filter for AI pipelines and human analysts.**

Signal Sieve audits the *shape* of a text before it gets trusted, summarized, or repeated.
It does **not** fact-check the world. It asks: does this text *behave* like signal or noise?

```
python signal_sieve.py --demo --pretty
```

---

## What it does

It scores text across four dimensions and returns a `recommended_action`:

| Action | Meaning |
|---|---|
| `use_as_anchor` | Anchor-grade primary — suitable to anchor the evidence chain for the claim it directly supports (custody is proximity, not proof of truth) |
| `verify_primary` | Usable signal — check key claims at the source |
| `treat_as_lead` | Mixed signal — read, don't trust uncritically |
| `seek_receipts` | Claims outpace evidence — demand support |
| `drop` | High pressure / low trust — don't pass forward |
| `reject` | Manipulation + attack language — reject as evidence |

It is a **receipt clerk, not an oracle**. It can flag a propaganda piece. It cannot confirm that a clean-looking primary source is telling the truth.

---

## Live web app

Try it now — no install, no signup:

👉 **[signal-sieve.onrender.com](https://signal-sieve.onrender.com)**  
*(Render free tier — first load may take ~30 s while the service wakes up)*

Or deploy your own copy in two clicks:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/vanzantium/signal-sieve)

The web app also exposes a REST API:

```bash
curl -X POST https://signal-sieve.onrender.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "BREAKTHROUGH miracle cure!", "source_type": "social"}'
```

---

## Quick start (CLI / library)

**No install needed. Zero dependencies. Python 3.8+.**

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/signal-sieve.git
cd signal-sieve

# Run the built-in demo (four cases spanning the verdict range)
python signal_sieve.py --demo --pretty

# Analyze a string
python signal_sieve.py --text "Scientists proved the miracle cure works. Trust me." --pretty

# Analyze a file
python signal_sieve.py --file article.txt --source-type secondary --pretty

# Pipe text
cat article.txt | python signal_sieve.py --stdin --json

# Get JSON output (for pipeline use)
python signal_sieve.py --text "..." --json
```

---

## Plugging into an AI pipeline

```python
from signal_sieve import analyze

result = analyze(
    text,
    source_type="auto",          # or: primary, secondary, social, opinion, ...
    source_name="Reuters",
    source_url="https://reuters.com/...",
    language="en",               # en reference pack; es starter pack
)

if result["recommended_action"] in {"drop", "reject"}:
    # refuse, re-prompt, or warn user
    pass
else:
    # feed result["questions_to_ask"] back into the model as a refinement step
    # weight confidence by result["confidence_band"] not just overall_confidence
    pass
```

The full output is JSON-serializable and designed to be fed back into a model as a system note.

---

## Language-aware analysis

Signal Sieve separates language-independent structure from declared-language
lexical risk:

- `language="en"` uses the reference English pack and remains the default.
- `language="es"` uses a starter Spanish pack. It is useful but not yet
  corpus-calibrated to English parity.
- Other language codes run in explicit `structural_only` mode. They inspect
  URLs, dates, identifiers, numbers, and custody shape, but cannot earn
  `verify_primary` while lexical risk is unread.

The language layer also checks the worst paragraph so a clean, well-anchored
wrapper cannot hide a reject/drop-grade payload.

```bash
python signal_sieve.py --text "Fuentes dicen..." --language es --json
python test_i18n.py
```

---

## Output schema (stable keys)

```json
{
  "recommended_action": "use_as_anchor | verify_primary | treat_as_lead | seek_receipts | drop | reject",
  "triage_summary":     "one-line summary for fast routing / downstream AI",
  "verdict":            "human-readable summary",
  "scores": {
    "signal":                   0.0,
    "noise":                    0.0,
    "pressure":                 0.0,
    "source_custody":           0.0,
    "certainty_bias":           0.0,
    "evidence":                 0.0,
    "attribution":              0.0,
    "manipulation_risk":        0.0,
    "certainty_to_evidence_gap": 0.0,
    "overall_confidence":       0.0
  },
  "confidence_band":    { "low": 0.0, "high": 0.0 },
  "custody_label":      "PRIMARY / raw record or research",
  "custody_warning_level": "clean | watch | mismatch | weak",
  "evidence_breakdown": { "anchored": 0, "unanchored": 0 },
  "flags":              ["..."],
  "questions_to_ask":   ["..."],
  "ai_instruction":     "..."
}
```

**`anchored` vs `unanchored` evidence**: an evidence marker (e.g. "methodology", "dataset") is *anchored* if a number, URL, date, or proper-noun pair appears nearby inside the text. This does **not** mean externally verified — it means the claim has local support. Unanchored markers are claims without receipts.

---

## Source types

Pass `source_type="auto"` and Signal Sieve will infer from text + URL signals. Or declare explicitly:

| Type | Example |
|---|---|
| `first_hand` | Eyewitness account, personal recording |
| `primary` | SEC filing, academic paper, trial transcript |
| `official` | Press release, government statement |
| `expert` | Named expert interpretation |
| `secondary` | News article citing primary sources |
| `tertiary` | Summary of summaries, Wikipedia |
| `anonymous` | "Sources familiar with the matter" |
| `social` | Tweet, Reddit post, TikTok |
| `opinion` | Op-ed, essay, blog take |
| `unknown` | Unclear provenance |

A declared type that contradicts inferred signals (e.g. passing `source_type="primary"` for anonymous rumor text) is flagged and reduces the custody score — preventing laundering.

---

## X / Twitter News Posts

Signal Sieve has a dedicated `x_news_post` genre for short breaking-news posts
from X/Twitter URLs or X-style post text. It treats these posts as leads, not
receipt-grade evidence.

The X news layer checks for:

- Off-platform or primary receipts such as official statements, raw video,
  court records, named wire reports, or non-social links.
- Timestamp/time-zone shape so breaking-news claims can be placed in sequence.
- Rumor-chain markers such as "unconfirmed", "sources say", screenshots, or
  repost-only framing.
- Engagement bait such as "repost before they delete this" or "follow for
  updates".

API responses for this genre include `x_post_assessment` with a receipt score
and marker counts. Even a strong X news post is capped at `treat_as_lead`; use
the follow-up questions to find primary/off-platform confirmation before
repeating the claim as fact.

```bash
curl -X POST https://signal-sieve.onrender.com/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"BREAKING: Officials confirm...","source_url":"https://x.com/example/status/123","source_type":"auto"}'
```

---

## Cross-document comparison (`POST /api/v1/compare`)

Compares numeric claims across 2–10 documents. Claims normalize to
canonical entity keys (Treasury yields, indices, commodities, macro
indicators, benchmark scores) so different surface forms still match.

- **Dates are claims too.** Full dates normalize to ISO and bare years
  stay years; shared dates and shared years align across documents.
  Differing dates are never conflicts — different events legitimately
  have different dates.
- **Benchmark scores** (GPQA, SWE-bench, HumanEval, MMLU, MMMU, AIME,
  GSM8K, MATH, ARC-AGI, Humanity's Last Exam) match in colon form
  ("GPQA: 61.3%") and reverse form ("61.3% on GPQA").
- **Pricing rows** match only when pricing words sit near the $ amount,
  so market prices never leak into pricing claims. Input vs output is
  decided by the nearest input/output word.
- **Unit guard.** The same entity in different units (a count vs a
  percentage) is reported as a unit mismatch — not comparable, not a
  conflict, not an alignment.
- **Presence/absence channel.** PRESENT / EXPLICITLY_ABSENT /
  NOT_OBSERVED / UNKNOWN per entity per document. Only explicit absence
  language ("no Gold row", "Gold not listed") counts as absence; a
  merely-missing entity is NOT_OBSERVED and never becomes a conflict.

## Custody notes (v1.7.0)

- **First-party announcements keep their type.** A declared primary on
  a first-person announcement ("we are announcing…") is not downgraded
  to secondary merely because the URL contains "blog". The publisher
  speaking about its own policy is a primary source; the genre downgrade
  still fires for reporting on others. When it does fire, the flag names
  the genre that triggered it.
- **Questions adapt to custody.** A verified primary feed gets
  follow-ups about implementation dates, scope, and supersession — not
  the generic "closest primary source?" question, which only makes
  sense while the primary is still missing.

## Running the tests

```bash
python run_tests.py
```

The suite includes 19 source/genre fixtures, API contract checks, and the
declared-language contract:

```bash
python run_tests.py
python test_api.py
python test_i18n.py
```

---

## Known calibration weak spots

These are **open questions**, not bugs. Contributions and field data welcome.

1. **News articles** — longer articles with mixed sourcing get averaged scores; the worst paragraph can be buried.
2. **X / Twitter posts** — source links may be hidden behind short links or screenshots. Treat the mode as a receipt ladder, not verification.
3. **Abstracts** — academic abstracts are dense with evidence markers and often score higher than the full paper warrants.
4. **Forum / conversational text** — casual hedges ("I think", "maybe") suppress certainty_bias even in irresponsible claims.
5. **Spanish calibration** — the starter pack needs multi-author corpus calibration before parity claims.
6. **Additional languages** — unsupported languages are structural-only until a reviewed lexicon pack is added.
7. **Adversarial salting** — a motivated actor can inflate scores by pasting evidence marker words near numbers. Signal Sieve is a filter, not a firewall.

---

## Design philosophy

- **Receipt clerk, not an oracle.** For any consequential decision, demand primary sources.
- **Honest framing.** The `ai_instruction` field in every output reminds downstream models that the verdict is heuristic, not truth.
- **Extract → Identify → Contextualize → Compare → Recommend.** Each stage preserves the evidence supporting its decisions (Arch review 2026-11-05, v1.8.1). Recognizing a pattern is not establishing what the pattern represents: a number means nothing without its relationship to a measurement, a word does not identify an entity without context, a document does not establish primary custody merely by declaring itself primary, and a recommendation is not justified merely because an entity extractor produced a name. New failure modes should be investigated against this pipeline before becoming another collection of special-case exemptions.
- **Zero dependencies.** Standard library only. Drop `signal_sieve.py` anywhere Python 3.8+ runs.
- **Tunable.** All scoring weights are named constants. The test corpus makes regression easy.

---

## Contributing

Bug reports, false-positive/false-negative reports, and fixture additions are the most useful contributions right now.

Use the issue templates:
- **False positive** — Signal Sieve flagged something it shouldn't have
- **False negative** — Signal Sieve missed something it should have caught

For code changes: open an issue first describing what you want to change and why.

---

## License

MIT — see [LICENSE](LICENSE).
