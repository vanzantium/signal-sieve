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
| `verify_primary` | Usable signal — check key claims at the source |
| `treat_as_lead` | Mixed signal — read, don't trust uncritically |
| `seek_receipts` | Claims outpace evidence — demand support |
| `drop` | High pressure / low trust — don't pass forward |
| `reject` | Manipulation + attack language — reject as evidence |

It is a **receipt clerk, not an oracle**. It can flag a propaganda piece. It cannot confirm that a clean-looking primary source is telling the truth.

---

## Quick start

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

## Output schema (stable keys)

```json
{
  "recommended_action": "verify_primary | treat_as_lead | seek_receipts | drop | reject",
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

## Running the tests

```bash
python run_tests.py
```

Seven fixture-based tests covering: clean primary source, evidence-word salad, hype/social, anonymous claim, attack language, hedged opinion, short text.

---

## Known calibration weak spots

These are **open questions**, not bugs. Contributions and field data welcome.

1. **News articles** — longer articles with mixed sourcing get averaged scores; the worst paragraph can be buried.
2. **Abstracts** — academic abstracts are dense with evidence markers and often score higher than the full paper warrants.
3. **Forum / conversational text** — casual hedges ("I think", "maybe") suppress certainty_bias even in irresponsible claims.
4. **Translated text** — non-English text gets a conservative penalty but heuristics remain English-only.
5. **Adversarial salting** — a motivated actor can inflate scores by pasting evidence marker words near numbers. Signal Sieve is a filter, not a firewall.

---

## Design philosophy

- **Receipt clerk, not an oracle.** For any consequential decision, demand primary sources.
- **Honest framing.** The `ai_instruction` field in every output reminds downstream models that the verdict is heuristic, not truth.
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
