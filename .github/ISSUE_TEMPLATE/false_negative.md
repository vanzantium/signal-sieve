---
name: False negative
about: Signal Sieve missed something it should have caught (too lenient)
labels: false-negative
---

## Text analyzed

<!-- Paste the text (or a representative excerpt) here. Anonymize if needed. -->

```
<paste text here>
```

## Arguments used

```
source_type=???   source_url=???
```

## What Signal Sieve returned

```json
{
  "recommended_action": "...",
  "verdict": "...",
  "flags": [...],
  "scores": { "overall_confidence": 0.0 }
}
```

## What the problem actually was

<!-- What should Signal Sieve have caught? Hype? Attack language? Unanchored claims? Source mismatch? -->

## Why this matters

<!-- Is the gap a missing pattern, a weight issue, or a new category of manipulation? -->
