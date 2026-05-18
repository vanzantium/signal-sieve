---
name: False positive
about: Signal Sieve flagged or downgraded something it shouldn't have
labels: false-positive
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

<!-- Paste the key fields: recommended_action, verdict, flags, scores.overall_confidence -->

```json
{
  "recommended_action": "...",
  "verdict": "...",
  "flags": [...],
  "scores": { "overall_confidence": 0.0 }
}
```

## What you expected

<!-- What action/verdict did you expect, and why? -->

## Why this matters

<!-- Is this a calibration gap, a missing heuristic, or a threshold issue? -->
