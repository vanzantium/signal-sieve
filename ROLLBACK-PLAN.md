# Signal Sieve v1.8.9 — Rollback plan (restore live v1.8.9-first-party-rc9)

**Candidate this package rolls back from:** v1.8.9-first-party-rc10d
(`v1.8.9-first-party-rc10d`). **Known-good live version:** v1.8.9-first-party-rc9
(`https://signal-sieve.onrender.com`, deploy **`dep-dc25top42hec73clcivg`**,
Live 2026-11-23 08:43 MST, merge commit
**`69a18c9b714753c69e5318eb60a3d69f7986fc3b`**).

## How to roll back

One manual Render redeploy of the known-good deploy
**`dep-dc25top42hec73clcivg`** — or, equivalently, a manual deploy of
commit **`69a18c9b714753c69e5318eb60a3d69f7986fc3b`**.

The service is stateless — no database, no data to migrate — and
Render auto-deploy does not fire in practice (deploying a specific
commit disabled it), so nothing can silently undo the rollback.

**Estimated time: 10–20 minutes** from decision to verified-restored.

## Verify the rollback (all must pass)

1. `GET /health` returns `analysis_version: v1.8.9-first-party-rc9`.
2. Smoke analyze on the run-20 primary returns `use_as_anchor` /
   `official_release` with **no** guardrail flag.
3. Smoke compare on the 4 run-20 feeds returns the **live-verified RC9**
   numbers — the RC9 deployment record (2026-11-23, live-verified
   0 failed) is the reference; do not verify against RC6 or RC10
   numbers.

## Earlier fallback

If the RC9 deploy itself cannot be restored, the earlier known-good
deploy is RC6: **`dep-dc168i59fdbs739tt0cg`**, Live 2026-11-21 20:41:33
MST, merge commit **`7602bbe75f173b52b2a662c679196f88a23424ce`**. A
rollback verified against RC6 must return the RC6 live numbers —
`aligned`, **19 shared / 18 aligned / 0 conflicts /
146 uniques / 4 not_comparable / 1 benchmark-mismatch** — with the
RC-only fields present: `separability`,
`headline_weighting`, `speed_ratio_checks`, `revision_notes`;
`has_unresolved_comparisons: true`;
`needs_source_clarification: []`.

   (These are the RC6 live numbers, verified 2026-11-21. They differ
   from live v1.8.8's 24 shared / 16 aligned / 0 conflicts / 115
   uniques / 4 not_comparable / 0 mismatches, and from RC2's local
   replay 19 shared / 18 aligned / 145 uniques / 3 not_comparable /
   1 mismatch. A rollback verified against the wrong numbers would
   look like a failure.)

## Limits

- The GitHub repo HEAD points at the RC6 merge (commit `7602bbe7`, PR
  #3), so a durable code rollback needs a separate approved revert
  commit; redeploying the known-good deploy restores the running
  service only.
- Anything already served or copied while v1.8.9 RC10/RC10a/RC10b was live
  (responses, capsules, posts) is not recalled by redeploying.
