# RC10c build command log (2026-11-25)

Every shell command run during this build, with its outcome. Kept so
failed calls are accounted for, not left unexplained (Arch's ask after
RC10b's six unexplained shell failures).

## Setup
- `sha256sum rc10b/signal-sieve-v1.8.9-first-party-rc10b.zip rc10a/... rc10/...` → exit 0. All three shas match the task brief (RC10b d2c05eb2…, RC10a 5b5e6c04…, RC10 7133a5e0…).
- `rm -rf rc10c && mkdir -p rc10c && unzip -q ../rc10b/...zip` → exit 0. 115 files extracted.
## 2026-11-25 RC10c implementation session (continued)

- Baseline probes run against unfixed rc10c tree (venv rc3-clean-venv-3119, py 3.11.9): probe_fails_345, probe_fails_123, probe_competing, probe_competing2/3/4, probe_arch_56, replay_run26_indep — all RC10b failures reproduced (F3 doc1 no claims; Nov9 30/30/43 temporal 0; Nov7 15/15/26 temporal 0; item7 only Muse 73.4; competing4 case A both flags drop while merge refuses; arch56 reuters/nature False, byline denial/quote True; five-rows SSH opus-5-5).
- RC9 reference run (rc9/code tree): F3 doc1 mints 0.330+0.364, doc0 mints 0.641 only; Nov9 34/33/42 temporal 1; Nov7 16/16/26 (expected — see probe_fails_345 output; Nov7 line truncated in display, re-verify at end).

## 2026-11-25 (verification) — full own-runs suite, fresh rc10c tree
- `python -m unittest test_review_fixes.RC10cRegression test_review_fixes.RC10bRun26Regression` (venv rc3-clean-venv-3119, Py 3.11.9): 19 tests OK (8 RC10c + 11 RC10b). NOTE: first run of RC10cRegression was 7/8 — the methodology pipeline assertion expected treat_as_lead on a thin synthetic text; the text took the seek_receipts path instead. Rewrote the assertion to pin the downgrade flag on the first-person primary text (mini-swe-agent -> use_as_anchor, no flags; reuters/nature -> downgrade flag fires). No code change.
- `python run_tests.py`: 47/47 passed.
- `python -m unittest discover`: 59 tests, OK.
- `python test_api.py`: 163/164 passed, 1 failed — "POST /api/v1/compare returns the Nov 9 packet to aligned with a temporal comparison (v1.8.9 correction, Fix 1)". The test hard-codes RC9's 34 shared / 33 aligned / 42 unique; RC10c re-derives 33 / 32 / 43, temporal 1, clarification 1. Verified the divergence is exactly the 128.0-sessions group routing to clarification (missing ['threshold-vs-point'], compatible True) because alfw.ai's "up to 128 dreaming sessions" is a genuine ceiling and the other eight 128.0 values are points — the accepted threshold-vs-point semantics item 6 enshrines. Everything else the test asserts passes: status aligned, 0 conflicts, temporal pair (0.33, 0.641) with also_aligned_as ['generic|measurement|score|||sonnet-4-6']. The Nov 7 ledger test PASSES (16/16/26). This test's expectations were NOT changed (task allows only the F1 expectation change); the failure is reported honestly for the parent's decision.

## 2026-11-25 (verification, continued) — two follow-on fixes found in the run-26 replay
- Run-26 replay (fresh, /tmp/rc10c-probe-run/replay_run26_indep.py): shared 11, conflicts 0, SSH 49 (RC10b: 41), unique 52. SSH delta vs RC10b: +8/-1. Seven of the +8 were newly-minted from-to endpoints (FrontierCode 24.1/24.2, ExploitBench 36.1/80.0, ExploitGym 12.1/27.3, FrontierCode 50.0) — the 50.0 and 24.2 were WRONG-bench (Terminal-Bench figures attached to FrontierCode by the forward "on <Name>" attachment in _generic_benchmark, which scanned past the closing paren of "(24.2 to 50.0)").
- FOLLOW-ON FIX 1 (code/signal_compare.py _generic_benchmark tier 1): when the number sits inside an open parenthetical, the forward "on/in/at/for <Name>" scan truncates at the closing paren (only when the close is the next paren character; a paren opening after the number is an aside and does not bound the scan). New regression test test_from_to_pair_benchmark_stays_with_its_own_pair (24.2/50.0 -> Terminal-Bench, 24.1 -> FrontierCode). NOTE: 41.3 does not mint in the isolated two-pair sentence — verified pre-existing RC9 behavior (_generic_metric returns None; RC9 identical); the figure surfaces via the table pass in the full doc. Test asserts the three that mint + the two must-nots, with the 41.3 note recorded.
- FOLLOW-ON FIX 2 (code/signal_compare.py _endpoint_compatible + three SSH corroboration arms): the same replay showed 7 figures listed in BOTH a shared group and single_source_headlines — the SSH arms required exact endpoint equality, so a from-value was flagged lone even when another doc stated the same value as a plain point. A plain point now corroborates either endpoint; before vs after stay incompatible (RC10a rule). New regression test test_plain_point_corroborates_from_to_endpoint (main + lone-pair control + before-vs-after control). After the fix: replay SSH 42, zero figures in both shared and SSH; delta vs RC10b is exactly +2 (Grok 73.4 item 7; five-rows unstated item 5) and -1 (five-rows opus-5-5 re-keyed).
- Full suite re-run after each follow-on fix: run_tests.py 47/47; unittest discover 61 OK (was 59 + 2 new tests); test_api.py 163/164 (same single Nov 9 packet failure, documented); test_i18n.py 6/6; reproduce_run25.py 11/11; reproduce_runs23_24.py 12/12; all RC10a probes (probe_api, probe_claims, probe_claims2, probe_endpoint, probe_findings, probe_perplexity, probe_slash) + all RC10b probes (probe_fails_123, probe_fails_345, probe_fromto, probe_packets_tree, probe_packet_diff, probe_arch_56, probe_competing, probe_competing2, probe_competing3, probe_competing4) re-run clean; probe_fromto_cmp.py not run directly against rc10c (takes a 'rc10b'/'rc10' tree name pointing at verify-{which}-indep) — probe_fromto.py used instead, per plan.

## 2026-11-25 (packaging)
- Validation outputs written fresh to validation-results/ (rc10c-run_tests, rc10c-unittest, rc10c-test_api, rc10c-test_i18n, rc10c-run25, rc10c-runs23_24, rc10c-probes, rc10c-replay-run26.txt + .json). The rc10b-*.txt files that came with the fresh RC10b extraction were removed from rc10c/validation-results (they belong to RC10b).
- Packaging mistake, caught by verification (not shipped): the first SHA256SUMS was built with a sed that added an extra space before "./" filenames, so `sha256sum -c` on a fresh extraction failed all 117 entries ("No such file or directory" — filenames read with a leading space). Also, an intermediate regeneration included SHA256SUMS itself (hash of the empty file). Both fixed: manifest regenerated with plain `find ... | xargs sha256sum` excluding SHA256SUMS itself and the zip; zip rebuilt.
- Final package: signal-sieve-v1.8.9-first-party-rc10c.zip — sha/size: see the build note (rc10c-build-note-2026-11-25.md); no sha is claimed inside this package, by design (a zip cannot truthfully name its own sha). [RC10d fix: this line previously named stale sha 1b0dbc51… / 453,160 bytes, which matched no shipped zip.] 118 files (117 manifest entries + SHA256SUMS). Fresh extraction to ~/workspace/zaau-signal-sieve/verify-rc10c/: `sha256sum -c SHA256SUMS` 117/117 OK, zero failures. Spot checks on the fresh extraction: RC10cRegression + RC10bRun26Regression 21 tests OK; test_api.py 163/164 (same single Nov 9 packet failure); version stamp v1.8.9-first-party-rc10c confirmed at code/app.py:28.
- Preservation re-checked after the build: RC10b d2c05eb2e21db9509a03c39431944369c3fdf0dccc671136aaef27621e65945d, RC10a 5b5e6c04d8c6101cff3030b708e5ced3ee6f70eb107a7140339399dbfdee2162, RC10 7133a5e0157568b62206c4514785ee9bdb4f6d1234ed6ec311326117ce17be69 — all unchanged.

## 2026-11-25 (addendum) — Taylor-authorized Nov 9 packet test update
- Baseline before change: `sha256sum rc10c/rc10b/rc10a/rc10 zips` → exit 0; all four shas match the task brief (RC10c 9d9a27e4…, RC10b d2c05eb2…, RC10a 5b5e6c04…, RC10 7133a5e0…). NOTE: the packaging entry above records a "final package" sha 1b0dbc51… / 453,160 bytes, but the zip on disk is 9d9a27e4… / 453,918 bytes (verified by the sha256sum above) — the log entry is stale relative to the shipped zip; the shipped zip's sha is the one in the build note and the task brief.
- `python test_api.py` on the work tree (venv rc3-clean-venv-3119, Py 3.11.9) BEFORE the change → 163/164 passed, 1 failed: exactly the Nov 9 packet test, as reported in the build note.
- Probe `/tmp/rc10c-nov9-probe.py` (calls _packet_compare("2026-11-09", _NOV9_FEEDS) on the work tree) → exit 0: agreement aligned, conflicts 0, shared 33, aligned 32, unique 43, temporal 1, clarification 1 — entity generic|task_count|sessions|||opus-5, compatible true, missing ["threshold-vs-point"]. Matches the build note's re-derived counts exactly; no hand-editing of expectations beyond the authorized values.
- Edit `code/test_api.py` (Nov 9 packet test only): counts 34/33/42 → 33/32/43; added exactly-1-clarification assertion (sessions/opus-5, compatible true, missing threshold-vs-point) plus the reason comment in the test (second and last allowed expectation change; F1 was the first). No other test's expectations touched; no behavior file touched.

- `python test_api.py` on the work tree AFTER the change → **164/164 passed**, exit 0. (The Nov 9 packet test now passes with 33/32/43, temporal 1, clarification 1.)
- Full suite on the work tree (venv rc3-clean-venv-3119, Py 3.11.9): `run_tests.py` 47/47; `unittest discover` 61 OK; `test_i18n.py` 6/6; `reproduce_run25.py` 11 PASS / 0 failed; `reproduce_runs23_24.py` 12 PASS / 0 failed; run-26 replay (verifier's script, code symlinked to the rc10c work tree) → shared 11, conflicts 0, aligned 11, SSH 42, unique 50, agreement aligned, Cognition use_as_anchor, others treat_as_lead — all as expected.
- Validation records regenerated in validation-results/ (rc10c-run_tests, rc10c-unittest, rc10c-test_api now 164/164, rc10c-test_i18n, rc10c-run25, rc10c-runs23_24, rc10c-replay-run26.txt/.json). rc10c-probes.txt unchanged (no behavior file touched; probes re-run clean in the RC10c build).
- Preservation: superseded zip moved to `rc10c/superseded/signal-sieve-v1.8.9-first-party-rc10c-9d9a27e4.zip`, sha re-verified 9d9a27e4… after the move. NOT deleted.
- Packaging, first attempt (mistake, caught before shipping): manifest regenerated while 20 `__pycache__` .pyc files from the suite runs were present in the tree → 137 entries; the first zip also included 14 directory entries (118-file original had none). Fixed: deleted all `__pycache__`/.pyc, regenerated the manifest (117 entries), rebuilt with `zip -qrD` (no directory entries). Zip entry list verified byte-identical to the superseded zip's entry list (diff clean).
- Final package: signal-sieve-v1.8.9-first-party-rc10c.zip — sha/size: see the build note (rc10c-build-note-2026-11-25.md); no sha is claimed inside this package, by design (a zip cannot truthfully name its own sha). [RC10d fix: this line previously named stale sha e40ab4a9… / 461,078 bytes, which matched no shipped zip.] 118 files (117 manifest entries + SHA256SUMS).
- Fresh-extraction verification (new dir verify-rc10c-v2/): `sha256sum -c SHA256SUMS` → **117/117 OK, zero failures**. Suites re-run from the fresh extraction: test_api.py 164/164; run_tests.py 47/47; unittest discover 61 OK; test_i18n.py 6/6; reproduce_run25.py 11 PASS / 0 failed; reproduce_runs23_24.py 12 PASS / 0 failed; run-26 replay → shared 11, conflicts 0, aligned, SSH 42, unique 50, Cognition use_as_anchor, others treat_as_lead.
- Preservation re-checked after all work: RC10b d2c05eb2…, RC10a 5b5e6c04…, RC10 7133a5e0… — all unchanged.

- NOTE on log closure: the entries above were appended after the first rebuild of this addendum's zip, so the packaged BUILD-COMMAND-LOG.md did not yet contain them. This entry closes the log for packaging: manifest regenerated and zip rebuilt after this entry. The authoritative final sha/size are recorded in the build-note addendum, not here, to avoid a self-referential stale sha in the log (the RC10c build log's "final package" line went stale the same way).

## 2026-11-25 (RC10d) — documentation-only fix for the RC10c verification defect

Taylor authorized 2026-11-25 ~17:34. The RC10c independent verification
receipt passed RC10c with ONE named defect, documentation only: this
log's two "Final package" lines named shas (1b0dbc51… / 453,160 bytes
and e40ab4a9… / 461,078 bytes) that matched no shipped zip. A zip
cannot truthfully name its own sha — writing it changes the sha — so
RC10d removes the claim instead of correcting the number.

- Base: fresh extraction of the shipped RC10c zip into a new tree
  `rc10d/` (never edited the `rc10c/` tree or the shipped zip).
  Base zip sha verified at start: c65ce203… (461,878 bytes);
  RC10b d2c05eb2…, RC10a 5b5e6c04…, RC10 7133a5e0… — all match the
  task brief.
- The fix (packaged log): both stale "Final package" sha/size lines
  replaced with a pointer to the build note
  (`rc10c-build-note-2026-11-25.md`, RC10d addendum) as the
  authoritative record, with the note that no sha is claimed inside
  this package, by design. The rest of the log's history is kept
  unchanged, including the lines documenting that the old entries
  went stale. The truncated old shas (1b0dbc51…, e40ab4a9…) appear
  only inside the "[RC10d fix: …]" notes on the replaced lines, as
  a record of what was wrong — not as claims.
- Version stamp: `code/app.py:28` → `v1.8.9-first-party-rc10d` (the
  ONLY code change). `ROLLBACK-PLAN.md`: candidate updated to rc10d.
  `KNOWN-LIMITATIONS.md`: new RC10d section (docs-only log fix, no
  behavior change vs RC10c). No test expectation changed; no other
  behavior file touched. Delta vs the RC10c extraction confirmed by
  diff: exactly these 4 files (BUILD-COMMAND-LOG.md,
  KNOWN-LIMITATIONS.md, ROLLBACK-PLAN.md, code/app.py) plus the
  regenerated validation records.
- Full suite on the work tree (venv rc3-clean-venv-3119, Py 3.11.9):
  `test_api.py` 164/164; `run_tests.py` 47/47; `unittest discover`
  61 OK; `test_i18n.py` 6/6; `reproduce_run25.py` 11 PASS / 0 failed;
  `reproduce_runs23_24.py` 12 PASS / 0 failed.
- Run-26 replay (verifier's script, original hashed inputs): shared
  11, conflicts 0, alignment 11, SSH 42, unique 50, agreement
  aligned; Cognition `use_as_anchor`, others `treat_as_lead`;
  coverage 0.278 / 0.385 / 0.583 / 0.700 — identical to the RC10c
  verification receipt. Five-rows SSH is `...||unstated`; both
  SWE-Atlas 73.4 claims present.
- Probes re-run against the rc10d tree (16 scripts, all exit 0):
  all 7 RC10a probes + probe_fails_123, probe_fails_345,
  probe_fromto, probe_packets_tree (tree path passed explicitly),
  probe_arch_56, probe_competing–probe_competing4. Spot-checked:
  F3 temporal pair restored (0.330 before / 0.641 after); Nov 9
  33/32/43 temporal 1; Nov 7 16/16/26 temporal 0; reuters vetoes at
  function and pipeline level; competing case A flags stand, merge
  refuses to attach. NOT run: probe_packet_diff (hard-codes an
  RC9-vs-RC10b comparison by construction) and probe_fromto_cmp
  (takes only 'rc10b'/'rc10') — neither applies to an rc10d tree;
  the packet counts they covered were verified directly above.
- Validation records regenerated in `validation-results/` as
  `rc10d-*` (run_tests, unittest, test_api 164/164, test_i18n,
  run25, runs23_24, probes, replay-run26.txt + .json). The old
  `rc10c-*` records were removed. The temp probe-script copies and
  the temp replay-script copy were removed from the tree before
  packaging (they are verifier tooling, not package files).
- Packaging: all `__pycache__` deleted before manifest generation
  (the RC10c addendum round caught a 137-entry manifest from .pyc
  files; not repeated). Manifest regenerated with plain
  `find ... | xargs sha256sum`, excluding `SHA256SUMS` itself and
  the zip; zip built with `zip -qrD` (no directory entries).
- Final package: sha/size — see the build note
  (`rc10c-build-note-2026-11-25.md`, RC10d addendum). No sha is
  claimed inside this package, by design; this log is closed for
  packaging before the zip is built, and the work-tree log records
  the final sha only after packaging, outside the package.
