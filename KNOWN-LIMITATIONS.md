# RC10d update (documentation only)

Fix for the one defect in the RC10c independent verification receipt
(2026-11-25): the packaged `BUILD-COMMAND-LOG.md` named two stale
"Final package" shas (1b0dbc51… / 453,160 bytes and e40ab4a9… /
461,078 bytes), neither matching any shipped zip. A zip cannot
truthfully name its own sha — writing it changes the sha — so the
packaged log no longer claims a final-package sha/size at all: both
stale lines are replaced with a pointer to the build note
(`rc10c-build-note-2026-11-25.md`, RC10d addendum) as the
authoritative record. No behavior change vs RC10c: no code file
touched except the version stamp (`v1.8.9-first-party-rc10d`).

# RC10c update

Bounded correction round on RC10b (independent verification receipt,
2026-11-25 — the five `test_api` failures were RC10 regressions, all
five passing 162/162 on the RC9 tree). All 8 items fixed in code.
Full per-item account in the RC10c build note
(`rc10c-build-note-2026-11-25.md`); regression tests in
`code/test_review_fixes.py` (`RC10cRegression`, 10 tests).

1. **From-value minting.** The RC10 blanket "X to Y" skip in
   `_generic_numbers_in` is removed (it dropped BOTH endpoints of
   every from-to pair in the generic pass). The from-value mints
   with endpoint `before`, the to-value with `after`, including
   under a lead-in benchmark and in gapped shapes ("from 0.330 at
   baseline to 0.641"). A qualifier attaches only when no other
   number intervenes ("more than five times the 166" — "more than"
   governs "five", not 166); "over"/"just over" before a count
   followed by a duration noun is temporal, not a floor ("over 128
   sessions" is not a threshold; "over 90%" still is). Restores the
   Nov 7 ledger exactly (16/16/26) and the F3 temporal pair.
   Follow-ons found while verifying: (a) in a sentence naming two
   benchmarks with one pair each, the forward "on <Name>"
   attachment in `_generic_benchmark` must not cross the closing
   paren of the parenthetical containing the number — without the
   bound, Terminal-Bench's 24.2/50.0 attached to FrontierCode;
   (b) the single-source corroboration arms required exact
   endpoint equality, so a from-value was flagged lone even when
   another doc stated the same value as a plain point — a plain
   point now corroborates either endpoint (before vs after stay
   distinct figures, RC10a rule).
2. **Methodology exemption.** "reported in <lowercase>" vetoes only
   for recognized outlet names (reuters, nature, science, ap, afp,
   bloomberg, axios, politico, wired, verge, forbes, fortune,
   guardian, economist, bbc, wsj, nytimes, ft, thehill, huffpost,
   vice, vox, slate, atlantic, newyorker, lancet, jama — "cell"
   deliberately excluded, ambiguous with cell biology). "reported
   in mini-swe-agent" still exempts; "reported in The Times" still
   vetoes. The docstring/code mismatch is fixed (the docstring
   claimed "reported in the Devin CLI" was exempt; the code vetoes
   determined forms — the docstring now says so).
3. **Byline negation/attribution.** `detect_ai_generated_byline`
   returns false for denials ("The researcher denied that this
   article is AI-generated.") and for quotations about other
   articles ('The website says "this article is AI-generated"
   about a different story.'), true for direct disclosures. The
   field is renamed `disclosed_ai_generated` — it reports what the
   document discloses, not a detection of fact.
4. **Competing subjects in the same table.** The subject- and
   role-tolerant corroboration arms no longer skip the focal doc
   when searching for competing stated subjects, so one unstated
   prose figure no longer clears both Grok 73.4 and Muse 73.41 in
   the same table. The outer corroboration loops still skip the
   focal doc (a same-doc restatement is not independent
   corroboration). The flag now stands where the merge refuses to
   attach.
5. **"Five rows".** The count of Cognition's table rows is no
   longer attributed to Claude Opus 5.5: when the family is
   "rows"/"columns" the subject is cleared to unstated. The figure
   still mints; the subject abstains.
6. **F1 test update.** The up-to ceiling test now expects
   clarification + `compatible: true` with `missing:
   ["threshold-vs-point"]`, with the reason documented in the test
   (accepted semantics change). This is the ONE expectation change
   in the suite.
7. **Duplicate table value.** The dedup keeps BOTH claims (one per
   model cell): fresh-minted claims are registered in
   `_upgraded_claim_ids`, so the CellCog SWE-Atlas QA row's
   repeated 73.4 yields both Grok 4.7's and Muse Wonder's claims.
   Decision documented here and in the build note.
8. **Trivia.** `KNOWN-LIMITATIONS.md` "RC10bRun26Regression, 10
   tests" → "11 tests".

**Re-derived packet counts (not hand-edited).** Nov 9 re-derives
to 33 shared / 32 aligned / 43 unique, temporal 1, clarification
1 — NOT RC9's 34/33/42, for two principled reasons: (a) the
128.0-sessions group routes to clarification because alfw.ai's
"Building and maintaining the wiki required up to 128 dreaming
sessions" is a genuine ceiling and the other eight 128.0 values
are points — under the accepted threshold-vs-point semantics
(item 6) a ceiling + points clarify rather than align (RC9
aligned it only because RC9 predates the threshold diversion);
(b) the fourth 48.0 (alfw.ai COVERAGE header "Wiki Scores 48%")
does not mint — pre-existing RC10 feed-header hardening, not a
regression; the 48.0 group still aligns with 3 values. Nov 7 is
exactly RC9 (16/16/26). `test_api.py` is 163/164: the Nov 9
packet test still expects RC9's 34/33/42 and fails on the
re-derived counts — its expectations were NOT changed (only the
F1 change is allowed); the failure is reported for the parent's
decision, not hidden.

**Known boundaries (not regressions, identical in RC9):** bare
from-to shapes with no metric cue still mint nothing ("GPQA
improved from 47.6 to 56.3.", "Function-level pass rate rose
from 22.0% to 31.8%.", "From 10 to 20, the count doubled.", "It
rose from 5 to 9."). The 41.3 to-value does not mint in the
isolated two-pair sentence (the "FrontierCode" cue is claimed by
the nearer 24.1) — pre-existing RC9 behavior, verified; in the
full CellCog doc the 41.3 figure surfaces via the table pass.

**Known limitation (not fixed, conservative direction):** the
reverse of item 4 — a bench-less prose restatement can attach in
the merge while the stated group's SSH flag stands (SSH arms are
prefix-exact; the merge is bench-tolerant). Flag standing is the
safe side.

**Known residuals of this round:**
- Run-26 replay (local, RC10c tree, original hashed inputs):
  Cognition `use_as_anchor`, others `treat_as_lead`; shared 11,
  conflicts 0, unresolved 0, agreement `aligned`;
  `single_source_headlines` 42 (was 41 in RC10b) — the delta is
  exactly +2 (Grok 73.4, item 7; five-rows unstated, item 5) and
  −1 (five-rows opus-5-5 re-keyed). No figure appears in both a
  shared group and SSH.
- `test_api.py` 163/164 (see above); `run_tests.py` 47/47;
  `unittest discover` 61 tests OK; `test_i18n.py` 6/6;
  `reproduce_run25.py` 11/11; `reproduce_runs23_24.py` 12/12.
- Pre-existing RC10 subject diffs vs RC9 on Nov 9 remain (9.0
  sonnet-5 → sonnet-4-6; 6.0 sonnet-3-7 → unstated uniques).

**Standing measurement gates — still open (unchanged from RC10b):**
recall 45% unremeasured; 25% tolerance + +0.12 bump uncalibrated;
held-out relabel OPEN; Z.ai/Zhipu alias deferred (do not assert);
item 8 provenance/independence tracking still a design decision
for Arch/Taylor; image-table extraction OPEN.

# RC10b update

Bounded correction round on RC10a (run-26 gaps, 2026-11-25 — the
Cognition SWE-2.1 feeds). All 8 run-26 gaps addressed: 7 fixed in
code, 1 documented as not fixable by text extraction. Full per-gap
account in the RC10b build note
(`rc10b-build-note-2026-11-25.md`); regression tests in
`code/test_review_fixes.py` (`RC10bRun26Regression`, 11 tests —
corrected from "10 tests" in RC10c item 8; the class has 11).

1. **First-person methodology text keeps primary.** The bare
   "reported" secondary-reporting veto fired on the primary's own
   methodology text ("We report upstream or self-reported scores",
   "Other models are reported in mini-swe-agent"). Three methodology
   frames are now exempt, and only those three: "self-reported",
   "we report(ed)", and "reported in <lowercase tool>". "Reported in
   The Times" / "the paper reported" / "the company reported record
   sales" still veto. The Cognition primary now anchors
   (`use_as_anchor`, no flags) — 8th downgrade in 8 runs fixed.
2. **"Unite.AI" publisher parse.** Already fixed by RC10a's
   dotted-name parse (`_feed_declared_org` returns `unite.ai`);
   locked in by a regression test, no code change.
3. **Opus 5.5 misattributions.** Two fixes: (a) the exactly-one
   subject rule is now scoped to the number's own semicolon segment
   ("27.3% ...; Claude Opus 5.5 is listed at 91 and 34.5" no longer
   makes Opus the 27.3%'s subject); (b) "trails"/"lags"/"behind" join
   the comparator exclusion, with an optional trailing family word
   ("trails Claude Opus 5.5" names Opus as the baseline). Follow-ons
   found while verifying: a baseline number ("59.7 against 50.0")
   never takes the exactly-one subject; the clause's own CamelCase
   benchmark wins when the sentence benchmark is not in the clause
   (the primary's 27.3% was bench=exploitbench, now
   bench=exploitgym); "Kimi K3.1" is a recognised model token
   (the 59.7 was read as bare "Opus's"). Terminal-Bench 50.0 and
   ExploitGym 27.3% no longer read as Opus's.
4. **pass@1 / six-hour pooling.** A digit immediately after "@" is
   a rank specifier, not a claim value ("pass@1" no longer mints
   "1"); a word number followed by a hyphen + duration unit no
   longer mints ("six-hour" no longer mints "six"). The "five runs
   per task" count still mints.
5. **Pipe-table extraction.** `_parse_table_rows` now parses
   pipe-delimited markdown tables (header naming the models, data
   rows naming the benchmark): one claim per cell, subject = column
   model via `_model_slug`. CellCog extraction 0.129 → 0.643.
   Terminal-Bench 24.2, FrontierCode 41.3/24.1/47.3, Terminal-Bench
   50.0 now align with prose. The table-row upgrade is now
   role-consistent with the sentence loop (`_generic_role` rule: a
   non-integer bare decimal is a measurement, an integer-valued bare
   decimal is a task count), and a claim upgraded by one row entry
   cannot be re-upgraded by a second entry with the same value
   (SWE-Atlas QA repeats 73.4 for two models).
6. **Rate percentages flagged.** The single-source-headline rule
   gains two arms: a percentage naming a rate ("44.8% broader
   flag-capture rate") and a generic score whose key carries a
   `bench=` (unmapped benchmark scores, including pipe-table rows).
   Corroboration is subject-tolerant (the merge's own uniqueness
   rule, both directions) and role-tolerant for bench= scores (a
   bare 12.1 and a 12.1% are the same figure), so genuinely shared
   figures are not flagged as lone. Two guards found while
   verifying: (a) corroboration uses a tight same-figure test (1%
   relative) in these two arms, not `_generic_close` — its ±0.15
   window let the OpenAI primary's 0.76 log-perplexity threshold
   "corroborate" ExplainX's 0.819 MAI-2 figure and silently drop
   the run-24 MAI-2 0.819 headline; (b) the bench= arm does not
   flag a number that appears in its own span as an "out of"
   denominator ("scored 33.2 out of 60" mints a 60.0 claim — the
   scale, not a score).
7. **AI-generated byline surfaced.** `detect_ai_generated_byline`
   (narrow disclosure patterns only) is carried on every signal
   brief as `ai_generated_byline`. Unite.AI's note ("this article
   is AI-generated (author Jonas Reeve, described as an AI-generated
   research agent at Unite.AI)") surfaces; a bare mention of AI in
   passing does not trip it. Checked against the run 23–25 feeds:
   zero false positives.
8. **Primary key signals.** The bare "via " caveat cue is gone —
   it fired on plain mechanism ("27.3% flags captured via intended
   exploits"), so the primary's headline sentence never reached the
   key-signal check. Sourcing is still caught by "source:", "data
   provided by", "as of", "estimated", "subject to change".

**NOT fixed — gap 8, image table.** The primary's benchmark table
is an image; no text extraction can read 50.0/41.3 from it. The
figures reach the pipeline only through secondary coverage
(DevPlace, Unite.AI, CellCog). Documented, not attempted.

**NOT fixed — ExploitGym 869 as a claim.** The 869-task figure is
recorded as context (the sentence's context numbers), not minted
as a claim, by design: a bare set size ("over 869 tasks") is
context, and no second source states it as a claim either, so
minting it would surface it only as a unique claim. If a second
source ever states an ExploitGym task count, the context numbers
are what the merge compares.

**Known residuals of this round:**
- Run-26 compare (local, RC10b tree): shared 11 (was 7 local RC10a),
  conflicts 0, unresolved 0 (was 1), agreement `aligned`.
  `single_source_headlines` 5 → 41 — almost all genuinely-lone
  CellCog table values (other models' scores no other doc states)
  plus the 44.8%/18.1% rates; the flag is accurate per figure, but
  the list is long on a table-heavy feed.
- "Five rows" (CellCog, "all five rows of Cognition's own table")
  still mints as a count with subject opus-5-5 via the exactly-one
  rule and is SSH-flagged via its denominator — the figure is true
  (the table has five rows), only the subject label is loose.
  Left as-is: the claim is true and single-source, and the subject
  is the sentence's topic.
- `test_api.py` baseline is RC10a's 159/164 (5 known pre-existing
  failures); see the build note for the RC10b count.

**Standing measurement gates — still open (unchanged from RC10a):**
recall 45% unremeasured; 25% tolerance + +0.12 bump uncalibrated;
held-out relabel OPEN; Reuters coverage gap; Z.ai/Zhipu alias
deferred (do not assert); "2T-3T" unminted, fail-closed; item 8
provenance/independence tracking still a design decision for
Arch/Taylor.

# RC10a update

Bounded correction round on RC10 (Metta's independent verification,
2026-11-25 — Arch's two findings reproduced as real failures).
Seven items fixed, all in `code/signal_compare.py` except the capsule
sections and version stamp in `code/app.py`. Full per-item account
in `RC10a-PORT-SUMMARY.md`; regression tests in
`code/test_review_fixes.py` (`RC10aCompareRegression`, 10 tests) and
`code/test_api.py` (2 new `@test` functions).

1. **Denominator in the figure key.** A figure's identity is now
   (entity key, value, denominator, endpoint, qualifier). The
   denominator is extracted for "out of" phrasing in every minting
   pass (colon, inline, reverse, version-rescan) as well as the
   slash form, and a new `denominator_mismatches` gate (mirroring
   the version gate: only positive disagreement fires) blocks
   comparison of 33.2/60 vs 33.2/100. Both are now separate
   single-source headlines, not an alignment.
2. **Bare endpoint adjectives.** "baseline" mints endpoint `before`,
   "improved" mints `after`; a bare score with neither stays
   unmarked (no guessing). Baseline 47.6 vs improved 47.6 no longer
   align.
3. **Benchmark-path qualifiers.** Threshold words are now extracted
   on benchmark-key claims in every pass, so threshold-vs-point
   routing applies there too. "below 0.8" vs "exactly 0.8" routes
   to clarification, and the tolerance loop is now qualifier-
   symmetric: a point cannot corroborate a threshold-qualified
   focal figure either.
4. **Financial-price tolerance.** Explicitly-quoted currency prices
   get zero excess tolerance in the label-contradiction check, and
   a superlative-labelled value exceeded in the same session routes
   the group to clarification (shared entry dropped). A $100.00
   intraday high vs a $101.50 close is now a confirmed contradiction,
   not an agreement.
5. **Temporal scope on label_contradictions.** Every entry now
   carries `status` ("confirmed" when the sessions do not
   distinguish the claims, else "candidate"), `unit`, `instrument`,
   and per-value `session` fields. A Monday high vs a Tuesday high
   is a candidate inconsistency, not an established contradiction.
6. **Capsule sections.** `build_compare_capsule` now carries
   SINGLE-SOURCE HEADLINES, SHARED CLAIMS EXCLUDED, and LABEL
   CONTRADICTIONS sections (plus a DENOMINATOR MISMATCHES line in
   the mismatch family), with values on every line.
7. **Threshold-vs-point note.** The clarification entry now carries
   a `compatible` flag and a note that distinguishes the two: 0.76
   vs a 0.8 ceiling is compatible; 0.9 vs a 0.8 ceiling VIOLATES
   the bound and is NOT compatible. Both still route to
   clarification (Arch: acceptable for now).

**NOT fixed — item 8 (provenance/independence tracking).**
A secondary repeat of a publisher's figure still erases the
single-source flag (publisher 56.3 vs secondary repeat 56.3
aligns; no provenance signal). This is a design decision for
Arch/Taylor, not a one-line fix, and is documented here rather
than attempted. A versioned org registry (separate publisher /
brand / legal-entity / source-independence fields) is the shape
Arch pointed at; nothing of it is built in this round.

**Standing measurement gates — still open (unchanged from RC10):**
recall 45% unremeasured; 25% tolerance + +0.12 bump uncalibrated;
held-out relabel OPEN; Reuters coverage gap; Z.ai/Zhipu alias
deferred (do not assert); "2T-3T" unminted, fail-closed.

**Known residuals of this round:**
- `test_api.py` own runner is 159/164 — the same 5 pre-existing
  failures as the RC10 baseline (up-to ceiling as compatible;
  benchmark lead-in owns its clause's percentages; before/after
  pair as temporal comparison; Nov 9 packet; Nov 7 ledger
  alignments). No new failures; the 2 new tests pass.
- Finer figure identity moves headline counts: run-25
  `single_source_headlines` 8 → 11, run-23
  `label_contradictions` 1 → 3 (zero currency tolerance surfaces
  more excesses), run-24 `single_source_headlines` 3 → 4. The
  reproduce harnesses still pass 11/11 and 12/12 — they check
  presence, not exact counts.
- The `instrument` field on a `label_contradictions` entry falls
  back to the entity label (e.g. "cost") when no subject was
  extracted; the unit and sessions are the reliable fields.
- Em-dash correction: RC9's `signal_sieve.py` had **2** literal
  `\u2014` sequences, not 8 as the RC10 build note said. The
  conversion itself was functionally identical either way.

# RC9 scope

See REVIEW-HANDOFF.md for the current comparison results and remaining limits. The dated release notes below describe earlier candidates.

## RC8 update

Arch's RC7 conditional accept is answered in RC8: the bar change now
has permanent regression tests — four new tests in
`code/test_review_fixes.py` (`GuardrailBarRegression`): the exact
run-23 press release anchors; a short legitimate release (1 anchored
marker + 6 figures/dates) stays capped, which is the heuristic's
documented boundary, not a bug; a number-heavy but weakly supported
secondary does not anchor even though the language layer verifies it;
and the run-16 Qwen vendor summary declared secondary does not become
`official_release`. The run-23 reproduction evidence is now bundled
in the package (`reproduce/run23/` — script, exact feed files, saved
live RC6 responses) so final verification runs from the package
alone. Known residuals carried forward unchanged: the `figures >= 8`
bar is a heuristic constant like the old `>= 2` (a short formal
release with < 8 figures/dates still caps — now pinned by test);
Fix B re-labels a third-person vendor summary hosted on the vendor's
own domain as `official_release` when declared primary (run-17 Qwen
case — slight overstatement, type already held at primary by the same
exemption; the declared-secondary case is now pinned by test); the
run-23 compare / pricing-tier gaps are untouched by this patch and
remain open. Arch's standing guardrail is preserved: `use_as_anchor`
is custody, not certification.

## RC7 update

The language guardrail's evidence bar (item: run-23 gap 1) is fixed in
RC7: a formal third-person press release from the announcing org no
longer caps at `treat_as_lead` on its own lead sentence. Root cause was
not the first-party exemption — the own-domain exemption already fired
and the main verdict reached `use_as_anchor` — but the language engine
dropping the custody hint because `evidence_anchored` scored 1.0
against a bar of 2; the evidence lexicon barely fires on formal release
prose, though the announcement carries 23 numbers and 5 dates. Fix A
adds a second hint route (1 anchored marker + at least 8 figures/dates).
Fix B re-labels such a release `official_release` via the own-domain
route, narrowed to caller-declared high-trust types. Known residuals
carried forward: the `figures >= 8` bar is a heuristic constant like
the old `>= 2` (a short formal release with < 8 figures/dates would
still cap); Fix B re-labels a third-person vendor summary hosted on the
vendor's own domain as `official_release` (run-17 Qwen case — slight
overstatement, type already held at primary by the same exemption);
no new regression test was added for the bar change — behaviour is
covered by the run-23 reproduce script and the existing suites (see the
RC7 report). Arch's standing guardrail is preserved: `use_as_anchor`
is custody, not certification — a matching official passage never
auto-certifies the whole document.

## RC6 update

The December-byline guard (item 8, cosmetic) is fixed in RC6: a bare
month alone on the byline line (`"By December\n\n<body>"`) no longer
yields a byline or an org slug. The `"By December Smith"` ambiguity
remains — a month word followed by a same-line surname is still read
as a first name. The run-20 figures quoted in item 3 are the saved
live comparison / archived replay, not a fresh result.

## RC5 update

REVIEW-HANDOFF.md supersedes the dated RC4 observations below for the tested CPI lead pattern, unemployment level, calendar byline and UTC warning. Wider extraction coverage remains open.

# Signal Sieve v1.8.9 RC4 — Known limitations (listed separately, not folded into a pass)

Source: correction-pass report §8
(`hidden_files/sieve-1.8.9-correction-pass-report-2026-11-20.md`) and
Worker B's independent RC4 verification
(`hidden_files/rc4-raw/RC4-VERIFICATION.md`). RC4 is the RC3 tree with
only the version stamp changed; every item below was observed in RC3
and re-verified by Worker B.

1. **Nov 10: uniques 109 vs live 92 (+17).** RC4 surfaces more unique
   claims than live v1.8.8. Finer attribution (table rows, attributed
   subjects) keeps more claims unique; down from RC2's 121. Values are
   surfaced, not lost. Not a regression.
2. **Nov 9: 0.330 double-count still present (known residual).** 0.330
   appears both as the temporal `before` endpoint AND as an aligned
   score value (`generic|measurement|score|||sonnet-4-6`, ×2 sources).
   Surfaced via `also_aligned_as` on the temporal entry; not eliminated
   — removing it from either group is an extraction-level dedup. RC4
   also shares 4 more claims than live (34 vs 30), all `value: null`
   non-numeric entries (the memory temporal pair + badges/files/
   links/tokens).
3. **Run 20: shared 19 vs live-saved 24; uniques 146 vs 115 (+31).**
   Explained by finer subject attribution: RC4-only shared entries are
   `value: null` with `gemini-4-neon` subjects; live-only are
   unstated-subject entries. `needs_source_clarification` RC4 0 vs
   live-saved 8 is explained, not a gate failure — live's 8 are all
   "subject unstated on every side" generic discrepancies, and in RC4
   those claims now HAVE subjects (HLE 71.3 aligns, RiemannBench 84.0
   aligns under `bench=riemannbench||gemini-4-neon`), so the
   identity-incompleteness gate correctly does not fire. Nothing is
   hidden. RC4 has 1 `benchmark_mismatch` (Terminal-Bench Science,
   `gemini-4-neon`) vs live 0. The four original run-20 texts are
   absent from this archive; no fresh run-20 replay was performed for
   RC5 — the run-20 figures quoted here are the saved live comparison
   / archived replay, not a fresh RC5 result.
4. **Repo-work divergence — the release candidate is the RC4 tree.**
   In repo-work, `test_nuance` is unrunnable (`ImportError: cannot
   import name '_era_suggestions'` — pre-existing; `_era_suggestions`
   occurs 2× in RC3 and 2× in frozen RC2, 0× in repo-work) and
   `test_first_party` is unrunnable (`ModuleNotFoundError: No module
   named 'statistical_evidence'` — file exists in RC3 and frozen RC2,
   absent in repo-work). Neither is caused by this patch. The RC4 tree
   runs all five suites: 230/230 under Python 3.11.9.
5. **Speed-ratio summary verdict `within_tolerance` with contradictory
   pairings preserved, per Arch's ruling 3.** RC4 run-20
   `speed_ratio_checks` verdict is `within_tolerance` (tolerance 25%):
   same-effort pairing measured 2.5068×, deviation 16.44%, within 25%;
   cross-effort pairing measured 1.7256× (≈1.73×), deviation 42.48%,
   outside tolerance — preserved alongside the summary verdict, not
   folded away. Note: repo-work's run-20 row is the older shape (verdict
   `mixed`, pairings without `comparison_scope`) — a pre-existing nuance
   divergence, not this patch.
6. **The two uncertainty-gate exceptions, accepted by Arch ruling 1.**
   Both documented in code comments in `signal_compare.py`
   (`_identity_gaps`, ~5636–5678): (a) era gap NOT applied — era is
   partition-scoped, mixed-era groups split into `not_comparable`;
   (b) subject gap not applied to mapped claims when subject is unstated
   on EVERY side — extraction gap, not incomplete identity. Both preserve
   pinned behaviour (undated $2 vs $3 must still conflict; GPQA 87.0 vs
   83.5 same-model conflict must still fire).
7. **Nov 10: aligned 44 vs live 43.** The extra alignment is the
   `generic|task_count|days|||unstated` 8.0-days pair (×2), which live
   does not align. The cost $75B vs $14.83 pair sits in
   `unit_mismatches` on both sides, so it is NOT the source of the
   difference. RC4 `not_comparable` is 6 vs live's 7 (RC4 lacks live's
   `pricing` not_comparable entry with unstated/`gpt-6-astra`
   identities).
8. **Other residuals carried forward.**
   - Nov 10 duration/agents/vulnerabilities/days keys still differ from
     live (RC2 attribution gains, e.g. CVE 1829 under its bench) —
     pre-existing, left alone per review.
   - Medium $6.721 per-task stays subject `""` (single-doc unique) —
     correct; no model token attaches in that sentence and it must not
     be forced.
   - Nov 7/8/9/10/run-20: 0 conflicts in RC4 on all five packets (live
     also 0 on all five). Unmatched = 0 on every packet in RC4.
   - Still-open gates from before: recall 45% unremeasured, independent
     held-out relabeling, Nov 7 true-primary downgrade, Reuters coverage
     gap; the 25% tolerance and the +0.12 sponsorship bump uncalibrated;
     the 0.09% vs 0.0% discrepancy still held for human review.
   - Cosmetic: "December" misread as a byline (both versions);
     `datetime.utcnow()` deprecation warning in `app.py:313`.
   - Claim extraction gaps: a genuine Jan-2025 CPI lead yields zero
     statistical claims; "edged down to 4.0 percent" is not extracted
     as a level claim.
