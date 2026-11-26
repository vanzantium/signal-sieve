import json
import unittest
from pathlib import Path
from unittest.mock import patch
from app import app, MAX_TEXT_CHARS
from signal_compare import compare_documents
from signal_sieve import _byline_name, _extract_claimed_org
from signal_sieve_i18n.structural_core import extract_structural
import statistical_evidence as evidence

FIXTURES = Path(__file__).resolve().parent / "tests"

class Handoff(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_private_fields_stay_local_even_for_related_versions(self):
        old, new = (dict(row, reviewer="private initials", operator_note="internal memo")
                    for row in (evidence.RELEASES[0], evidence.RELEASES[-1]))
        with patch.object(evidence, "RELEASES", (old,new)):
            for text in (old["passage"], old["passage"].replace("142,000","143,000")):
                doc = dict(text=text,source_url=old["url"],source_name="BLS",source_type="primary")
                for route, payload in [("/api/v1/analyze",doc),("/api/v1/compare",{"documents":[doc,doc]})]:
                    resp=self.client.post(route,json=payload)
                    self.assertEqual(resp.status_code,200)
                    self.assertIn(old["html_sha256"], resp.get_data(as_text=True))
                    self.assertNotIn(b"private initials",resp.data)
                    self.assertNotIn(b"internal memo",resp.data)

    def test_other_release_cannot_clear_cited_discrepancy(self):
        old=evidence.RELEASES[0]
        other=dict(old, passage=old["passage"].replace("142,000","143,000"),
                   url="https://example.org/other",release_id="other")
        with patch.object(evidence,"RELEASES",(old,other)):
            row=evidence.passage_evidence(other["passage"],old["url"],"primary",[])
        self.assertEqual(row["status"],"numeric_discrepancy")
        self.assertEqual(row["comparison_release"]["release_id"],old["release_id"])
        self.assertEqual(row["matched_releases"][0]["release_id"],"other")

    def test_compare_rejects_partial_document(self):
        doc=dict(text="a"*(MAX_TEXT_CHARS+1))
        r=self.client.post("/api/v1/compare",json={"documents":[{"text":"first"},doc]})
        self.assertEqual(r.status_code,413)
        self.assertEqual(r.get_json()["error"],"text_too_long")

    def test_invalid_metadata(self):
        for route in ("/api/analyze","/api/v1/analyze","/api/v1/analyze.txt","/api/v1/analyze.capsule","/api/v1/compare"):
            for payload in ([1],{"text":"Test","source_name":7}):
                if route.endswith("compare") and isinstance(payload,dict):
                    payload={"documents":[payload,{"text":"another"}]}
                self.assertEqual(self.client.post(route,json=payload).status_code,400,route)

    def test_statistic_scope(self):
        body=("Consumer Price Index for All Urban Consumers (CPI-U) increased 0.5 percent "
              "on a seasonally adjusted basis in January, after rising 0.4 percent in December.")
        rows=evidence.statistics(body)
        self.assertEqual([(r["value"],r["period"],r["kind"]) for r in rows],
                         [(0.5,"January","change"),(0.4,"December","change")])
        self.assertEqual(rows[0]["adjustment"],"seasonally adjusted")
        level=evidence.statistics("The unemployment rate edged down to 4.0 percent in January.")[0]
        self.assertEqual((level["kind"],level["value"]),("level",4.0))

    def test_calendar_word_is_not_byline(self):
        self.assertIsNone(_byline_name("Complete by December 1."))
        self.assertEqual(_byline_name("Complete by December 1. By May Chen"),"May")

    def test_bare_month_byline_line_is_not_byline(self):
        # RC6: bare month alone on the byline line, body text following
        # — the realistic byline shape. Neither a byline nor an org.
        body = "By December\n\nTotal nonfarm payroll employment increased by 142,000 in August."
        self.assertIsNone(_byline_name(body))
        self.assertEqual(_extract_claimed_org(body), "")


class GuardrailBarRegression(unittest.TestCase):
    """RC8 — Arch's RC7 conditional-accept correction (2026-11-22).

    Permanent regression tests for the run-23 guardrail fix (Fix A:
    evidence-bar second route in signal_sieve_i18n/engine.py;
    Fix B: genre override via own-domain in signal_sieve.py).
    The distinction Arch wants preserved is pinned throughout:
    publisher custody, document genre, and claim verification are
    three separate things — a fix to one must not silently move the
    others.
    """

    def setUp(self):
        self.client = app.test_client()

    def analyze(self, text, kind, name, url):
        resp = self.client.post("/api/analyze", json={
            "text": text, "source_type": kind,
            "source_name": name, "source_url": url})
        self.assertEqual(resp.status_code, 200)
        return resp.get_json()

    def test_run23_press_release_anchors(self):
        # (1) The actual press-release case: the exact run-23 primary
        # (Anthropic IPO pricing announcement, feed text as posted to
        # the live service, "Fetched" lines stripped). 1 anchored
        # marker + 28 figures/dates — the second route earns the
        # custody hint the old >= 2 bar dropped, so the primary
        # anchors instead of being capped for the 6th run in a row.
        # Originator voice stays False: the fix does not fake
        # first-person voice; custody comes from the own-domain
        # exemption + claimed publisher org.
        text = (FIXTURES / "run23_anthropic_ipo_announcement.txt").read_text()
        st = extract_structural(text)
        self.assertEqual(st.n_numbers + st.n_dates, 28)
        r = self.analyze(
            text, "primary",
            "Anthropic — Anthropic prices initial public offering (Nov 18, 2026)",
            "https://www.anthropic.com/news/anthropic-prices-initial-public-offering")
        self.assertEqual(r["recommended_action"], "use_as_anchor")
        self.assertEqual(r["source_genre"], "official_release")
        self.assertEqual(r["source_type"], "primary")
        self.assertEqual(r["language_assessment"]["recommended_action"], "verify_primary")
        scores = r["language_assessment"]["scores"]
        self.assertEqual(scores["evidence_anchored"], 1.0)
        self.assertEqual(scores["has_strong_custody"], 1.0)
        self.assertEqual(r["flags"], [])
        self.assertFalse(r["custody_breakdown"]["speaks_as_originator"])
        self.assertEqual(r["custody_breakdown"]["claimed_publisher_org"], "anthropic")

    SHORT_RELEASE = (
        "Source: Acme Corp — \"Acme prices initial public offering\", acme.com. Published 2026-11-20.\n\n"
        "Today Acme Corp priced its initial public offering of common stock at $20.00 per share. "
        "Trading begins Monday on the Nasdaq under the symbol \"ACME.\""
    )

    def test_short_legitimate_release_stays_capped(self):
        # (2) A short legitimate release — 1 anchored marker + 6
        # figures/dates — stays capped. The second route needs >= 8
        # figures/dates, so the custody hint is still dropped and the
        # language guardrail caps the primary at treat_as_lead. This
        # is the heuristic's documented boundary, not a bug: do NOT
        # lower the bar to make this pass. Genre still becomes
        # official_release (Fix B): genre and verdict are independent
        # — custody/genre is not claim verification.
        st = extract_structural(self.SHORT_RELEASE)
        self.assertEqual(st.n_numbers + st.n_dates, 6)
        r = self.analyze(
            self.SHORT_RELEASE, "primary",
            "Acme Corp — Acme prices initial public offering",
            "https://www.acme.com/news/acme-prices-initial-public-offering")
        self.assertEqual(r["recommended_action"], "treat_as_lead")
        self.assertEqual(r["source_genre"], "official_release")
        self.assertEqual(r["source_type"], "primary")
        self.assertEqual(r["language_assessment"]["recommended_action"], "treat_as_lead")
        scores = r["language_assessment"]["scores"]
        self.assertEqual(scores["evidence_anchored"], 1.0)
        self.assertEqual(scores["has_strong_custody"], 0.0)
        self.assertTrue(any("guardrail" in f.lower() for f in r["flags"]))

    NUMBER_STUFFED_SECONDARY = (
        "Source: Reuters — \"Markets rally on strong jobs data\", reuters.com. Published 2026-11-20.\n\n"
        "Stocks rose on Friday after a report showed the economy added 250,000 jobs in November, "
        "beating the 200,000 forecast. The unemployment rate held at 4.2 percent, while wages rose "
        "0.4 percent in November and 4.1 percent over the year. The S&P 500 gained 45 points to "
        "6,120, the Dow added 320 points to 44,500, and the Nasdaq rose 180 points to 19,800. "
        "Bond yields fell to 4.15 percent from 4.28 percent. The data, published November 20, 2026, "
        "also showed retail sales up 0.6 percent in October after a 0.4 percent rise in September. "
        "Analysts expected a 0.3 percent gain. The Federal Reserve meets December 16, 2026. "
        "Oil rose $2.10 to $78.40 a barrel, gold added $15 to $2,650 an ounce, and the dollar "
        "index fell 0.5 percent to 106.2."
    )

    def test_number_stuffed_secondary_does_not_anchor(self):
        # (3) A number-heavy but weakly supported document — 5 anchored
        # markers + 34 figures/dates, own-domain hint firing, language
        # layer verifying — must still NOT anchor. Stuffing a document
        # with numbers can earn the language hint, but the final
        # verdict for a secondary stays treat_as_lead: the guardrail
        # cannot raise, and the custody/genre gates apply downstream.
        # The language layer verifying while the verdict holds is the
        # point, not a contradiction.
        st = extract_structural(self.NUMBER_STUFFED_SECONDARY)
        self.assertGreaterEqual(st.n_numbers + st.n_dates, 8)
        r = self.analyze(
            self.NUMBER_STUFFED_SECONDARY, "secondary",
            "Reuters — Markets rally on strong jobs data",
            "https://www.reuters.com/markets/us/markets-rally-strong-jobs-data-2026-11-20/")
        self.assertEqual(r["recommended_action"], "treat_as_lead")
        self.assertNotEqual(r["recommended_action"], "use_as_anchor")
        self.assertEqual(r["source_genre"], "financial_market_snapshot")
        self.assertEqual(r["language_assessment"]["recommended_action"], "verify_primary")
        scores = r["language_assessment"]["scores"]
        self.assertGreaterEqual(scores["evidence_anchored"], 1.0)
        self.assertEqual(scores["has_strong_custody"], 1.0)

    def test_vendor_summary_not_official_release_when_declared_secondary(self):
        # (4) The run-16 Qwen feed — a third-person vendor summary
        # hosted on the vendor's own domain (qwen.ai), header declares
        # "Qwen Team". Declared secondary (not high-trust), so Fix B's
        # declared-type gate must hold: the genre must NOT become
        # official_release, even though the own-domain exemption fires.
        # (Known overstatement, deliberately NOT pinned here: the same
        # doc declared primary IS labelled official_release by Fix B —
        # see KNOWN-LIMITATIONS.md. That is a label question, not a
        # verdict question: its type was already held at primary by
        # the same exemption.)
        text = (FIXTURES / "run16_qwen_vendor_summary.txt").read_text()
        r = self.analyze(
            text, "secondary",
            "Qwen Team — Qwen4-Max: Your New Teammate Has Arrived (Qwen Blog, dated 2026/11/15)",
            "https://qwen.ai/blog?id=qwen4-max")
        self.assertNotEqual(r["source_genre"], "official_release")
        self.assertEqual(r["source_genre"], "news_article")
        self.assertEqual(r["source_type"], "secondary")
        self.assertEqual(r["recommended_action"], "treat_as_lead")
        self.assertEqual(r["custody_breakdown"]["claimed_publisher_org"], "qwen team")


class RC10aCompareRegression(unittest.TestCase):
    """RC10a (v1.8.10a) — one regression test per fixed verification
    finding. Each test pins the exact probe case from the RC10
    verification receipt (rc10-verification-note-2026-11-25.md) so a
    future change that re-merges these figures fails loudly here
    instead of silently in production."""

    def setUp(self):
        self.client = app.test_client()

    @staticmethod
    def _compare(*docs):
        from signal_compare import compare_documents
        return compare_documents([
            {"text": t, "source_name": n,
             "source_type": "primary" if i == 0 else "secondary",
             "source_url": ""}
            for i, (n, t) in enumerate(docs)
        ])

    @staticmethod
    def _headlines(r, key):
        return [h for h in r.get("single_source_headlines", [])
                if h["entity_key"] == key]

    def test_denominator_slash_form_splits_figures(self):
        # RC10a item 1 (slash): "33.2/60" vs "33.2/100" aligned as the
        # same figure even though the denominators were recorded.
        r = self._compare(
            ("Publisher", "KingBench 4: the model scored 33.2/60."),
            ("Secondary", "KingBench 4: the model scored 33.2/100."))
        self.assertEqual(r.get("alignment"), [])
        mism = r.get("denominator_mismatches", [])
        self.assertEqual(len(mism), 1)
        self.assertEqual(mism[0]["entity_key"], "kingbench")
        self.assertEqual(sorted(mism[0]["denominators"]), ["100.", "60."])
        heads = self._headlines(r, "kingbench")
        self.assertEqual(len(heads), 2)
        self.assertEqual(sorted(h["denominator"] for h in heads),
                         ["100.", "60."])

    def test_denominator_out_of_phrasing_splits_figures(self):
        # RC10a item 1 ("out of"): the denominator was "" for "out of"
        # phrasing — only the slash form recorded it — so "33.2 out of
        # 60" vs "33.2 out of 100" aligned as the same figure.
        r = self._compare(
            ("Publisher",
             "KingBench 4 evaluates reasoning. The model scored 33.2 "
             "out of 60 on KingBench 4, taking 431 minutes."),
            ("Secondary",
             "On KingBench 4, the model scored 33.2 out of 100."))
        self.assertEqual(r.get("alignment"), [])
        mism = r.get("denominator_mismatches", [])
        self.assertEqual(len(mism), 1)
        self.assertEqual(sorted(mism[0]["denominators"]), ["100", "60"])
        heads = self._headlines(r, "kingbench")
        self.assertEqual(len(heads), 2)
        self.assertEqual(sorted(h["denominator"] for h in heads),
                         ["100", "60"])

    def test_bare_baseline_vs_improved_endpoints_do_not_align(self):
        # RC10a item 2: bare "The baseline GPQA score was 47.6." vs
        # "The improved GPQA score was 47.6." aligned as the same
        # figure — only explicit from-to was protected, by abstention.
        r = self._compare(
            ("Publisher", "The baseline GPQA score was 47.6."),
            ("Secondary", "The improved GPQA score was 47.6."))
        self.assertEqual(r.get("alignment"), [])
        self.assertEqual(r.get("shared_claims"), [])
        heads = self._headlines(r, "gpqa")
        self.assertEqual(len(heads), 2)
        self.assertEqual(sorted(h.get("endpoint", "") for h in heads),
                         ["after", "before"])

    def test_benchmark_ceiling_vs_point_routes_to_clarification(self):
        # RC10a item 3: benchmark-path claims minted qualifier null —
        # "below" was lost — so "The GPQA score was below 0.8." vs
        # "exactly 0.8." aligned.
        r = self._compare(
            ("Publisher", "The GPQA score was below 0.8."),
            ("Secondary", "The GPQA score was exactly 0.8."))
        self.assertEqual(r.get("alignment"), [])
        clar = [c for c in r.get("needs_source_clarification", [])
                if c["entity_key"] == "gpqa"]
        self.assertEqual(len(clar), 1)
        self.assertIn("threshold-vs-point", clar[0]["missing"])
        self.assertTrue(clar[0]["compatible"])
        heads = self._headlines(r, "gpqa")
        self.assertEqual(len(heads), 2)
        self.assertEqual(
            sorted(h.get("qualifier", "") for h in heads), ["", "below"])

    def test_focal_benchmark_ceiling_keeps_single_source_flag(self):
        # RC10a item 3 (focal asymmetry): focal "below 0.8" vs point
        # 0.79 aligned and the single-source flag disappeared — the
        # tolerance loop excluded threshold-qualified claims on the
        # other side only, never the focal claim.
        r = self._compare(
            ("Publisher", "The GPQA score was below 0.8."),
            ("Secondary", "The GPQA score was 0.79."))
        self.assertEqual(r.get("alignment"), [])
        heads = self._headlines(r, "gpqa")
        self.assertEqual(len(heads), 2)
        self.assertEqual(
            sorted(h.get("qualifier", "") for h in heads), ["", "below"])
        # And the reverse direction too: focal point, other threshold.
        r2 = self._compare(
            ("Publisher", "The GPQA score was 0.79."),
            ("Secondary", "The GPQA score was below 0.8."))
        self.assertEqual(r2.get("alignment"), [])
        self.assertEqual(len(self._headlines(r2, "gpqa")), 2)

    def test_same_session_price_high_exceeded_is_contradiction(self):
        # RC10a item 4: "Anthropic shares hit an intraday high of
        # $100.00 at noon." vs "closed at $101.50." produced NO
        # contradiction AND aligned as agreement — the 2% tolerance
        # hid a $1.50 excess on a $100 price.
        r = self._compare(
            ("Publisher",
             "Anthropic shares hit an intraday high of $100.00 at noon."),
            ("Secondary", "Anthropic shares closed at $101.50."))
        self.assertEqual(r.get("alignment"), [])
        lcs = r.get("label_contradictions", [])
        self.assertEqual(len(lcs), 1)
        self.assertEqual(lcs[0]["status"], "confirmed")
        self.assertEqual(lcs[0]["unit"], "$")
        self.assertEqual(lcs[0]["superlative_value"]["value"], 100.0)
        self.assertEqual(lcs[0]["exceeding_value"]["value"], 101.5)
        # The verdict loop must not align it either, and the shared
        # entry must be dropped by the shared_claims_excluded gate.
        clar = [c for c in r.get("needs_source_clarification", [])
                if "label-contradiction" in c.get("missing", [])]
        self.assertEqual(len(clar), 1)
        self.assertEqual(r.get("shared_claims"), [])
        self.assertEqual(len(r.get("shared_claims_excluded", [])), 1)

    def test_cross_session_price_high_is_candidate_not_confirmed(self):
        # RC10a item 5: "On Monday, ... high of $111.50." vs "On
        # Tuesday, ... $114.99." was flagged with NO date/session/unit/
        # instrument fields — both can be correct.
        r = self._compare(
            ("Publisher",
             "On Monday, Anthropic shares hit an intraday high of $111.50."),
            ("Secondary",
             "On Tuesday, Anthropic shares hit an intraday high of $114.99."))
        lcs = r.get("label_contradictions", [])
        self.assertEqual(len(lcs), 1)
        self.assertEqual(lcs[0]["status"], "candidate")
        self.assertEqual(lcs[0]["unit"], "$")
        self.assertEqual(lcs[0]["superlative_value"]["session"], "monday")
        self.assertEqual(lcs[0]["exceeding_value"]["session"], "tuesday")
        self.assertIn("both may be correct", lcs[0]["note"])
        # Different sessions must NOT trip the verdict-loop
        # superlative guard: no label-contradiction clarification.
        clar = [c for c in r.get("needs_source_clarification", [])
                if "label-contradiction" in c.get("missing", [])]
        self.assertEqual(clar, [])

    def test_threshold_vs_point_compatible_flag_true(self):
        # RC10a item 7 (compatible): 0.76 vs a ceiling of 0.8 IS
        # compatible — the point sits inside the bound.
        r = self._compare(
            ("Publisher", "The score must stay below 0.8."),
            ("Secondary", "The measured score was 0.76."))
        clar = r.get("needs_source_clarification", [])
        self.assertEqual(len(clar), 1)
        self.assertTrue(clar[0]["compatible"])
        self.assertIn("compatible but do not assert", clar[0]["note"])

    def test_threshold_vs_point_incompatible_flag_false(self):
        # RC10a item 7 (incompatible): 0.9 vs a ceiling of 0.8 is NOT
        # compatible — the point violates the bound — and the note must
        # say so instead of asserting compatibility.
        r = self._compare(
            ("Publisher", "The score must stay below 0.8."),
            ("Secondary", "The measured score was 0.9."))
        clar = r.get("needs_source_clarification", [])
        self.assertEqual(len(clar), 1)
        self.assertFalse(clar[0]["compatible"])
        self.assertIn("VIOLATES", clar[0]["note"])
        self.assertIn("NOT compatible", clar[0]["note"])

    def test_compare_capsule_carries_new_field_sections(self):
        # RC10a item 6: POST /api/v1/compare returns all three fields
        # in JSON but the capsule mentioned none of them.
        docs = [
            {"text": "KingBench 4: the model scored 33.2/60. "
                     "The model has 750B parameters.",
             "source_name": "Publisher", "source_type": "primary"},
            {"text": "On Monday, Anthropic shares hit an intraday high "
                     "of $111.50.",
             "source_name": "Secondary-A", "source_type": "secondary"},
            {"text": "On Tuesday, Anthropic shares hit an intraday high "
                     "of $114.99.",
             "source_name": "Secondary-B", "source_type": "secondary"},
        ]
        resp = self.client.post("/api/v1/compare", json={"documents": docs})
        self.assertEqual(resp.status_code, 200)
        d = resp.get_json()
        for field in ("shared_claims_excluded", "single_source_headlines",
                      "label_contradictions"):
            self.assertIn(field, d)
            self.assertTrue(d[field], f"{field} should be non-empty here")
        cap = d["capsule"]
        for needle in ("SINGLE", "LABEL", "CONTRADICTION", "EXCLUDED",
                       "33.2", "111.50"):
            self.assertIn(needle, cap, f"capsule missing {needle!r}")


class RC10bRun26Regression(unittest.TestCase):
    """RC10b (v1.8.10b) — one regression test per fixed run-26 gap.
    Each test pins the exact run-26 probe case (2026-11-25 Cognition
    SWE-2.1 feeds) so a future change that re-breaks one of these
    fails loudly here instead of silently in production."""

    def setUp(self):
        self.client = app.test_client()

    @staticmethod
    def _compare(*docs):
        from signal_compare import compare_documents
        return compare_documents([
            {"text": t, "source_name": n,
             "source_type": "primary" if i == 0 else "secondary",
             "source_url": ""}
            for i, (n, t) in enumerate(docs)
        ])

    @staticmethod
    def _claims(text, name="Doc"):
        from signal_compare import (extract_numeric_claims,
                                    extract_generic_quantitative_claims)
        base = extract_numeric_claims(text)
        return base + extract_generic_quantitative_claims(text, name, base)

    def test_first_person_methodology_text_keeps_primary(self):
        # RC10b gap 1: the Cognition primary was downgraded to
        # secondary because "We report ... self-reported scores" and
        # "Other models are reported in mini-swe-agent" tripped the
        # bare "reported" secondary-reporting veto. Methodology
        # language is not secondary reporting — the exemption fires
        # and the primary anchors.
        text = (
            "We are releasing SWE-2.1 today, our new coding model, and we are proud of the progress "
            "it represents for our team and our users. "
            "SWE-2.1 achieves 80% on ExploitBench and 27.3% flags captured via intended exploits on ExploitGym. "
            "On Terminal-Bench 4.1, SWE-2.1 scores 50.0, up from 24.2 for SWE-2, and on FrontierCode 2.0 it scores 41.3, up from 24.1. "
            "We report upstream or self-reported scores if they are available, and we evaluate every model on our own internal evaluation framework. "
            "Other models are reported in mini-swe-agent with each model's native context window, up to about one million tokens. "
            "We run SWE-2.1 and SWE-2 in our harness with a 250k-token context, with up to three compactions for one million tokens total. "
            "The model was trained on a large corpus of code and documentation, and we measured its performance across five benchmarks. "
            "We believe these results show meaningful progress, and we will continue to publish our evaluations as we improve the model further.")
        from signal_sieve import analyze
        a = analyze(text, source_type="primary",
                    source_name="Cognition — Introducing SWE-2.1 (Nov 24, 2026)",
                    source_url="https://cognition.com/blog/swe-2-1")
        self.assertEqual(a.get("recommended_action"), "use_as_anchor")
        self.assertEqual(a.get("flags"), [])

    def test_genuine_secondary_reporting_still_vetoes(self):
        # RC10b gap 1 (guardrail): the methodology exemptions are
        # narrow — a third-party reporting verb ("reported in The
        # Times") still vetoes the exemption even with announcement
        # voice present.
        from signal_sieve import _first_party_announcement
        fires, _reason = _first_party_announcement(
            "We are releasing a new model today. "
            "The results are reported in The Times this morning.")
        self.assertFalse(fires)

    def test_unite_ai_publisher_parse(self):
        # RC10b gap 2: "Unite.AI" parsed as publisher "unite" with a
        # NULL feed-declared publisher (same class as "Z.ai", fixed
        # in RC10). The RC10a dotted-name parse covers it — this test
        # locks it in.
        from signal_sieve import _feed_declared_org
        self.assertEqual(
            _feed_declared_org("Unite.AI — Cognition Releases SWE-2.1 With Major Coding and Cybersecurity Gains"),
            "unite.ai")
        self.assertEqual(_feed_declared_org("Z.ai — GLM-5.4 announcement"), "z.ai")

    def test_trails_comparator_and_semicolon_scope(self):
        # RC10b gap 3: Terminal-Bench 50.0 and ExploitGym 27.3% were
        # both labelled Claude Opus 5.5. "trails Claude Opus 5.5" names
        # Opus as the baseline, not the subject, and the 27.3% sits in
        # a different semicolon segment from the Opus mention.
        doc_a = ("It still trails Claude Opus 5.5 on all five rows of Cognition's own table. "
                 "SWE-2.1 scores 50.0 on Terminal-Bench 4.1.")
        doc_b = ("Cognition reports 80.0 on ExploitBench (SWE-2: 36.1) and 27.3% success on ExploitGym "
                 "(SWE-2: 12.1); in the same table Claude Opus 5.5 is listed at 91 and 34.5.")
        r = self._compare(("A", doc_a), ("B", doc_b))
        for group in r.get("shared_claims", []) + r.get("alignment", []):
            for v in group.get("values", []):
                self.assertNotEqual(v.get("subject"), "opus-5-5",
                                    f"Opus misattribution survives in {group.get('entity_key')}")
        # The 50.0 and 27.3% claims exist and are not Opus's.
        claims = self._claims(doc_a, "A") + self._claims(doc_b, "B")
        fifty = [c for c in claims if c.get("numeric_value") == 50.0]
        self.assertTrue(fifty, "50.0 claim missing")
        self.assertTrue(all(c.get("subject") != "opus-5-5" for c in fifty))
        twenty = [c for c in claims if c.get("numeric_value") == 27.3]
        self.assertTrue(twenty, "27.3 claim missing")
        self.assertTrue(all(c.get("subject") != "opus-5-5" for c in twenty))

    def test_baseline_number_stays_unstated(self):
        # RC10b gap 3 follow-on: "59.7 against 50.0" — the 59.7 is
        # Kimi K3.1's (new model-token alternative), the 50.0 is the
        # baseline it is measured against and stays unstated instead
        # of taking the sentence's sole model.
        text = ("Kimi K3.1, a newer model from the family SWE-2.1 is trained on, also scores higher "
                "on Terminal-Bench 4.1 (59.7 against 50.0).")
        claims = self._claims(text)
        fifty_nine = [c for c in claims if c.get("numeric_value") == 59.7]
        self.assertTrue(fifty_nine, "59.7 claim missing")
        self.assertEqual(fifty_nine[0].get("subject"), "kimi-k3-1")
        fifty = [c for c in claims if c.get("numeric_value") == 50.0]
        self.assertTrue(fifty, "50.0 claim missing")
        self.assertEqual(fifty[0].get("subject"), "")

    def test_pass_at_and_hyphen_duration_do_not_mint_task_counts(self):
        # RC10b gap 4: pass@1's "1" pooled with "six" (six-hour
        # timeout) as ExploitGym task counts; the real 869-task figure
        # never compared. Neither the rank specifier nor the hyphenated
        # duration mints as a task count.
        text = ("ExploitGym uses pass@1, meaning one evaluated attempt, over 869 tasks "
                "with a six-hour timeout and mitigations disabled, while ExploitBench "
                "combines captured flags from five runs per task.")
        claims = self._claims(text)
        pooled = [c for c in claims
                  if c.get("numeric_value") in (1.0, 6.0)
                  and "task_count" in c.get("entity_key", "")
                  and "tasks" in c.get("entity_key", "")]
        self.assertEqual(pooled, [], f"pass@1/six-hour still mint as task counts: {pooled}")
        # The genuine count in the sentence still mints.
        five = [c for c in claims if c.get("numeric_value") == 5.0]
        self.assertTrue(five, "five runs claim missing")

    def test_pipe_table_parses(self):
        # RC10b gap 5: the CellCog pipe-delimited table was invisible
        # (0.129 extraction) — Terminal-Bench 24.2, FrontierCode 41.3,
        # ProgramBench 44.0/17.0, SWE-Atlas 71.0, SWE Marathon 54.9
        # never aligned. One claim per cell, subject = column model.
        table = (
            "Benchmark | SWE-2.1 | SWE-2 | Kimi K3.1 | Claude Opus 5.5\n"
            "Terminal-Bench 4.1 | 50.0 | 24.2 | 59.7 | 65.8\n"
            "FrontierCode 2.0 | 41.3 | 24.1 | 19.9 | 47.3\n"
            "ProgramBench | 44.0 | 17.0 | 31.5 | 65.0\n"
            "SWE-Atlas QA | 71.0 | 63.7 | 70.2 | 77.4\n"
            "SWE Marathon v1.2 | 54.9 | 32.5 | 38.8 | 73.1\n")
        claims = self._claims(table, "CellCog")
        want = {
            ("terminal-bench", 50.0, "swe-2-1"),
            ("terminal-bench", 24.2, "swe-2"),
            ("frontiercode", 41.3, "swe-2-1"),
            ("programbench", 44.0, "swe-2-1"),
            ("programbench", 17.0, "swe-2"),
            ("swe-atlas-qa", 71.0, "swe-2-1"),
            ("swe-marathon", 54.9, "swe-2-1"),
            ("terminal-bench", 65.8, "opus-5-5"),
        }
        got = set()
        for c in claims:
            key = c.get("entity_key", "")
            if "bench=" in key and c.get("metric") == "score":
                bench = key.split("bench=")[1].split("|")[0]
                got.add((bench, c.get("numeric_value"), c.get("subject", "")))
        for w in want:
            self.assertIn(w, got, f"table claim missing: {w}")
        # And the table values align with prose stating the same figures.
        prose = ("Cognition released SWE-2.1 on November 24, 2026, reporting a Terminal-Bench 4.1 score "
                 "of 50.0, up from 24.2 for SWE-2. Its FrontierCode 2.0 score reached 41.3, compared with 24.1 for SWE-2.")
        r = self._compare(("CellCog", table), ("Prose", prose))
        aligned_keys = {g.get("entity_key") for g in r.get("shared_claims", [])}
        self.assertTrue(any("terminal-bench" in k and "swe-2-1" in k for k in aligned_keys),
                        f"table 50.0 did not align: {aligned_keys}")
        self.assertTrue(any("frontiercode" in k and "swe-2-1" in k for k in aligned_keys),
                        f"table 41.3 did not align: {aligned_keys}")

    def test_rate_percentage_flagged_single_source(self):
        # RC10b gap 6: the 44.8% broader flag-capture rate (Unite.AI
        # only) had no presence flag — the headline rule needed a
        # denominator, a mapped benchmark key, or a perplexity metric.
        # A percentage naming a rate is a headline figure too.
        doc = ("Its broader flag-capture rate reached 44.8%, compared with 18.1%; that metric also "
               "counts successful real exploits unrelated to the vulnerability specified in the initial prompt.")
        other = ("Cognition reports 80% on ExploitBench and 27.3% success on ExploitGym.")
        r = self._compare(("Unite.AI", doc), ("Other", other))
        heads = [h for h in r.get("single_source_headlines", [])
                 if h.get("value") == 44.8]
        self.assertTrue(heads, "44.8% rate not flagged as a single-source headline")

    def test_ai_generated_byline_surfaced(self):
        # RC10b gap 7: Unite.AI's AI-generated research-agent byline
        # (Jonas Reeve) never surfaced. The brief now carries it, and
        # a bare mention of AI in passing does not trip it.
        # RC10c item 3: the boolean is 'disclosed_ai_generated' —
        # the field reports what the document discloses, not a
        # detection of fact.
        from signal_extract import detect_ai_generated_byline, extract_signal_brief
        t = ("Cognition released SWE-2.1. Note: this article is AI-generated (author Jonas Reeve, "
             "described as an AI-generated research agent at Unite.AI) and reviewed by the editorial team.")
        found = detect_ai_generated_byline(t)
        self.assertTrue(found["disclosed_ai_generated"])
        self.assertIn("Jonas Reeve", found["detail"])
        brief = extract_signal_brief(t, "news_article", "secondary")
        self.assertTrue(brief.get("ai_generated_byline", {}).get("disclosed_ai_generated"))
        plain = "The team uses AI to write code faster, and the model was trained with AI assistance."
        self.assertFalse(detect_ai_generated_byline(plain)["disclosed_ai_generated"])

    def test_denominator_not_flagged_as_score(self):
        # RC10b gap 6 follow-on (caught by the RC10a A1 probe): the
        # bench= headline arm must not flag a denominator as a score.
        # "33.2 out of 60" mints a 60.0 KingBench-score claim (the
        # generic pass mints the "out of" number too); the 33.2s are
        # still flagged, the 60.0 is not. (Docs are the RC10a A1 probe
        # pair verbatim.)
        doc_a = "KingBench 4 evaluates reasoning. The model scored 33.2 out of 60 on KingBench 4, taking 431 minutes."
        doc_b = "On KingBench 4, the model scored 33.2 out of 100."
        r = self._compare(("Publisher", doc_a), ("Secondary", doc_b))
        heads = r.get("single_source_headlines", [])
        self.assertTrue(any(h.get("value") == 33.2 for h in heads),
                        "33.2 headlines missing")
        self.assertFalse(any(h.get("value") == 60.0 for h in heads),
                         f"denominator 60.0 flagged as a headline: {heads}")
        self.assertTrue(r.get("denominator_mismatches"),
                        "denominator mismatch gate did not fire")

    def test_via_mechanism_not_a_caveat(self):
        # RC10b gap 8: the primary's headline sentence was classified
        # as a source_caveat — the bare "via " cue fired on plain
        # mechanism ("via intended exploits") — so it never reached
        # the key-signal check.
        from signal_sieve import analyze
        text = ("SWE-2.1 achieves 80% on ExploitBench and 27.3% flags captured via intended exploits "
                "on ExploitGym. The model was evaluated on five benchmarks and the results were strong.")
        a = analyze(text, source_type="primary", source_name="Cognition",
                    source_url="https://cognition.com/blog/swe-2-1")
        brief = a.get("signal_brief", {})
        self.assertIn("SWE-2.1 achieves 80% on ExploitBench and 27.3% flags captured via intended exploits on ExploitGym.",
                      brief.get("key_signals", []))
        self.assertEqual(brief.get("source_caveats"), [])


class RC10cRegression(unittest.TestCase):
    """RC10c (v1.8.10b-first-party-rc10c) — one regression test per
    bounded item in the 2026-11-25 independent verification receipt.
    Each test pins the exact probe case so a future change that
    re-breaks one of these fails loudly here instead of silently in
    production."""

    def setUp(self):
        self.client = app.test_client()

    @staticmethod
    def _compare(*docs):
        from signal_compare import compare_documents
        return compare_documents([
            {"text": t, "source_name": n,
             "source_type": "primary" if i == 0 else "secondary",
             "source_url": ""}
            for i, (n, t) in enumerate(docs)
        ])

    @staticmethod
    def _generic_claims(text, name="Doc"):
        from signal_compare import (extract_numeric_claims,
                                    extract_generic_quantitative_claims)
        base = extract_numeric_claims(text)
        return base + extract_generic_quantitative_claims(text, name, base)

    def test_from_value_mints_under_lead_in_benchmark(self):
        # RC10c item 1 (F2): the from-endpoint of a from-to pair
        # mints under a lead-in benchmark, with endpoint "before",
        # the way the to-value mints with endpoint "after". RC10's
        # blanket "X to Y" skip dropped both.
        claims = self._generic_claims(
            "- On Last Secure Code Bench, function-level pass rate rose "
            "from 22.0% to 31.8%, and the rate of passing all three "
            "granularities rose from 8% to 14.7%.")
        by_value = {c["numeric_value"]: c for c in claims
                    if c.get("entity_key", "").startswith(
                        "generic|percentage|score|bench=last-secure-code-bench")}
        self.assertIn(22.0, by_value)
        self.assertIn(31.8, by_value)
        self.assertEqual(by_value[22.0].get("endpoint"), "before")
        self.assertEqual(by_value[31.8].get("endpoint"), "after")

    def test_from_to_pair_mints_both_endpoints(self):
        # RC10c item 1 (F3): "Sonnet 4.6 barely moved (0.330 to
        # 0.364)" mints both memory scores, before/after. The
        # gapped shape ("from 0.330 at baseline to 0.641") marks
        # its endpoints too.
        claims = self._generic_claims(
            "Sonnet 4.6 barely moved (0.330 to 0.364) until prompted "
            "to read its memory and test its theories, which took it "
            "to 0.641.")
        by_value = {c["numeric_value"]: c for c in claims
                    if c.get("metric") == "memory"}
        self.assertIn(0.33, by_value)
        self.assertIn(0.364, by_value)
        self.assertEqual(by_value[0.33].get("endpoint"), "before")
        self.assertEqual(by_value[0.364].get("endpoint"), "after")
        claims2 = self._generic_claims(
            "The score went from 0.330 at baseline to 0.641 with memory.")
        by_value2 = {c["numeric_value"]: c for c in claims2}
        self.assertEqual(by_value2[0.33].get("endpoint"), "before")
        self.assertEqual(by_value2[0.641].get("endpoint"), "after")

    def test_over_sessions_is_not_a_threshold(self):
        # RC10c item 1 (F4): "dreamed over 128 sessions" is temporal
        # ("during"), not a floor — the qualifier must not divert an
        # identical-value group to clarification. And "more than five
        # times the 166" governs the ratio, not the 166 (F5).
        claims = self._generic_claims(
            "Opus 5 agents dreamed over 128 sessions.")
        c128 = [c for c in claims if c.get("numeric_value") == 128.0]
        self.assertTrue(c128)
        self.assertEqual(c128[0].get("qualifier"), "")
        r = self._compare(
            ("A", "Opus 5 agents dreamed over 128 sessions."),
            ("B", "Opus 5 agents dreamed 128 sessions."))
        aligned = [a for a in r.get("alignment", [])
                   if "sessions" in a.get("entity_key", "")]
        self.assertTrue(aligned,
                        "over-128 sessions must align, not clarify")
        self.assertFalse([c for c in r.get("needs_source_clarification", [])
                          if "sessions" in c.get("entity_key", "")])

    def test_methodology_exemption_outlet_names_still_veto(self):
        # RC10c item 2: "reported in reuters" / "reported in nature"
        # are secondary reporting (lowercase-styled outlets), while
        # "reported in mini-swe-agent" stays exempt and "reported in
        # The Times" still vetoes.
        from signal_sieve import _reported_cue_present, analyze
        self.assertFalse(_reported_cue_present(
            "Other models are reported in mini-swe-agent."))
        self.assertTrue(_reported_cue_present(
            "The results were reported in reuters."))
        self.assertTrue(_reported_cue_present(
            "The results were reported in nature."))
        self.assertTrue(_reported_cue_present(
            "The results were reported in The Times."))
        self.assertFalse(_reported_cue_present(
            "We report upstream or self-reported scores."))
        # Pipeline level: the same first-person primary text the
        # RC10b methodology test uses — with the harness swapped for
        # an outlet, the first-party exemption must not apply and the
        # downgrade flag must fire. (The action differs from the real
        # feed's treat_as_lead because the full scoring turns on the
        # whole document; the downgrade flag is the product of this
        # fix, so that is what the test pins.)
        text = (
            "We are releasing SWE-2.1 today, our new coding model, and we are proud of the progress "
            "it represents for our team and our users. "
            "SWE-2.1 achieves 80% on ExploitBench and 27.3% flags captured via intended exploits on ExploitGym. "
            "On Terminal-Bench 4.1, SWE-2.1 scores 50.0, up from 24.2 for SWE-2, and on FrontierCode 2.0 it scores 41.3, up from 24.1. "
            "We report upstream or self-reported scores if they are available, and we evaluate every model on our own internal evaluation framework. "
            "Other models are reported in mini-swe-agent with each model's native context window, up to about one million tokens. "
            "We run SWE-2.1 and SWE-2 in our harness with a 250k-token context, with up to three compactions for one million tokens total. "
            "The model was trained on a large corpus of code and documentation, and we measured its performance across five benchmarks. "
            "We believe these results show meaningful progress, and we will continue to publish our evaluations as we improve the model further.")
        a = analyze(text, source_type="primary",
                    source_name="Cognition — Introducing SWE-2.1 (Nov 24, 2026)",
                    source_url="https://cognition.com/blog/swe-2-1")
        self.assertEqual(a.get("recommended_action"), "use_as_anchor")
        self.assertEqual(a.get("flags"), [])
        for outlet in ("reuters", "nature"):
            t = text.replace("reported in mini-swe-agent",
                             f"reported in {outlet}")
            ao = analyze(t, source_type="primary",
                         source_name="Cognition — Introducing SWE-2.1 (Nov 24, 2026)",
                         source_url="https://cognition.com/blog/swe-2-1")
            self.assertTrue(
                any("downgraded to 'secondary'" in f
                    and "secondary-reporting cue present ('reported')" in f
                    for f in ao.get("flags", [])),
                f"expected the secondary-reporting downgrade flag for {outlet}, got {ao.get('flags')}")

    def test_byline_denial_and_quote_are_not_disclosures(self):
        # RC10c item 3: a denial and a quotation about another article
        # are not disclosures; a direct disclosure still is. The
        # field is 'disclosed_ai_generated' — what the document
        # discloses, not a detection of fact.
        from signal_extract import detect_ai_generated_byline
        self.assertTrue(detect_ai_generated_byline(
            "This article is AI-generated.")["disclosed_ai_generated"])
        self.assertFalse(detect_ai_generated_byline(
            "The researcher denied that this article is AI-generated."
            )["disclosed_ai_generated"])
        self.assertFalse(detect_ai_generated_byline(
            'The website says "this article is AI-generated" about a '
            "different story.")["disclosed_ai_generated"])
        self.assertTrue(detect_ai_generated_byline(
            "The author is an AI-generated research agent."
            )["disclosed_ai_generated"])

    def test_competing_subjects_in_same_table_keep_flags(self):
        # RC10c item 4: one unstated prose figure must not clear the
        # single-source flags of BOTH Grok 73.4 and Muse 73.41 in the
        # same table — the merge refuses to attach it to either, so
        # the flags must stand. With a single subject, the flag still
        # drops (corroboration).
        table = ("Benchmark | Grok 4.7 | Muse Wonder\n"
                 "FrontierCode 2.0 | 73.4 | 73.41\n")
        prose = "Its FrontierCode 2.0 score reached 73.4."
        r = self._compare(("TableDoc", table), ("ProseDoc", prose))
        ssh = {(h["entity_key"], h["value"])
               for h in r.get("single_source_headlines", [])}
        self.assertIn(
            ("generic|measurement|score|bench=frontiercode||grok-4-7", 73.4), ssh)
        self.assertIn(
            ("generic|measurement|score|bench=frontiercode||muse-wonder", 73.41), ssh)
        self.assertFalse(r.get("alignment"),
                         "ambiguous prose figure must not attach")
        single = ("Benchmark | Grok 4.7 | Kimi K3.1\n"
                  "FrontierCode 2.0 | 73.4 | 70.2\n")
        r2 = self._compare(("TableDoc", single), ("ProseDoc", prose))
        ssh2 = {(h["entity_key"], h["value"])
                for h in r2.get("single_source_headlines", [])}
        self.assertNotIn(
            ("generic|measurement|score|bench=frontiercode||grok-4-7", 73.4), ssh2)

    def test_five_rows_not_attributed_to_opus(self):
        # RC10c item 5: the count of Cognition's table rows is not
        # Claude Opus 5.5's figure. The claim still mints (the table
        # has five rows) but the subject abstains.
        claims = self._generic_claims(
            "It still trails Claude Opus 5.5 on all five rows of "
            "Cognition's own table, which one reply asked about.")
        rows = [c for c in claims if c.get("metric") == "rows"]
        self.assertTrue(rows)
        self.assertEqual(rows[0].get("subject"), "")
        self.assertNotIn("opus-5-5", rows[0].get("entity_key", ""))

    def test_duplicate_table_value_keeps_both_subjects(self):
        # RC10c item 7: the CellCog SWE-Atlas QA row repeats 73.4
        # for Grok 4.7 and Muse Wonder — both claims survive. The
        # dedup keeps BOTH (one claim per model cell), documented in
        # KNOWN-LIMITATIONS.md.
        text = ("| Benchmark | SWE-2.1 | SWE-2 Flash | Claude Opus 5.5 | Grok 4.7 | Muse Wonder | GPT-6.2 | Kimi K3 |\n"
                "|---|---|---|---|---|---|---|---|\n"
                "| SWE-Atlas QA | 71.0 | 63.7 | 70.2 | 73.4 | 73.4 | 71.8 | 61.5 |\n")
        claims = self._generic_claims(text)
        subjects = sorted(
            c.get("subject") for c in claims
            if c.get("numeric_value") == 73.4
            and c.get("benchmark") == "SWE-Atlas QA")
        self.assertEqual(subjects, ["grok-4-7", "muse-wonder"])

    def test_from_to_pair_benchmark_stays_with_its_own_pair(self):
        # RC10c item 1 follow-on: a sentence naming two benchmarks with
        # one from-to pair each must not attach the first pair's figures
        # to the second benchmark. The forward "on <Name>" attachment
        # in _generic_benchmark must not cross the closing paren of the
        # parenthetical containing the number.
        # NOTE: the 41.3 to-value does not mint in this isolated
        # sentence — _generic_metric finds no cue for it (the
        # "FrontierCode" cue is claimed by the nearer 24.1) — and
        # that is pre-existing RC9 behavior, not an RC10c gap
        # (verified: RC9 returns None for 41.3 here too). In the
        # full CellCog doc the 41.3 figure still surfaces via the
        # table pass.
        text = ("SWE-2.1 roughly doubles SWE-2 on Terminal-Bench 4.1 "
                "(24.2 to 50.0) and on FrontierCode 2.0 (24.1 to 41.3).")
        claims = self._generic_claims(text)
        got = {(c.get("benchmark"), c.get("numeric_value")) for c in claims}
        self.assertIn(("Terminal-Bench", 24.2), got)
        self.assertIn(("Terminal-Bench", 50.0), got)
        self.assertIn(("FrontierCode", 24.1), got)
        self.assertNotIn(("FrontierCode", 50.0), got)
        self.assertNotIn(("FrontierCode", 24.2), got)

    def test_plain_point_corroborates_from_to_endpoint(self):
        # RC10c item 1 follow-on: a from-to endpoint is not a lone
        # figure when another doc states the same value as a plain
        # point. Without this, item 1's newly-minted from-values were
        # listed as BOTH shared/aligned and single-source headlines
        # (the SSH corroboration arms required exact endpoint
        # equality). Before vs after stay distinct: a baseline 47.6
        # and an improved 47.6 are different figures (RC10a rule).
        docs = [
            {"text": "ExploitBench from 36.1% to 80%.",
             "source_name": "DevPlace"},
            {"text": "ExploitBench 80%, up from 36.1%.",
             "source_name": "Unite"},
        ]
        d = compare_documents(docs)
        ssh = {(c.get("entity_key"), c.get("value"))
               for c in d.get("single_source_headlines", [])}
        self.assertNotIn(
            ("generic|percentage|score|bench=exploitbench||unstated", 36.1), ssh)
        self.assertNotIn(
            ("generic|percentage|score|bench=exploitbench||unstated", 80.0), ssh)
        # Control: a lone pair with no corroborating doc stays flagged.
        d2 = compare_documents([
            {"text": "ExploitBench from 36.1% to 80%.",
             "source_name": "DevPlace"},
            {"text": "The weather was fine and the launch went well.",
             "source_name": "Other"},
        ])
        ssh2 = {(c.get("entity_key"), c.get("value"))
                for c in d2.get("single_source_headlines", [])}
        self.assertIn(
            ("generic|percentage|score|bench=exploitbench||unstated", 36.1), ssh2)
        # Control: before vs after do NOT corroborate each other.
        d3 = compare_documents([
            {"text": "GPQA improved from 47.6 to 56.3.",
             "source_name": "A"},
            {"text": "GPQA improved to 47.6.",
             "source_name": "B"},
        ])
        ssh3 = {(c.get("entity_key"), c.get("value"))
                for c in d3.get("single_source_headlines", [])}
        self.assertIn(("gpqa", 47.6), ssh3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
