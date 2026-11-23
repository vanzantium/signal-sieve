import json
import unittest
from pathlib import Path
from unittest.mock import patch
from app import app, MAX_TEXT_CHARS
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
