"""First-party receipt, scope, extraction and local registration contracts."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from app import app
import statistical_evidence as evidence
sys.path.insert(0, str(Path(__file__).parent / "evidence"))
from register_release import register

PAY, CPI, PPI, NEXT_CPI, REVISION = evidence.RELEASES[:5]
NEWS = "https://www.reuters.com/markets/us/view-august-us-payrolls-short-expectations-boosts-bigger-rate-cut-view-2024-09-06/"
FRESH = "Total nonfarm payroll employment rose by 227,000 in November, and the unemployment rate changed little at 4.2 percent"
FRESH_URL = "https://www.bls.gov/news.release/archives/empsit_12062024.htm"

class Receipts(unittest.TestCase):
    def request(self, text, url="", name="BLS", kind="primary", **extra):
        doc = dict(text=text, source_url=url, source_name=name, source_type=kind, **extra)
        response = app.test_client().post("/api/v1/analyze", json=doc)
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_original_targets(self):
        for record, expected in [(PAY, [(142000,"change","August"), (4.2,"level","August")]),
                                 (CPI, [(0.1,"change","August"), (0.2,"change","July")]),
                                 (PPI, [(0.2,"change","August")])]:
            r = self.request(record["passage"], record["url"])
            self.assertEqual(r["resolved_source_type"], "primary")
            self.assertEqual(r["passage_evidence"]["status"], "matched_release_excerpt")
            self.assertTrue(r["custody_breakdown"]["primary_source_present"])
            self.assertNotEqual(r["recommended_action"], "use_as_anchor")
            self.assertEqual([(c["value"],c["kind"],c["period"]) for c in r["statistical_claims"]],expected)
            for c in r["statistical_claims"]:
                self.assertEqual(record["passage"][c["start"]:c["end"]],c["span"])

    def test_alteration_and_injected_receipt(self):
        changed = PAY["passage"].replace("142,000","143,000")
        r = self.request(changed, PAY["url"], passage_evidence={"status":"matched_release_excerpt"})
        self.assertEqual(r["statistical_claims"][0]["value"],143000)
        self.assertEqual(r["passage_evidence"]["status"],"numeric_discrepancy")
        self.assertEqual(r["passage_evidence"]["comparison_release"]["passage"], PAY["passage"])
        self.assertNotEqual(r["source_type"],"primary")
        self.assertFalse(r["passage_evidence"]["matched_releases"])
        self.assertEqual(r["passage_evidence"]["differing_numbers"],[{"submitted":"143,000","publisher":"142,000"}])

    def test_mirror_and_wrapper(self):
        for url, body in [(NEWS,PAY["passage"]),(PAY["url"],'According to BLS, "' + PAY["passage"] + '".')]:
            r = self.request(body,url,"Reuters","secondary")
            self.assertEqual(r["source_type"],"secondary")
            self.assertEqual(r["passage_evidence"]["matched_releases"][0]["url"],PAY["url"])
        self.assertNotEqual(self.request(PAY["passage"],PAY["url"],"Different publisher")["source_type"],"primary")
        self.assertNotEqual(self.request("False statement: " + PAY["passage"],PAY["url"])["source_type"],"primary")
        self.assertNotEqual(self.request(PAY["passage"],PAY["url"] + ".other")["source_type"],"primary")

    def test_new_release_and_uncatalogued_passage(self):
        r = self.request(NEXT_CPI["passage"],NEXT_CPI["url"])
        self.assertEqual([c["value"] for c in r["statistical_claims"]],[0.4,0.1])
        r = self.request(FRESH,FRESH_URL)
        self.assertEqual(r["source_type"],"unknown")
        self.assertEqual(r["passage_evidence"]["status"],"passage_match_pending")
        self.assertEqual([c["value"] for c in r["statistical_claims"]],[227000,4.2])
        self.assertNotEqual(r["recommended_action"],"use_as_anchor")

    def test_actual_news_and_two_release_dates(self):
        news = "Nonfarm payrolls increased by 142,000 jobs last month after a downwardly revised 89,000 rise in July"
        r = self.request(news,NEWS,"Reuters","secondary")
        self.assertEqual(r["source_type"],"secondary")
        self.assertNotIn("passage_evidence",r)
        self.assertEqual(r["statistical_claims"][0]["value"],142000)
        old = self.request(PAY["passage"],PAY["url"])["passage_evidence"]
        new = self.request(REVISION["passage"],REVISION["url"])["passage_evidence"]
        self.assertEqual(old["later_revisions"],new["matched_releases"])
        self.assertEqual(new["matched_releases"][0]["revision_of"],PAY["release_id"])

    def test_metadata_units_and_wrapping(self):
        for text in ["Source: BLS — " + PPI["passage"],"The food index rose 0.2 percentage points in August.",
                     "Nonfarm payroll employment increased by 1.5 million in August.",
                     "Nonfarm payroll employment increased by 12 million in August."]:
            self.assertFalse(evidence.statistics(text),text)
        metadata = self.request("Source: " + PAY["passage"], PAY["url"])
        self.assertFalse(metadata["passage_evidence"]["matched_releases"])
        wrapped = PAY["passage"].replace("employment increased","employment\nincreased")
        r = self.request(wrapped,PAY["url"])
        self.assertEqual(r["source_type"],"primary")
        self.assertEqual(len(r["statistical_claims"]),2)
        for c in r["statistical_claims"]:
            self.assertEqual(wrapped[c["start"]:c["end"]],c["span"])
        question = self.request("What if the " + CPI["passage"][4:] + "?",CPI["url"])
        self.assertTrue(all(c["assertion"] == "review" for c in question["statistical_claims"]))

    def test_receipt_registration_and_replay(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "release.html"
            raw.write_text("<article>" + FRESH + "</article>")
            args = raw,FRESH,"Independent Agency","2024-12-06",FRESH_URL,"Local reviewer",root/"records"
            path = register(*args)
            self.assertEqual(register(*args),path)
            record = json.loads(path.read_text())
            with patch.object(evidence,"RELEASES",evidence.RELEASES+(record,)):
                r = self.request(FRESH,FRESH_URL,"Independent Agency")
                self.assertEqual(r["passage_evidence"]["status"],"matched_release_excerpt")
            raw.write_text("<script>" + FRESH + "</script>")
            with self.assertRaises(ValueError):
                register(*args)
            self.assertEqual(json.loads(path.read_text()),record)

    def test_form_reports_and_compare(self):
        doc = dict(text=PAY["passage"],source_url=PAY["url"],source_name="BLS",source_type="primary")
        client=app.test_client()
        for route in ("/api/v1/analyze.txt","/api/v1/analyze.capsule"):
            r=client.post(route,json=doc)
            self.assertIn(b"Excerpt matched to the dated publisher release",r.data)
        self.assertIn(b"Passage origin",client.post("/",data=doc).data)
        compared=client.post("/api/v1/compare",json={"documents":[doc,{**doc,"source_name":"Reuters","source_type":"secondary","source_url":NEWS}]}).get_json()
        rows=compared["statistical_documents"]
        self.assertEqual([r["passage_evidence"]["document_type"] for r in rows],["primary","secondary"])
        self.assertEqual([len(r["statistical_claims"]) for r in rows],[2,2])

if __name__ == "__main__":
    unittest.main(verbosity=2)
