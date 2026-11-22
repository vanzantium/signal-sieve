import json
import unittest
from unittest.mock import patch
from app import app, MAX_TEXT_CHARS
from signal_sieve import _byline_name, _extract_claimed_org
import statistical_evidence as evidence

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

if __name__ == "__main__":
    unittest.main(verbosity=2)
