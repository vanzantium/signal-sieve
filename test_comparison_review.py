"""RC9 regression controls for evidence scope and comparison identity."""
import unittest
from pathlib import Path
from app import app
from signal_compare import extract_numeric_claims
from statistical_evidence import RELEASES

class ComparisonReview(unittest.TestCase):
    def compare(self,*texts):
        r=app.test_client().post("/api/v1/compare",json={"documents":[{"text":t} for t in texts]})
        self.assertEqual(r.status_code,200)
        return r.get_json()

    def test_listing_venue_and_index_can_coexist(self):
        rows=extract_numeric_claims("Trading on Nasdaq Global Select Market on November 19, 2026. The Nasdaq had shed 550 points.")
        self.assertEqual([r["numeric_value"] for r in rows if r["entity_key"]=="nasdaq"],[550])

    def test_window_positions_and_genuine_disagreement(self):
        base="Stocks fell. Since Oct. 1, the S&P 500 is down 7%."
        other="Stocks fell. The S&P 500 is down 3% since Oct. 1."
        r=self.compare(base,other)
        self.assertIn("sp500|since oct 1",[c["entity_key"] for c in r["conflicts"]])
        r=self.compare(base,"Stocks fell. The S&P 500 is down 2% today.")
        self.assertFalse(r["conflicts"])

    def test_debut_close_keeps_issuer_and_cent_precision(self):
        a="Acme finished its first session, closing at $97.17 per share. Shares trade on Nasdaq as ANT."
        b="Acme, the manufacturer, made its Nasdaq debut today. The stock with ticker ANT closed its first day at $97.18 a share."
        key="share-close|acme|ANT|ipo-day-one"
        self.assertIn(key,[c["entity_key"] for c in self.compare(a,b)["conflicts"]])
        for text in [b.replace("Acme","Atlas"),b.replace("ANT","XYZ"),b.replace("closed its first day","opened"),"If "+b]:
            self.assertNotIn(key,[c["entity_key"] for c in self.compare(a,text)["conflicts"]])

    def test_summary_label_and_short_receipt(self):
        raw=Path(__file__).with_name("tests").joinpath("run16_qwen_vendor_summary.txt").read_text()
        client=app.test_client()
        r=client.post("/api/v1/analyze",json=dict(text=raw,source_name="Qwen Team",source_type="primary",source_url="https://qwen.ai/blog?id=qwen4-max")).get_json()
        self.assertNotEqual(r["source_genre"],"official_release")
        self.assertEqual(r["recommended_action"],"treat_as_lead")
        record=RELEASES[0]
        doc=dict(text=record["passage"],source_url=record["url"],source_type="primary",source_name="BLS")
        r=client.post("/api/v1/analyze",json=doc).get_json()
        self.assertEqual(r["passage_evidence"]["quotation_use"],"attribute_exact_words_to_dated_release")
        doc["text"]=doc["text"].replace("142,000","143,000")
        r=client.post("/api/v1/analyze",json=doc).get_json()
        self.assertEqual(r["passage_evidence"]["quotation_use"],"review")

if __name__=="__main__":
    unittest.main(verbosity=2)
