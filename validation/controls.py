"""Measure short disclosure and speed assertion errors on targeted controls."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from signal_sieve import _footer_sponsorship_disclosure
from app import app

times = ("Gemini 4 Neon High: 60, $3.74 per task, 14.8 minutes. "
         "Gemini 4 Carbon High: 60, $7.76 per task, 37.1 minutes.")
def checked(text):
    response = app.test_client().post("/api/v1/compare", json={"documents":[{"text":text},{"text":times}]})
    assert response.status_code == 200
    return any(row["pairings"] for row in response.get_json()["speed_ratio_checks"])

for kind, predict, cases in [
    ("footer", lambda t: _footer_sponsorship_disclosure(t)["present"], [
        (False, "The work was funded by Delta Fund."),
        (False, "A short note.\nThe work was funded by Delta Fund."),
        (True, "Reporting ends.\nFunded by U.S. Steel Corp."),
        (True, "Disclosure: Funded by U.S. Steel Corp.")]),
    ("speed assertion", checked, [
        (True, "Neon is three times faster than Carbon."),
        (False, "Neon is not three times faster than Carbon."),
        (False, "Neon could be three times faster than Carbon."),
        (False, "Is Neon three times faster than Carbon?")])]:
    counts = dict(TP=0, TN=0, FP=0, FN=0)
    for expected, text in cases:
        actual = bool(predict(text))
        counts[("T" if actual == expected else "F") + ("P" if actual else "N")] += 1
    print(kind, json.dumps(counts))
