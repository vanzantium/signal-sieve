"""Rebuild the local reference index from operator-acquired official HTML files."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime
import hashlib
import json
import sys

class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden_depth = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "template", "noscript"):
            self.hidden_depth += 1
    def handle_endtag(self, tag):
        if tag in ("script", "style", "template", "noscript"):
            self.hidden_depth = max(0, self.hidden_depth - 1)
    def handle_data(self, data):
        if not self.hidden_depth:
            self.parts.append(data)

RECORDS = [
    ("empsit_09062024", "Total nonfarm payroll employment increased by 142,000 in August, and the unemployment rate changed little at 4.2 percent"),
    ("cpi_09112024", "The food index increased 0.1 percent in August, after rising 0.2 percent in July."),
    ("ppi_09122024", "The Producer Price Index for final demand increased 0.2 percent in August, seasonally adjusted"),
    ("cpi_10102024", "The food index increased 0.4 percent in September, after rising 0.1 percent in August."),
    ("empsit_10042024", "the change for August was revised up by 17,000, from +142,000 to +159,000."),
]

def build(directory):
    entries = []
    for release, passage in RECORDS:
        data = (directory / (release + ".htm")).read_bytes()
        parser = VisibleText()
        parser.feed(data.decode("utf-8", errors="replace"))
        visible = " ".join(" ".join(parser.parts).split())
        if passage not in visible:
            raise ValueError("Passage absent from release " + release)
        row = dict(release_id=release, publisher="U.S. Bureau of Labor Statistics",
                   release_date=datetime.strptime(release.split("_")[1], "%m%d%Y").date().isoformat(),
                   url=f"https://www.bls.gov/news.release/archives/{release}.htm",
                   html_sha256=hashlib.sha256(data).hexdigest(), passage=passage)
        if release == "empsit_10042024":
            row["revision_of"] = "empsit_09062024"
            row["revision_scope"] = "August 2024 nonfarm payroll employment change"
        entries.append(row)
    return entries

if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1])), indent=2))
