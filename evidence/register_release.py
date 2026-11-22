"""Operator-only registration of a passage checked against independently acquired release HTML."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import date
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from make_catalog import VisibleText

def register(html, passage, publisher, release_date, url, reviewer, directory):
    location = urlsplit(url)
    if location.scheme != "https" or not location.hostname or location.username or location.password or location.fragment:
        raise ValueError("Supply the final HTTPS publisher URL without credentials or fragment")
    date.fromisoformat(release_date)
    if not reviewer.strip() or not publisher.strip():
        raise ValueError("Publisher and reviewer are required")
    data = html.read_bytes()
    parser = VisibleText()
    parser.feed(data.decode("utf-8"))
    passage = " ".join(passage.split())
    if len(passage.split()) < 8 or passage not in " ".join(" ".join(parser.parts).split()):
        raise ValueError("An exact passage of at least eight words must appear in the supplied HTML")
    identity = hashlib.sha256(json.dumps([url,release_date,passage], ensure_ascii=False).encode()).hexdigest()
    row = dict(release_id=identity, publisher=publisher.strip(), release_date=release_date,
               url=url, html_sha256=hashlib.sha256(data).hexdigest(), passage=passage,
               reviewer=reviewer.strip())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (identity + ".json")
    with NamedTemporaryFile(mode="w", dir=directory, delete=False) as writer:
        temp = Path(writer.name)
        try:
            json.dump(row, writer, indent=2)
            writer.flush()
            os.fsync(writer.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                if json.loads(path.read_text()) != row:
                    raise ValueError("Existing registration differs. Preserve it and review the source version.")
        finally:
            temp.unlink(missing_ok=True)
    return path

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("publisher", "reviewer", "url", "release-date", "passage"):
        cli.add_argument("--" + key, required=True)
    cli.add_argument("--html", type=Path, required=True)
    args = cli.parse_args()
    print(register(args.html, args.passage, args.publisher, args.release_date, args.url,
                   args.reviewer, Path(__file__).parent / "reviewed"))
