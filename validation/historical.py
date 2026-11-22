"""Replay the included November 7–10 packets through either code revision."""
import json
import re
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from app import app

client = app.test_client()
results = []
for packet in sorted((root / "tests/packets").iterdir()):
    documents = []
    types = {}
    for path in sorted(packet.glob("*-feed-*.txt")):
        name = path.stem.split("-feed-")[-1]
        doc = {"text": path.read_text(), "source_name": name}
        url = re.search(r"^URL: (https?://\S+)", doc["text"], re.M)
        if url:
            doc["source_url"] = url.group(1)
        documents.append(doc)
        response = client.post("/api/v1/analyze", json={**doc, "source_type": "primary" if name in ("primary", "linux") else "secondary"})
        assert response.status_code == 200
        types[name] = response.get_json().get("source_type")
    response = client.post("/api/v1/compare", json={"documents": documents})
    assert response.status_code == 200
    comparison = response.get_json()
    results.append({"packet": packet.name, "resolved_types": types,
                    "counts": {key: len(comparison[key]) for key in ("conflicts", "alignment", "unique_claims")}})
print(json.dumps(results, indent=2))
