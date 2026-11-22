"""Same frozen, out-of-development-source inputs against either code directory."""
import sys, json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from signal_sieve import analyze
from signal_compare import extract_numeric_claims, extract_generic_quantitative_claims
out=[]
for row in json.loads(Path(__file__).with_name('public_holdout.json').read_text()):
    text=row['text']
    claims=extract_numeric_claims(text)
    claims+=extract_generic_quantitative_claims(text,'BLS',claims)
    a=analyze(text,source_type='primary',source_name='U.S. Bureau of Labor Statistics',source_url=row['source'])
    claims.extend(dict(numeric_value=c["value"], entity_key=c["measure"], unit=c["unit"],
                       context=dict(kind=c["kind"], period=c["period"]), span=c["span"])
                  for c in a.get("statistical_claims", []))
    out.append(dict(source=row['source'],targets=row['targets'],claims=[{k:c.get(k) for k in ('numeric_value','entity_key','unit','context','span')} for c in claims],resolved_type=a['source_type'],action=a['recommended_action']))
print(json.dumps(out,indent=2))
