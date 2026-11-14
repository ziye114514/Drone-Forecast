"""Read the official pinned PVT++ source inventory, without model execution."""
from pathlib import Path
import json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
REV='d71cad0ed0bfede979469088704f6ba1d5a6f38c'
out=ROOT/'external/pvt_reference_d71cad0';out.mkdir(exist_ok=True)
request=urllib.request.Request('https://api.github.com/repos/Jaraxxus-Me/PVT_pp/git/trees/'+REV+'?recursive=1',headers={'User-Agent':'event-drone-forecast-research'})
with urllib.request.urlopen(request,timeout=30) as response:data=json.load(response)
assert not data.get('truncated')
(out/'tree.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
paths=[r['path'] for r in data['tree'] if r['type']=='blob' and
    ('predict' in r['path'].lower() or 'model_zoo' in r['path'].lower() or 'license' in r['path'].lower() or r['path'].endswith('.yaml'))]
print(json.dumps(dict(revision=REV,files=paths),indent=2))
