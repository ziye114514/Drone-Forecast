"""Fetch only the selected pinned public reference sources; local study only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'external/pvt_reference_d71cad0'
REV='d71cad0ed0bfede979469088704f6ba1d5a6f38c'
tree=json.loads((OUT/'tree.json').read_text('utf-8'));blobs={r['path']:r for r in tree['tree'] if r['type']=='blob'}
names=['MODEL_ZOO.md','README.md','pysot/models/predictor/lb_5.py','pysot/models/predictor/base_predictor.py',
    'pysot/models/predictor/__init__.py','experiments/siamrpn_mobilev2_l234_dwxcorr/pre_lb_config.yaml']
def fetch(name):
    request=urllib.request.Request('https://raw.githubusercontent.com/Jaraxxus-Me/PVT_pp/'+REV+'/'+name,headers={'User-Agent':'event-drone-forecast-research'})
    with urllib.request.urlopen(request,timeout=30) as response:data=response.read(2000000)
    assert len(data)==blobs[name]['size']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blobs[name]['sha']
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
    return dict(path=name,git_blob_sha1=blobs[name]['sha'],sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,names))
meta=dict(repository='https://github.com/Jaraxxus-Me/PVT_pp',revision=REV,files=records,
    license_file_found=any('license' in n.lower() for n in blobs),redistribution_assumed=False,local_reference_only=True)
(OUT/'UPSTREAM_SNAPSHOT.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))
