"""Inspect public model availability and official trajectory encoding."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,time,urllib.request
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'external/pvt_reference_d71cad0';REV='d71cad0ed0bfede979469088704f6ba1d5a6f38c'
def metadata():
    request=urllib.request.Request('https://g.api.mega.co.nz/cs?id='+str(time.time_ns()%1000000000),
        data=json.dumps([dict(a='g',g=1,p='hFVklIpZ')]).encode(),headers={'Content-Type':'application/json','User-Agent':'event-drone-forecast-research'})
    with urllib.request.urlopen(request,timeout=30) as response:data=json.load(response)
    (OUT/'public_model_metadata_local.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    return dict(public_model_response_type=type(data).__name__,download_available=isinstance(data,list) and isinstance(data[0],dict) and 'g' in data[0],
        bytes=data[0].get('s') if isinstance(data,list) and isinstance(data[0],dict) else None,
        failure=data if isinstance(data,list) and isinstance(data[0],int) else None)
def source():
    name='pysot/tracker/siamrpn_tracker_f.py'
    request=urllib.request.Request('https://raw.githubusercontent.com/Jaraxxus-Me/PVT_pp/'+REV+'/'+name,headers={'User-Agent':'event-drone-forecast-research'})
    with urllib.request.urlopen(request,timeout=30) as response:data=response.read()
    record=next(r for r in json.loads((OUT/'tree.json').read_text('utf-8'))['tree'] if r['path']==name)
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==record['sha']
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return dict(source=name,sha256=hashlib.sha256(data).hexdigest())
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(f) for f in [metadata,source]]
    for f in futures:
        try:print(json.dumps(f.result()),flush=True)
        except Exception as e:print(type(e).__name__+': '+str(e),flush=True)
