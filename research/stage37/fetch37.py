"""Fetch pinned public detector assets only, never run the upstream application."""
from pathlib import Path
import datetime,hashlib,json,urllib.request

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage37_upstream_detector_20261018'
REPO='takara-ai/DroneStalker';COMMIT='93e846eef8abc974da1e0c5cf71331729648f143'


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'event-drone-research-reproducibility'})
    with urllib.request.urlopen(request,timeout=60) as response:return response.read()


def main():
    folder=OUT/'upstream';folder.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
    selected=[r for r in tree['tree'] if r['type']=='blob' and (r['path'] in ['README.md','yolo/best.pt'] or (r['path'].startswith('yolo/') and r['path'].endswith(('.py','.yaml','.yml','.txt'))))]
    assert any(r['path']=='yolo/best.pt' for r in selected),'Pinned detector not available'
    manifest=[]
    for item in selected:
        assert item.get('size',0)<=60_000_000,item['path']
        relative=Path(item['path']);assert not relative.is_absolute() and '..' not in relative.parts
        path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():data=path.read_bytes()
        else:
            data=fetch(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{item["path"]}')
            assert len(data)==item['size'];assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['sha']
            path.write_bytes(data)
        assert len(data)==item['size'] and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['sha']
        manifest.append(dict(path=item['path'],bytes=len(data),git_blob_sha1=item['sha'],sha256=hashlib.sha256(data).hexdigest()))
        print('Pinned asset: '+item['path']+' '+str(len(data))+' bytes',flush=True)
    result=dict(repo=REPO,commit=COMMIT,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=manifest,
        scope='Public detection source/weights only, no application execution/services/API keys. Weight archive still needs nonexecuting serialization/class audit before load. No G or access to yolov10-main.')
    (OUT/'upstream_assets.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
