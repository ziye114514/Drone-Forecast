"""Read-only local archive inventory and pinned upstream source audit."""
from pathlib import Path
import hashlib,json,urllib.request,zipfile,shutil,sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent/'output/stage7_localization_budget_20261017'

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'event-uav-research-audit'}),timeout=40).read()

def upstream():
    folder=OUT/'upstream';folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for repo in ['miccunifi/FRED','takara-ai/DroneStalker']:
        try:
            meta=json.loads(get('https://api.github.com/repos/'+repo))
            commit=json.loads(get('https://api.github.com/repos/'+repo+'/commits/'+meta['default_branch']))['sha']
            tree=json.loads(get('https://api.github.com/repos/'+repo+'/git/trees/'+commit+'?recursive=1'))
            dest=folder/repo.split('/')[-1];dest.mkdir(exist_ok=True)
            save(dest/'tree.json',tree)
            wanted=[x['path'] for x in tree['tree'] if x['type']=='blob' and (x['path']=='README.md' or x['path'].lower().startswith('license') or (repo.endswith('/FRED') and (x['path'].startswith('dataset_splits/') or x['path']=='src/data/data.py')))]
            files=[]
            for name in wanted:
                b=get(f'https://raw.githubusercontent.com/{repo}/{commit}/{name}')
                p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
                files.append(dict(path=name,sha256=sha(p),bytes=len(b)))
            records.append(dict(repo=repo,commit=commit,license=meta.get('license'),files=files,
                selected_code=[x['path'] for x in tree['tree'] if x['path'].endswith('.py')][:100]))
        except Exception as e:records.append(dict(repo=repo,error=str(e)))
    save(folder/'audit.json',records);print(json.dumps(records,ensure_ascii=False),flush=True)

def local():
    records=[]
    for i in range(11):
        zpath=Path('E:/FRED')/f'{i}.zip';folder=OUT/'data'/f'sequence{i}';folder.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            names=z.namelist();raw=[n for n in names if n.endswith('/events.raw')]
            assert len(raw)==1,raw
            files=[]
            for name in ['coordinates.txt','interpolated_coordinates.txt','tracks.txt']:
                matches=[n for n in names if Path(n).name==name]
                if not matches:continue
                assert len(matches)==1,matches
                p=folder/name;p.write_bytes(z.read(matches[0]));files.append(dict(name=name,member=matches[0],sha256=sha(p),bytes=p.stat().st_size))
            cached={1:ROOT.parent/'output/stage1_observation_errors_20261016/public_truth_substitutes/fred_sequence1/events.raw',
                    2:ROOT.parent/'output/stage2_module_validation_20261016/activity_fallback_fixed/sequence2/input/events.raw',
                    3:ROOT.parent/'output/stage2_module_validation_20261016/activity_fallback_fixed/sequence3/input/events.raw'}.get(i)
            target=cached if cached and cached.exists() else folder/'events.raw'
            info=z.getinfo(raw[0])
            if not target.exists():
                with z.open(raw[0]) as source,target.open('wb') as f:shutil.copyfileobj(source,f)
            assert target.stat().st_size==info.file_size
            # Geometry of raw/pseudo-frame alignment is audited later, independently of forecast errors.
            records.append(dict(sequence=i,archive=str(zpath),archive_bytes=zpath.stat().st_size,
                raw=str(target.resolve()),raw_bytes=target.stat().st_size,raw_sha256=sha(target),raw_crc32=info.CRC,
                raw_member=raw[0],annotations=files,author_frames=[n for n in names if n.startswith('Event/Frames/') and n.endswith('.png')][:2]))
        print('prepared sequence',i,flush=True)
    save(OUT/'local_inventory.json',records)

if __name__=='__main__':globals()[sys.argv[1]]()
