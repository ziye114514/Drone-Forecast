"""Source inventory and prior metadata use; no event/frame decoding."""
from pathlib import Path
import subprocess,json,re,hashlib,zipfile,collections
BASE=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    out=BASE/'output/source_inventory521_20261113';out.mkdir(exist_ok=False)
    cmd=['rg','-l','--glob','*protocol*.json','--glob','*summary*.json','--glob','*manifest*.json',
        '--glob','!**/source_snapshot/**','--glob','!**/source_snapshot_final/**',
        r'dataset[/\\]+train|FRED|fred','output']
    result=subprocess.run(cmd,cwd=BASE,text=True,capture_output=True,encoding='utf8')
    uses=collections.defaultdict(list)
    def visit(value,path):
        if isinstance(value,dict):
            for k,v in value.items():visit(v,path)
        elif isinstance(value,list):
            for v in value:visit(v,path)
        elif isinstance(value,str):
            norm=value.replace('\\','/').lower()
            if (('dataset/train/' in norm or '/fred/' in norm) and len(norm)<500 and ('raw' in norm or '.zip' in norm or '.npy' in norm)):
                uses[norm].append(path)
    for name in result.stdout.splitlines():
        try:visit(json.loads((BASE/name).read_text('utf8')),name)
        except (ValueError,OSError):pass
    train=[]
    for p in sorted(Path('E:/dataset/train').rglob('*.raw')):
        with p.open('rb') as f:prefix=f.read(8192)
        header=[]
        for line in prefix.splitlines(keepends=True):
            if not line.startswith(b'%'):break
            header.append(line)
        h=b''.join(header)
        norm=str(p).replace('\\','/').lower();prior=uses.get(norm,[])
        # Also match old /media path variants, keeping uncertainty explicit.
        folder=p.relative_to(Path('E:/dataset/train')).parts[0]
        group={k:v[:3] for k,v in uses.items() if f'dataset/train/{folder}/' in k}
        train.append(dict(path=str(p),size=p.stat().st_size,header=h.decode('ascii',errors='replace'),
            header_sha256=hashlib.sha256(h).hexdigest(),exact_prior_metadata_count=len(prior),
            first_prior_metadata=prior[:5],same_train_folder_prior_paths=group))
    archives=[]
    for p in sorted(Path('E:/FRED').glob('*.zip')):
        try:
            with zipfile.ZipFile(p) as z:
                entries=z.infolist();archives.append(dict(path=str(p),entries=len(entries),
                    examples=[dict(name=q.filename,size=q.file_size) for q in entries[:12]]))
        except (OSError,zipfile.BadZipFile) as e:archives.append(dict(path=str(p),error=str(e)))
    (out/'results.json').write_text(json.dumps(dict(train=train,archives=archives,
        prior_metadata_paths={k:v[:8] for k,v in uses.items()},metadata_only_no_event_decode=True,
        no_found_use_is_not_exhaustive_proof_of_unseen=True),ensure_ascii=False,indent=2),'utf8')
    for r in train:print(r['path'],r['size'],'exact_prior',r['exact_prior_metadata_count'],'folder_prior',len(r['same_train_folder_prior_paths']))
    for r in archives:print(r['path'],r.get('entries'),[q['name'] for q in r.get('examples',[])[:3]])
