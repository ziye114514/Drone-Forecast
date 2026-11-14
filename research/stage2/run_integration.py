"""Freeze the integration and run two predeclared examples, no label inputs."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
from study import ROOT,OUT,save,digest,verify
sys.path.insert(0,str(ROOT))
from run_hybrid import source_hashes

def main():
    verify();dest=OUT/'integrated';p=dest/'protocol.json';dest.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','fred3','own_train6']);a=parser.parse_args()
    if a.action=='freeze':
        if p.exists():raise RuntimeError('Already frozen')
        hashes=source_hashes()
        for rel,h in hashes.items():
            target=dest/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/rel).read_bytes())
        save(p,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=hashes,
            cases={'fred3':dict(raw=str(OUT/'activity_fallback_fixed/sequence3/input/events.raw'),duration_s=20),
                   'own_train6':dict(raw=r'E:\dataset\train\6\media\event.raw',duration_s=30)},
            note='Integrated physics is newly combined with the already evaluated frontend. Do not transfer CV-only scores to physical outputs without explicit scoring.'))
        print('Integration frozen',flush=True);return
    protocol=json.loads(p.read_text());assert protocol['source_hashes']==source_hashes()
    case=protocol['cases'][a.action]
    subprocess.run([sys.executable,str(ROOT/'run_hybrid.py'),'--raw',case['raw'],'--output',str(dest/a.action),'--duration',str(case['duration_s'])],check=True)

if __name__=='__main__':main()
