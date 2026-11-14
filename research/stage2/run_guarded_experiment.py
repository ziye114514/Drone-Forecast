from pathlib import Path
import argparse,datetime,json,subprocess,sys
from study import ROOT,OUT,save,verify
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from run_guarded import source_hashes

def main():
    verify();dest=OUT/'guarded';dest.mkdir(exist_ok=True);p=dest/'protocol.json'
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','fred3','own_train6']);a=parser.parse_args()
    if a.action=='freeze':
        if p.exists():raise RuntimeError('Frozen guard experiment exists')
        hashes=source_hashes()
        for rel in hashes:
            q=dest/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/rel).read_bytes())
        save(p,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=hashes,
            reason='Unrestricted activity produced28/29 absent-label false publications and poor own-clip endpoints; retain failure and constrain identity.',
            cases={'fred3':dict(raw=str(OUT/'activity_fallback_fixed/sequence3/input/events.raw'),duration_s=20),
                   'own_train6':dict(raw=r'E:\dataset\train\6\media\event.raw',duration_s=30)},
            evidence_scope='Post-diagnostic development fix, not unseen test superiority. Known failure: small target without rotor confirmation remains unavailable.'))
        print('Guarded experiment frozen',flush=True);return
    protocol=json.loads(p.read_text());assert protocol['source_hashes']==source_hashes();case=protocol['cases'][a.action]
    subprocess.run([sys.executable,str(ROOT/'core/run_guarded.py'),'--raw',case['raw'],'--duration',str(case['duration_s']),'--output',str(dest/a.action)],check=True)

if __name__=='__main__':main()
