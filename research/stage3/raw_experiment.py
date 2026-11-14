from pathlib import Path
import argparse,datetime,json,subprocess,sys
from experiments import ROOT,OUT,verify,save
sys.path.insert(0,str(ROOT))
from run_normals import source_hashes

def main():
    verify();parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','train6','side6','train2']);a=parser.parse_args()
    folder=OUT/'raw_v2';folder.mkdir(exist_ok=True);protocol=folder/'protocol.json'
    if a.action=='freeze':
        if protocol.exists():raise RuntimeError('RAW protocol exists')
        assert (OUT/'multi_completion.json').exists() and (OUT/'temporal/completion.json').exists()
        hashes=source_hashes()
        for rel in hashes:
            q=folder/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/rel).read_bytes())
        save(protocol,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=hashes,
            cases={'train6':dict(raw=r'E:\dataset\train\6\media\event.raw',start=0,duration=30),
                   'side6':dict(raw=r'E:\dataset\train\6\media\event.raw',start=108,duration=6),
                   'train2':dict(raw=r'E:\dataset\train\2\aligned\event.raw',start=0,duration=30)},
            parameters_not_tuned_between_cases=True,no_fabricated_calibration=True,
            interface_repair='First RAW attempt in raw/ stopped on missing reporting attribute differential_gain. Adapter now reports0; no numerical measurement rule changed.',
            caveat='Earlier used local development videos. Single-arc audit still failed some short arcs; retained as separate experimental arm, not default.'))
        print('RAW ablation protocol frozen',flush=True);return
    p=json.loads(protocol.read_text());assert p['source_hashes']==source_hashes();case=p['cases'][a.action]
    subprocess.run([sys.executable,str(ROOT/'run_normals.py'),'--raw',case['raw'],'--start',str(case['start']),'--duration',str(case['duration']),'--output',str(folder/a.action)],check=True)

if __name__=='__main__':main()
