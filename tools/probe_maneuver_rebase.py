"""Reproducible fixed-observation causal comparison, not a RAW FPS test."""
from pathlib import Path
import argparse,hashlib,json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from probe_same_window_motion import replay
from compare_final_pipeline import score
from diagnose_maneuver_horizon import SPECS
OPTIONS=dict(curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,weak_support=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);frozen={};checks={}
    jobs=[(n,ROOT.parent/'output'/s,lab) for n,s,lab in SPECS]
    jobs.append(('fred7',ROOT.parent/'output/fred_domain_adaptation_20261029/RAW_7_baseline/predictions.jsonl',None))
    for name,path,lab in jobs:
        data=path.read_bytes();original=[json.loads(x) for x in data.splitlines()]
        control,_=replay(original,False,**OPTIONS)
        candidate,cost=replay(original,False,position_history_rebase='stable-axis',**OPTIONS)
        difference=max(float(np.max(abs(np.array(a['physical'][h])-b['physical'][h]))) for a,b in zip(original,control) for h in a['physical'])
        assert difference<1e-7,(name,'control mismatch',difference)
        for tag,rows in [('control',control),('candidate',candidate)]:
            (out/f'{name}_{tag}.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in rows)+'\n','utf-8')
        checks[name]=dict(source=str(path),source_sha256=hashlib.sha256(data).hexdigest(),control_max_difference=difference,
            current_body_and_validity_exact=all(a['body']==b['body'] and a['valid']==b['valid'] for a,b in zip(control,candidate)),
            rebased=sum(bool((r['image_motion'].get('position_history_gauge') or {}).get('applied')) for r in candidate),
            maximum_prediction_change=max(float(np.max(abs(np.array(a['physical'][h])-b['physical'][h]))) for a,b in zip(control,candidate) for h in a['physical']))
        frozen[name]=(control,candidate,lab)
    # All candidate decisions/predictions are now frozen before manual labels.
    for name,(control,candidate,lab) in frozen.items():
        if lab is None:continue
        labels=sorted(json.loads((ROOT.parent/'output'/lab/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        checks[name]['score']={k:score(rows,labels)['physical_future_error_px'] for k,rows in [('control',control),('candidate',candidate)]}
    (out/'summary.json').write_text(json.dumps(checks,indent=2,allow_nan=False),'utf-8');print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
