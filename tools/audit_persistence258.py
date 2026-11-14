"""Why the single force-memory candidate failed: teacher vs reference."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from physical_publication import evaluate
BASE=ROOT.parent/'output'


def pos_at(rows,t):
    good=[r for r in rows if r['body'].get('detected') and not r['body'].get('center_provisional') and r['motion'].get('current_observation')]
    left=[r for r in good if r['t']<=t+1e-8];right=[r for r in good if r['t']>=t-1e-8]
    if not left or not right:return None
    a=left[-1];b=right[0]
    if b['t']-a['t']>.08:return None
    q=0 if abs(b['t']-a['t'])<1e-8 else (t-a['t'])/(b['t']-a['t'])
    return (1-q)*np.asarray(a['body']['center'])+q*np.asarray(b['body']['center'])


if __name__=='__main__':
    report={}
    for n in ['13','15','16']:
        base=[json.loads(s) for s in source(n).read_text('utf-8').splitlines()]
        new=[json.loads(s) for s in (BASE/f'persistence257_20261108/{n}.jsonl').read_text('utf-8').splitlines()]
        scores=json.loads((BASE/'persistence257_20261108/results.json').read_text('utf-8'))['cases'][n];records=[]
        oldscores={x['t']:x for x in scores['control']['records'] if x['valid']};newscores={x['t']:x for x in scores['persistence']['records'] if x['valid']}
        for t,oldscore in oldscores.items():
            i=[j for j,r in enumerate(base) if r['t']<=t+1e-9][-1];a=base[i];b=new[i];target=pos_at(base,t+.4)
            if target is None:continue
            h=t-a['t']+.4;p=np.asarray(evaluate(a,[h])[str(h)]);q=np.asarray(evaluate(b,[h])[str(h)])
            centre=a['body'].get('center')
            records.append(dict(t=t,baseline_own_px=float(np.linalg.norm(p-target)),candidate_own_px=float(np.linalg.norm(q-target)),
                baseline_reference_px=oldscore['error_px'],candidate_reference_px=newscores[t]['error_px'],
                origin_state_vs_native_px=None if centre is None else float(np.linalg.norm(np.asarray(a['coefficient']['position'])-centre)),
                native_vs_reference_target_px=float(np.linalg.norm(target-oldscore['target'])),
                weights=b['state']['future_force_persistence']['weights']))
        report[n]=dict(common=len(records),baseline_own_mean=float(np.mean([r['baseline_own_px'] for r in records])),
            candidate_own_mean=float(np.mean([r['candidate_own_px'] for r in records])),
            origin_state_vs_native_mean=float(np.mean([r['origin_state_vs_native_px'] for r in records if r['origin_state_vs_native_px'] is not None])),
            native_vs_reference_target_mean=float(np.mean([r['native_vs_reference_target_px'] for r in records])),
            worst=sorted(records,key=lambda r:r['candidate_reference_px']-r['baseline_reference_px'],reverse=True)[:4],records=records)
        print(n,{k:v for k,v in report[n].items() if k not in ('records','worst')},flush=True);print(report[n]['worst'][:2],flush=True)
    (BASE/'persistence257_20261108/credit_audit.json').write_text(json.dumps(report,indent=2),'utf-8')
