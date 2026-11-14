"""Same frozen fixed-reference inputs; test geometry values, not only gates."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from layout_reference_observer import LayoutReferenceObserver,NativeControlLayoutObserver
from physical_publication import evaluate
from score_rotor164 import score
OUT=ROOT.parent/'output/layout176_20261107/source_control'

if __name__=='__main__':
    OUT.mkdir(exist_ok=False);results={}
    for case in ('side','train2'):
        data=[json.loads(s) for s in (OUT.parent/f'integrated/{case}_fixed.jsonl').read_text('utf-8').splitlines()];results[case]={}
        for arm,cls in [('layout',LayoutReferenceObserver),('native',NativeControlLayoutObserver)]:
            m=cls();output=[];maximum=0.
            for old in data:
                r=copy.deepcopy(old);meta=r['period_selection'];u=r['force']['input_u']
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
                c,v,f,s=m.step(r['t'],r['body'],None if u is None else np.asarray(u),.7 if u is not None else 0.,
                    allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'])
                if arm=='layout' and c:
                    for k in ('position','velocity','acceleration'):maximum=max(maximum,float(np.max(abs(np.array(c[k])-r['coefficient'][k]))))
                if m.P is not None:assert np.linalg.eigvalsh(m.P).min()>-1e-5
                r.update(coefficient=c,valid=v,force=f,state=s);r['physical']=evaluate(r,[.1,.2,.4]) if c else {};output.append(r)
            if arm=='layout':assert maximum<1e-8,maximum
            (OUT/f'{case}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output),'utf-8')
            results[case][arm]=dict(replay_parity=maximum,layout_updates=sum(r['state']['layout_observation_used'] for r in output))
    for case,arms in results.items():
        labels=ROOT.parent/('output/dense_labels_side_110_1195/labels.json' if case=='side' else 'output/dense_labels_train2_40_46/labels.json');q=np.arange(110.4,114.0001,.2) if case=='side' else np.arange(43.2,44.4001,.2)
        for arm,r in arms.items():r['score']=score(OUT/f'{case}_{arm}.jsonl',labels,q)
    (OUT/'scores.json').write_text(json.dumps(results,indent=2),'utf-8');print(json.dumps({c:{a:dict(mean=r['score']['error']['mean'],updates=r['layout_updates'],parity=r['replay_parity']) for a,r in arms.items()} for c,arms in results.items()}))
