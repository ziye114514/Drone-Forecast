"""Predeclared exposure timing hypothesis on causal saved observations."""
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from rotor_force_observer import RotorForceObserver
from physical_publication import evaluate
from score_rotor164 import score


if __name__=='__main__':
    folder=ROOT.parent/'output/native_plane165_20261104';paths={}
    for name,source in [('cold','raw_side_final'),('warm','warm_subpixel_source')]:
        data=[json.loads(s) for s in (ROOT.parent/f'output/side_temporal164_20261101/{source}/predictions.jsonl').read_text('utf-8').splitlines()]
        model=RotorForceObserver();result=[]
        for r in data:
            u=r['force']['input_u'];meta=r['period_selection']
            lag=.01 if r['body'].get('measurement_source')=='current_rotor_geometry' else .033333/2
            c,v,f,s=model.step(r['t'],r['body'],None if u is None else np.asarray(u),.7 if u is not None else 0.,
                allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,measurement_lag_s=lag)
            r.update(coefficient=c,valid=v,force=f,state=s);r['physical']=evaluate(r,[.1,.2,.4]) if c else {}
            result.append(r)
        path=folder/f'{name}_exposure.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in result),'utf-8');paths[name]=path
    labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
    result={n:dict(fixed=score(p,labels,np.arange(110.4,112.0001,.2)),
        extension=score(p,labels,np.arange(112.2,114.0001,.2))) for n,p in paths.items()}
    (folder/'exposure_scores.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({n:{k:v['error'] for k,v in s.items()} for n,s in result.items()}))
