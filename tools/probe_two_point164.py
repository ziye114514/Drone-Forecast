"""Causal two-observation initialization; no parameter or label search."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_force_observer import RotorForceObserver
from physical_publication import evaluate
from score_rotor164 import score


if __name__=='__main__':
    folder=ROOT.parent/'output/side_temporal164_20261101';saved={}
    for name in ['raw_side_final','warm_subpixel_source']:
        model=RotorForceObserver(two_point_init=True);result=[]
        for line in (folder/name/'predictions.jsonl').read_text('utf-8').splitlines():
            r=json.loads(line);u=r['force']['input_u'];meta=r['period_selection']
            c,v,f,s=model.step(r['t'],r['body'],None if u is None else np.asarray(u),.7 if u is not None else 0.,
                allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta)
            r.update(coefficient=c,valid=v,force=f,state=s);r['physical']=evaluate(r,[.1,.2,.4]) if c else {}
            if model.P is not None:assert np.linalg.eigvalsh(model.P).min()>-1e-6
            result.append(r)
        path=folder/f'{name}_two_point.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in result),'utf-8');saved[name]=path
    m=RotorForceObserver(two_point_init=True)
    m.step(.1,dict(detected=True,center=[0.,0.]),None,input_meta={})
    c,_,_,_=m.step(.133333,dict(detected=True,center=[33.333,0.]),None,input_meta={})
    np.testing.assert_allclose(c['velocity'],[1000.,0.],atol=1e-7)
    labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
    result={name:dict(fixed=score(path,labels,np.arange(110.4,112.0001,.2)),
        extension=score(path,labels,np.arange(112.2,114.0001,.2))) for name,path in saved.items()}
    (folder/'two_point_scores.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({n:{k:v['error'] for k,v in s.items()} for n,s in result.items()}))
