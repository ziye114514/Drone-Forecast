"""One fixed correlated-force derivative model; outputs before labels."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from matern_force_observer import MaternForceObserver,transition,coefficients,RATE
from rotor_force_observer import RotorForceObserver
from physical_publication import evaluate
from score_rotor164 import score


def checks():
    for h in [1e-6,.01,.2,.4]:
        A,Q=transition(round(h*1e6));ka,kj=coefficients(h)
        np.testing.assert_allclose([ka,kj],[A[0,4],A[0,6]],atol=1e-10,rtol=1e-6)
        assert np.linalg.eigvalsh(Q).min()>-1e-6
    m=MaternForceObserver()
    for i in range(10):m.step((i+1)/30,dict(detected=True,center=[0.,0.]),None,input_meta={})
    m.x[4:8]=[100.,-20.,300.,40.];m._set_input([0.,-1.],transfer_delta=np.array([0.,-1.]))
    before=m.x[4:6]+m.x[8]*m.u;jerk=m.x[6:8].copy();P=m.P.copy()
    m._set_input([0.,0.],transfer_delta=np.array([0.,1.]))
    np.testing.assert_allclose(m.x[4:6]+m.x[8]*m.u,before)
    np.testing.assert_allclose(m.x[6:8],jerk)
    assert np.linalg.eigvalsh(m.P).min()>-1e-6
    return dict(closed_form_matches_transition=True,visibility_partition_preserves_net_force_and_jerk=True,covariance_psd=True)


if __name__=='__main__':
    out=ROOT.parent/'output/native_plane165_20261104';result=dict(checks=checks(),runs={})
    for name,source in [('cold','raw_side_final'),('warm','warm_subpixel_source')]:
        m=MaternForceObserver();rows=[]
        for line in (ROOT.parent/f'output/side_temporal164_20261101/{source}/predictions.jsonl').read_text('utf-8').splitlines():
            r=json.loads(line);meta=r['period_selection'];u=r['force']['input_u']
            c,v,f,s=m.step(r['t'],r['body'],None if u is None else np.asarray(u),.7 if u is not None else 0.,
                allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta)
            r.update(coefficient=c,valid=v,force=f,state=s);r['physical']=evaluate(r,[.1,.2,.4]) if c else {};rows.append(r)
            if m.P is not None:assert np.linalg.eigvalsh(m.P).min()>-1e-5
        path=out/f'{name}_matern.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows),'utf-8');result['runs'][name]=path
    labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
    result['runs']={n:dict(fixed=score(p,labels,np.arange(110.4,112.0001,.2)),extension=score(p,labels,np.arange(112.2,114.0001,.2))) for n,p in result['runs'].items()}
    (out/'matern_scores.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(dict(checks=result['checks'],scores={n:{k:v['error'] for k,v in s.items()} for n,s in result['runs'].items()})))
