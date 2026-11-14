"""One predeclared model correction, saved inputs only, no parameter sweep."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_force_observer import RotorForceObserver
from physical_publication import evaluate
from score_rotor164 import score


def replay(source,out,consistent):
    model=RotorForceObserver(consistent_dynamics=consistent);results=[]
    for line in Path(source).read_text('utf-8').splitlines():
        row=json.loads(line);u=row['force']['input_u'];meta=row['period_selection']
        # Input qualification is fixed by the saved causal front. Never use
        # observed future acceleration to calibrate this replay.
        coefficient,valid,force,state=model.step(row['t'],row['body'],None if u is None else np.asarray(u),
            .7 if u is not None else 0.,allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta)
        row.update(coefficient=coefficient,valid=valid,force=force,state=state)
        row['physical']=evaluate(row,[.1,.2,.4]) if coefficient else {}
        results.append(row)
    Path(out).write_text(''.join(json.dumps(r)+'\n' for r in results),'utf-8')
    return results


def checks():
    m=RotorForceObserver(consistent_dynamics=True);t=0.
    for _ in range(10):
        t+=1/30;m.step(t,dict(detected=True,center=[0.,0.]),np.array([0.,-1.]),.7,
            allow_gain_learning=False,input_meta=dict(observed_components={'a':[0.,-1.]}))
    m.x[2:4]=[30.,10.];m.x[4:6]=[50.,-100.]
    c=dict(position=m.x[:2].tolist(),velocity=m.x[2:4].tolist())
    f=m._force(True,m.x[4:6]+m.x[6]*m.u)
    predicted=evaluate(dict(coefficient=c,force=f),[.2])['0.2']
    m.step(t+.2,dict(detected=False),None,0.,input_meta={})
    np.testing.assert_allclose(m.x[:2],predicted,atol=1e-8)
    # Transferring the current input to unknown must preserve the whole
    # conditional future, not just instantaneous net acceleration.
    c=dict(position=m.x[:2].tolist(),velocity=m.x[2:4].tolist())
    before=evaluate(dict(coefficient=c,force=m._force(False,m.x[4:6]+m.x[6]*m.u)),[.2])['0.2']
    m._set_input([0.,-2.],transfer_delta=np.array([0.,-2.]))
    after=evaluate(dict(coefficient=c,force=m._force(True,m.x[4:6]+m.x[6]*m.u)),[.2])['0.2']
    np.testing.assert_allclose(before,after,atol=1e-8)
    assert np.linalg.eigvalsh(m.P).min()>-1e-6
    return dict(update_equals_published_rollout=True,partition_preserves_future=True,covariance_psd=True)


if __name__=='__main__':
    folder=ROOT.parent/'output/side_temporal164_20261101';checks_result=checks();runs={}
    # Save all model outputs before opening development reference boxes.
    for name in ['side_fast_final','warm_center_fixed']:
        for arm in [False,True]:
            path=folder/f'{name}_consistent{int(arm)}.jsonl'
            replay(folder/name/'predictions.jsonl',path,arm);runs[f'{name}_{arm}']=path
    labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
    result=dict(checks=checks_result,parameters_searched=False,scores={})
    for name,path in runs.items():
        result['scores'][name]=dict(fixed=score(path,labels,np.arange(110.4,112.0001,.2)),
            extension=score(path,labels,np.arange(112.2,114.0001,.2)))
    (folder/'consistent_scores.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(dict(checks=checks_result,scores={n:{k:v['error'] for k,v in s.items()} for n,s in result['scores'].items()})))
