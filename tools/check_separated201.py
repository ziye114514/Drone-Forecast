"""Mechanics that matter for force credit and identical publication dynamics."""
from pathlib import Path
import sys,json,numpy as np
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from separated_force_response import SeparatedForceObserver,blocks,RATE_TAU
from force_process_controls201 import MatchedIsotropicControl
from physical_publication import evaluate
from run_rotor_only import RotorPredictor

if __name__=='__main__':
    out=ROOT.parent/'output/separated201_20261107';out.mkdir(exist_ok=True);checks={}
    errors=[]
    for tau in (.1,.35):
        for h in (.033333,.1,.4):
            A,_,_,_,_=blocks(round(h*1e6),1/tau)
            c=dict(position=[100.,200.],velocity=[30.,-50.],acceleration=[60.,70.])
            f=dict(acceleration_correlation_s=tau,acceleration_px_s2=[60.,70.],active=False,
                external_unknown_rate_px_s3=[700.,-900.],external_rate_correlation_s=.35)
            got=np.array(evaluate(dict(coefficient=c,force=f),[h])[str(float(h))]);expect=np.array([(A@np.array([c['position'][i],c['velocity'][i],c['acceleration'][i],f['external_unknown_rate_px_s3'][i]]))[0] for i in range(2)])
            errors.append(float(np.linalg.norm(got-expect)))
    assert max(errors)<1e-8,errors;checks['publication_matches_exact_state_transition_max_px']=max(errors)
    a=SeparatedForceObserver();b=MatchedIsotropicControl()
    for m in (a,b):m._initialize(np.array([100.,200.]))
    Fa,Qa=a._transition(.033333);Fb,Qb=b._transition(.033333);assert np.allclose(Qa,Qb)
    a.normal=b.normal=np.array([0.,-1.]);Fa,Qa=a._transition(.033333);Fb,Qb=b._transition(.033333)
    assert np.allclose(Fa,Fb);assert abs(np.trace(Qa[:6,:6])-np.trace(Qb[:6,:6]))<1e-6
    assert np.linalg.eigvalsh(Qa).min()>-1e-5
    checks.update(isotropic_control_same_qualification_and_net_noise_trace=True,process_covariance_PSD=True)
    before=a.x[4:6]+a.x[6]*a.u;e0=a.x[4:6]-a.x[11]*a.normal
    added=.1*a.normal;a._set_input(added,transfer_delta=added,transfer_cov=np.eye(2)*.001)
    assert np.allclose(before,a.x[4:6]+a.x[6]*a.u) and np.allclose(e0,a.x[4:6]-a.x[11]*a.normal)
    checks['visibility_transfer_preserves_net_and_external']=True
    model=RotorPredictor(relative_observation=True,unknown_force_model='plane-separated')
    assert isinstance(model.force,SeparatedForceObserver)
    try:RotorPredictor(unknown_force_model='plane-separated')
    except ValueError:pass
    else:raise AssertionError('Invalid mode accepted')
    checks['public_factory_validated_without_GPU']=True
    rows=[json.loads(s) for s in (ROOT.parent/'output/unseen191_20261107/recording16/plane.jsonl').read_text('utf-8').splitlines()]
    maximum=0.
    for r in rows:
        if r['coefficient'] is None:continue
        pred=evaluate(r,[.1,.2,.4])
        for h in pred:maximum=max(maximum,float(np.max(np.abs(np.array(pred[h])-r['physical'][h]))))
    assert maximum<1e-9,maximum;checks['old191_rollout_max_delta']=maximum
    (out/'checks.json').write_text(json.dumps(checks,indent=2),'utf-8');print(json.dumps(checks),flush=True)
