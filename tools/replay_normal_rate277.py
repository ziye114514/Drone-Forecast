"""One unknown-force rate change, identical measured front on all clips."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from normal_rate_response import NormalRateAdaptive,NormalRateSeparated,normal_rate_kernel,evaluate
from adaptive_response_observer import AdaptiveResponseObserver
from physical_publication import evaluate as original_evaluate
from separated_force_response import rate_kernel
from replay_joint236 import source
import score_rotor164 as scoring
from run_rotor_only import serial

if __name__=='__main__':
    OUT=ROOT.parent/'output/normal_rate277_20261108';OUT.mkdir(exist_ok=False)
    files=['normal_rate_response.py','tools/replay_normal_rate277.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(change='existing unknown acceleration rate updates the normal amplitude h; same14state/noise/qualification',
        source_sha256=hashes,development=['13','15','16','11'],heldout1_opened=False,
        no_parameter_search=True,no_recording_specific_rules=True,NN_used=False,GPU_used=False,
        observed_geometry_motion_and_periods_frozen=True,actual_mechanical_thrust_unavailable=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    m=NormalRateSeparated();m._initialize(np.array([100.,200.]));m.normal=np.array([.3,-.9]);psd=[]
    for decay in [False,True]:
        m.consistent_dynamics=decay
        for dt in [.000167,.033333,.1]:
            F,Q=m._transition(dt);psd.append(float(np.linalg.eigvalsh(Q).min()));assert psd[-1]>-1e-5,psd
    checks=dict(process_covariance_min_eigenvalue=min(psd))
    parity=max(abs(normal_rate_kernel(-1/tau,-1/.35,0.,h)-rate_kernel(-1/tau,-1/.35,h)) for tau in [.1,.35] for h in [0.,.1,.4])
    assert parity<1e-12,parity;checks['zero_rotation_rate_kernel_parity']=parity
    results={}
    for n in protocol['development']:
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        models=dict(control=AdaptiveResponseObserver(),rate=NormalRateAdaptive());outputs={arm:[] for arm in models};delta=0.
        for old in rows:
            for arm,m in models.items():
                r=copy.deepcopy(old);pm=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
                c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=r['structure'])
                r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
                r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs[arm].append(r)
                if arm=='control' and c:
                    for h,p in r['physical'].items():delta=max(delta,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
        assert delta<1e-7,delta
        for arm,rs in outputs.items():(OUT/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf8')
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16','11':'regional212_20261108/reference11'}[n]
        #11 uses the same saved reference convention as its previous scoring.
        if n=='11':
            folder='regional212_20261108/reference11' if (ROOT.parent/f'output/{folder}/references.json').exists() else 'regional212_20261108/ref11'
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        scoring.evaluate=evaluate
        results[n]={arm:scoring.score(OUT/f'{n}_{arm}.jsonl',ref,origins) for arm in outputs}
        print(n,'control parity',delta,{k:v['error'] for k,v in results[n].items()},flush=True)
        (OUT/'partial.json').write_text(json.dumps(dict(scores=results,checks=checks,protocol=protocol),indent=2),'utf8')
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(scores=results,checks=checks,protocol=protocol),indent=2),'utf8')
