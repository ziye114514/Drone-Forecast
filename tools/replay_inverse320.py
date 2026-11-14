"""Same native event state; causal inverse unknown-normal amplitude."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from inverse_normal_response import InverseNormalResponse,fit_bias,normal_drive_kernel
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score

def mechanisms():
    samples=[];bias=np.array([200.,3000.]);beta=1.4
    for i,t in enumerate(np.linspace(0,3,60)):
        angle=.3*np.sin(3*t);n=np.array([np.sin(angle),-np.cos(angle)]);q=np.array([-n[1],n[0]])
        v=np.array([600*np.cos(2*t),200*np.sin(t)]);a=bias-beta*v+(4000+1000*np.cos(t))*n
        samples.append((float(t),np.r_[q,-q@v],float(q@a),1.))
    fit=fit_bias(samples);assert fit['qualified'];assert np.max(abs(fit['bias']-bias))<.1;assert abs(fit['response']-beta)<1e-4
    constant=[(t,np.r_[[1.,0.],-200.],5.,4.) for t in np.linspace(0,1,20)];assert not fit_bias(constant)['qualified']
    from adaptive_force_memory import kernel
    nodes,weights=np.polynomial.legendre.leggauss(64);h=.4;lam=-1/.35;omega=.8;s=(nodes+1)*h/2
    acceleration=np.exp(lam*s)+1j*omega/(1j*omega-lam)*(np.exp(1j*omega*s)-np.exp(lam*s))
    numeric=np.sum(weights*(h-s)*acceleration)*h/2
    assert abs(normal_drive_kernel(lam,h,omega)-numeric)<1e-12
    assert abs(normal_drive_kernel(lam,h,0)-kernel(lam,h))<1e-15
    return dict(constant_plane_does_not_separate_hidden_thrust_and_external_bias=True,
        synthetic_varying_thrust_eliminated_before_bias_fit=True,normal_derivative_kernel_matches_numerical_dynamics=True)

if __name__=='__main__':
    out=ROOT.parent/'output/inverse320_20261109';out.mkdir(exist_ok=False)
    files=['inverse_normal_response.py','physical_publication.py','tools/replay_inverse320.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    checks=mechanisms();parity={};results={};diagnostics={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];models=dict(kernel_only=InverseNormalResponse(calibrate=False),inverse=InverseNormalResponse());outputs={k:[] for k in models};delta=0.
        for old in rows:
            if old['coefficient']:
                for h,p in evaluate(old,[.1,.2,.4]).items():delta=max(delta,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
            for arm,m in models.items():
                r=m.step(old);r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs[arm].append(r)
                assert r['coefficient']==old['coefficient'];assert r['force']['acceleration_px_s2']==old['force']['acceleration_px_s2']
        assert delta<1e-7;parity[n]=delta
        for arm,rr in outputs.items():(out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        items=[c for r in outputs['inverse'] for c in r['inverse_normal_response'].get('components',[]) if c['applied']]
        diagnostics[n]=dict(calibrated_packets=len(items),median_reliability=None if not items else float(np.median([i['reliability'] for i in items])))
        print(n,'parity',delta,diagnostics[n],flush=True)
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n];ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in models};print(n,{a:v['error'] for a,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (out/'results.json').write_text(json.dumps(dict(cases=results,diagnostics=diagnostics,checks=checks,control_parity=parity,
        NN_used=False,RAW_FPS_measured=False,no_recording_specific_rules=True,no_parameter_search=True,
        all_inference_before_reference_reads=True,current_state_frozen=True,metric_forces_not_measured=True),indent=2),'utf8')
