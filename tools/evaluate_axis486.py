"""Frozen six-source paired replay of shared normal endpoint covariance."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from equilibrium_gain_prior import CovarianceOnlyObserver
from shared_axis_error import SharedAxisObserver
from replay_endpoint432 import run
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate

class Audit:
    def step(self,*args,**kwargs):
        c,v,f,s=super().step(*args,**kwargs)
        values=[];relative=[]
        for ob in (self,self.alternative):
            if ob.x is not None:
                assert np.isfinite(ob.x).all() and np.isfinite(ob.P).all()
                eig=np.linalg.eigvalsh((ob.P+ob.P.T)/2);values.append(float(eig.min()))
                relative.append(float(eig.min()/max(1.,np.max(abs(eig)))))
        s['covariance_eigenvalue_audit']=dict(minimum=min(values) if values else None,
            relative_minimum=min(relative) if relative else None)
        # Mixed radian/px units and the baseline's nearly singular cloned
        # position state require a scale-aware numerical PSD check.
        assert not relative or min(relative)>-1e-9,(values,relative)
        return c,v,f,s

class Control(Audit,CovarianceOnlyObserver):pass
class Shared(Audit,SharedAxisObserver):pass

if __name__=='__main__':
    out=ROOT.parent/'output/shared_axis486b_20261112';out.mkdir(exist_ok=False)
    files=['normal_response_observer.py','shared_axis_error.py','tools/check_axis486.py','tools/evaluate_axis486.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(source_sha256=hashes,normal_sigma_deg=5.,same_old_angular_prior=True,
        all6_uniform=True,no_fit_or_parameter_search=True,all_inference_before_references=True,
        native_geometry_frequencies_and_motion_frozen=True,reserved_video_not_read=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'))
    ns=('13','15','16','11','3','4');parity={};mineig={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/raw_mean473_20261111/{n}_joint_only/predictions.jsonl').read_text('utf8').splitlines()]
        for arm,cls in [('control',Control),('shared',Shared)]:
            rr=run(rows,cls,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation()
            vv=[win.step(vis.step(boot.step(r))) for r in rr]
            assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
            if arm=='control':
                delta=max([np.max(abs(np.array(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,rows) if r['valid']] or [0.])
                assert delta<1e-7,(n,delta);parity[n]=float(delta)
            mineig[n+'_'+arm]=min(r['state']['covariance_eigenvalue_audit']['minimum'] for r in vv if r['state']['covariance_eigenvalue_audit']['minimum'] is not None)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    cases={};pool={a:[] for a in ('control','shared')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            s=score(out/f'{n}_{a}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1))
            v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids]
            assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30p5=sum(x<=30.5 for x in v)) for a,v in pool.items()}
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result=dict(cases=cases,pooled=pooled,control_parity=parity,covariance_min_eigenvalue=mineig,
        original102_preserved=True,new_video_validated=False,FPS_work=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print('POOLED',pooled,flush=True)
