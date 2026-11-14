"""Freeze anisotropic response before reading any trajectory references."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from anisotropic_response import AnisotropicResponse,rollout
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    out=ROOT.parent/'output/anisotropic354_20261109';out.mkdir(exist_ok=False)
    files=['anisotropic_response.py','physical_publication.py','tools/probe_anisotropic354.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    # Isotropic limit must reproduce the independent existing closed form,
    # including angular acceleration and the alternative external jerk state.
    rng=np.random.default_rng(354);maximum=0.
    for i in range(12):
        n=rng.normal(size=2);n/=np.linalg.norm(n);A=float(rng.uniform(0,2000));beta=1/.35
        c=dict(position=rng.normal(size=2).tolist(),velocity=(rng.normal(size=2)*500).tolist(),acceleration=(rng.normal(size=2)*1000).tolist())
        f=dict(acceleration_px_s2=c['acceleration'],acceleration_correlation_s=.35,normal_turn_vector_px_s2=(A*n).tolist(),future_normal_response_model='drag_filtered_normal_derivative',external_unknown_rate_px_s3=(rng.normal(size=2)*300).tolist(),external_rate_correlation_s=.35)
        normal=dict(applied=True,trend=dict(shrunk_angular_rate_rad_s=float(rng.uniform(-2,2)),angular_acceleration_rad_s2=float(rng.uniform(-3,3)),angular_acceleration_correlation_s=.35))
        r=dict(coefficient=c,force=f,normal_turn_forecast=normal);expected=evaluate(r,[0.,.1,.4,1.])
        new=copy.deepcopy(r);new['force']['anisotropic_disk_response']=dict(normal_xy=n.tolist(),tangent_response_s_inv=beta,normal_response_s_inv=beta,normal_drive_px_s2=A)
        actual=evaluate(new,[0.,.1,.4,1.])
        maximum=max(maximum,max(float(np.max(abs(np.asarray(actual[h])-v))) for h,v in expected.items()))
    assert maximum<2e-4,maximum
    cases={};counts={}
    for n in ('13','15','16','11','3','4'):
        rows=[json.loads(s) for s in (ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl').read_text('utf8').splitlines()];model=AnisotropicResponse();outputs=[];used=0
        for row in rows:
            r=model.step(row);assert r['coefficient']==row['coefficient'];assert r['body']==row['body'];assert r['valid']==row['valid']
            used+=r['anisotropic_response']['applied'];outputs.append(r)
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8');counts[n]=used;print(n,'frozen',used,flush=True)
    # All causal model output is frozen before reference access.
    for n in counts:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        cases[n]=dict(control=score(ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1)),candidate=score(out/f'{n}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1)))
        print(n,{k:v['error'] for k,v in cases[n].items()},flush=True)
    result=dict(cases=cases,applied_frames=counts,isotropic_closed_form_max_error_px=maximum,source_sha256=hashes,
        no_parameter_search=True,no_recording_specific_rules=True,all_cases_exposed_development=True,
        no_new_RAW_or_FPS_measured=True,NN_used=False,GPU_used=False,current_state_and_visibility_unchanged=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8')
