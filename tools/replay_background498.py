"""One root: background reference BEFORE two-force filtering, fixed487 front.

Compare original, matched uncertainty-only control, and observed background
mean compensation. All six inference outputs precede reference reads.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from equilibrium_gain_prior import CovarianceOnlyObserver
from inverse_normal_response import InverseNormalResponse
from anisotropic_response import AnisotropicResponse
from plane_conditioned_response import PlaneResponse
from reference_qualification import QualifiedReferenceResponse
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from background_relative_force import BackgroundReference
from replay_endpoint432 import run
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate,freeze

def compensated(rows,evidence,model,noise_only,*,image_calibration=False,shared_endpoints=False,preserve_foreground_floor=False):
    if shared_endpoints:
        from background_endpoint_observer import BackgroundEndpointObserver
        observer=BackgroundEndpointObserver()
    else:observer=CovarianceOnlyObserver()
    inv=InverseNormalResponse();disk=AnisotropicResponse()
    response=QualifiedReferenceResponse(PlaneResponse(model));boot=BootstrapNativeReference()
    visible=VisibleEnvelopeResponse();window=WindowPositionObservation();ref=BackgroundReference(noise_only,shared_endpoints,preserve_foreground_floor);outputs=[]
    for old,ev in zip(rows,evidence):
        assert ev['t']==old['t'];ref.advance(ev);r=copy.deepcopy(old)
        for k in ('conditional_unknown_force','visibility_response','plane_conditioned_response','future_drive_response',
                  'native_fov_observation','physical','window_position_observation','native_reference_bootstrap'):r.pop(k,None)
        body,motion=ref.inputs(r['t'],r['body'],r['motion']);r.update(body=body,motion=motion)
        u=r['force'].get('input_u');u=None if u is None else np.asarray(u,float);meta=r['period_selection']
        fresh=observer.x is None or observer.seen is not None and r['t']-observer.seen>.5
        if fresh and body.get('detected') and not body.get('center_provisional'):
            observer.two_point_init=bool(old['state'].get('two_point_initialization_selected_by_partial_entry'))
        c,v,f,s=observer.step(r['t'],body,u,.7 if u is not None else 0.,
            allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=motion,structure=r['structure'])
        s['two_point_initialization_selected_by_partial_entry']=bool(old['state'].get('two_point_initialization_selected_by_partial_entry'))
        r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
        r=disk.step(inv.step(r))
        if image_calibration:r=ref.attach(r,old,readout_in_image_coordinates=True)
        r=response.step(r);r=boot.step(r);r=window.step(visible.step(r))
        if not image_calibration:r=ref.attach(r,old)
        assert r['valid']==old['valid'] and r['body']==old['body'] and r['motion']==old['motion']
        if r['valid']:
            assert np.isfinite(evaluate(r,[.4])['0.4']).all()
            assert np.max(abs(np.asarray(evaluate(freeze(r,r['t']),[.4])['0.4'])-evaluate(r,[.4])['0.4']))<1e-7
        outputs.append(r)
    return outputs

if __name__=='__main__':
    out=ROOT.parent/'output/background_filter498_20261112';out.mkdir(exist_ok=False)
    files=['background_relative_force.py','physical_publication.py','tools/replay_background498.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,original102_preserved=True,
        one_uniform_rule_all6=True,no_parameter_or_tau_search=True,camera_tau_reuses_existing035_s=True,
        before_filter_coordinate_compensation_not_old428_posthoc_weight=True,
        uncertainty_only_control_included=True,no_model_fit=True,
        native_rotor_geometry_period_and_detector_frozen=True,all_inference_before_references=True,
        reserved_video_unread=True,FPS_work=False),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));parity={}
    ns=('13','15','16','11','3','4')
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl').read_text('utf8').splitlines()]
        ev=[json.loads(s) for s in (ROOT.parent/f'output/background_reference497_20261112/{n}.jsonl').read_text('utf8').splitlines()]
        zero=[dict(t=r['t'],proof={},pair=None) for r in rows]
        vv=compensated(rows,zero,model,False)
        delta=max(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,rows) if r['valid'])
        assert delta<1e-7,(n,delta);parity[n]=float(delta)
        for a,rr in [('control',rows),('noise_only',compensated(rows,ev,model,True)),('reference',compensated(rows,ev,model,False))]:
            (out/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        print(n,'all3 inference frozen;zero-background parity',delta,flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    cases={};pool={a:[] for a in ('control','noise_only','reference')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            s=score(out/f'{n}_{a}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1))
            v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids)
            pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30p5=sum(x<=30.5 for x in v)) for a,v in pool.items()}
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,no_background_control_parity=parity,
        original102_preserved=True,native_foreground_frozen=True,new_video_validated=False,FPS_work=False),indent=2),'utf8')
    print('POOLED',pooled,flush=True)
