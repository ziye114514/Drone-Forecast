"""Isolate20ms motion smear: exact masked-count control vs causal translation.

No body/motion/period observations, thresholds or force priors are changed.
Geometry input timestamps and polarities are preserved; only current selected
event coordinates are aligned using a velocity older than the whole20ms.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,copy,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,serial
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from native_event_alignment import past_warp
from fast_structure_front import count_pair
from adaptive_response_observer import AdaptiveResponseObserver
from physical_publication import evaluate
from replay_joint236 import source
from score_rotor164 import score


def make_geometry():
    return RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
        relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',
        unknown_force_model='plane-adaptive',projected_fov_guard=True).front.geometry


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['13','15','16'],required=True);args=p.parse_args()
    n=args.recording;OUT=ROOT.parent/f'output/geometry261_20261108/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    captured=ROOT.parent/f'output/geometry260_20261108/recording{n}';manifest=json.loads((captured/'manifest.json').read_text('utf-8'))
    rows=[json.loads(s) for s in source(n).read_text('utf-8').splitlines()];assert len(rows)==len(manifest['records'])
    (OUT/'protocol.json').write_text(json.dumps(dict(recording=n,geometry_window_us=20000,past_velocity_only=True,
        fixed_original_spatial_support=True,position_velocity_motion_and_period_inputs_unchanged=True,
        geometry_intervention_only=True,NN_used=False,GPU_used=False,labels_during_inference=False,
        thresholds_or_force_parameters_changed=False,no_recording_specific_rules=True,
        reserved_experiment1_opened=False,RAW_FPS_measured=False,
        files_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['tools/replay_geometry261.py','native_event_alignment.py','fast_structure_front.py']}),indent=2),'utf-8')
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    geometry={k:make_geometry() for k in ['control','aligned']};models={k:AdaptiveResponseObserver() for k in geometry}
    outputs={k:[] for k in geometry};hist=[];delta=0.;normdelta=0.;countdiff=0;modediff=0;applied=0;audits=[]
    for i,(old,item) in enumerate(zip(rows,manifest['records'])):
        t=old['t'];assert abs(t-item['t'])<1e-9
        hist=[v for v in hist if t-v[0]<=.16]
        admissible=[v for v in hist if v[0]<=t-.020+1e-7];prior=admissible[-1] if admissible else None
        data=None if 'file' not in item else np.load(captured/item['file'])
        for arm,g in geometry.items():
            meta=dict(applied=False,reason='no_current_geometry_or_past_velocity')
            if data is not None:
                on,off=data['on'],data['off']
                if arm=='aligned' and prior is not None:
                    e,meta=past_warp(data['events'],t,prior[1],prior[0],window_us=20000)
                    if meta['applied']:
                        on,off=count_pair(e,np.asarray(item['bounds']),t,cpp_histogram=True);applied+=1
                bounds_expected=np.asarray(item['bounds'])
                def counts(_e,bounds,_t,_selector,on=on,off=off,bounds_expected=bounds_expected):
                    assert np.array_equal(bounds,bounds_expected);return on,off
                g.count_current=counts
            body=copy.deepcopy(item['body'])
            structure,_,diag,_=g.step(np.empty(0),t,body,None,np.asarray(item['velocity']),**item['kwargs'])
            post=item['post_body'].get('center');before=body.get('center')
            if post is not None and before is not None:
                shift=np.asarray(post)-before
                if np.any(shift):g.rebase_body_reference(shift)
            if arm=='control':
                countdiff+=len(diag.get('rotors',[]))!=len(item['source_geometry'].get('rotors',[]));modediff+=structure['mode']!=item['source_structure']['mode']
                a=structure.get('axis_xy');b=item['source_structure'].get('axis_xy')
                if a is not None and b is not None:normdelta=max(normdelta,float(np.linalg.norm(np.asarray(a)-b)))
                else:assert (a is None)==(b is None)
            r=copy.deepcopy(old);pm=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
            m=models[arm]
            if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
            c,v,f,s=m.step(t,r['body'],u,.7 if u is not None else 0.,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=structure)
            r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'),structure=structure,geometry_motion_alignment=meta)
            r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs[arm].append(r)
            if arm=='control' and c:
                for h,value in r['physical'].items():delta=max(delta,float(np.max(abs(np.asarray(value)-old['physical'][h]))))
            audits.append(dict(t=t,arm=arm,alignment=meta,structure=structure,geometry=diag))
        if data is not None:data.close()
        motion=old['motion'];estimate=motion.get('estimate') or {};dt=motion.get('dt_s')
        if dt and motion.get('current_observation') and estimate.get('score',0)>0 and not estimate.get('peak_at_search_boundary') and estimate.get('current_displacement_qualified',True):
            hist.append((t,np.array([estimate['dx'],estimate['dy']])/dt))
        if i%30==0:print(n,i,'control count/mode differences',countdiff,modediff,'max forecast delta',delta,flush=True)
    configure_native(None)
    checks=dict(control_forecast_max_px=delta,control_normal_max_difference=normdelta,control_count_differences=countdiff,
        control_mode_differences=modediff,aligned_windows=applied,source_timestamps_and_polarity_unchanged=True)
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2),'utf-8')
    assert delta<1e-6 and normdelta<1e-8 and countdiff==0 and modediff==0,checks
    for arm,rs in outputs.items():(OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
    (OUT/'geometry_audits.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in audits),'utf-8')
    folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
    ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
    result={k:score(OUT/f'{k}.jsonl',ref,origins) for k in outputs}
    (OUT/'results.json').write_text(json.dumps(dict(checks=checks,scores=result),indent=2),'utf-8')
    print(n,{k:v['error'] for k,v in result.items()},flush=True)
