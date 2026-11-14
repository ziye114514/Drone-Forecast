"""Geometry-only exact-input control before testing motion-smear hypothesis."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse,copy,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,serial
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from replay_joint236 import source


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recording',default='15');args=parser.parse_args()
    OUT=ROOT.parent/f'output/geometry259_20261108/recording{args.recording}';OUT.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in source(args.recording).read_text('utf-8').splitlines()]
    parent='motion181_20261107' if args.recording=='15' else 'phase185_20261107';cache=ROOT.parent/f'output/{parent}/cache_recording{args.recording}'
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    model=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
        relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',
        unknown_force_model='plane-adaptive',projected_fov_guard=True)
    geometry=model.front.geometry;outputs=[];maxangle=0.;maxnorm=0.;counts=0;modes=0
    for i,r in enumerate(rows):
        t=r['t'];events=window(cache,t);tick=time.perf_counter();m=r['motion'];est=m.get('estimate') or {};dt=m.get('dt_s');v=np.zeros(2) if not dt else np.array([est.get('dx',0.),est.get('dy',0.)])/dt
        g,_,diag,_=geometry.step(events,t,copy.deepcopy(r['body']),None,v,defer_rates=True,
            defer_geometry=bool(r['geometry'].get('geometry_observation_deferred')))
        a=g.get('axis_xy');b=r['structure'].get('axis_xy');angle=None;norm=None
        if a is not None and b is not None:
            a=np.asarray(a);b=np.asarray(b);angle=float(np.rad2deg(np.arccos(np.clip(abs(a@b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12),-1.,1.))))
            norm=float(min(np.linalg.norm(a-b),np.linalg.norm(a+b)));maxangle=max(maxangle,angle);maxnorm=max(maxnorm,norm)
        counts+=len(diag.get('rotors',[]))!=len(r['geometry'].get('rotors',[]));modes+=g['mode']!=r['structure']['mode']
        outputs.append(dict(t=t,structure=g,geometry=diag,normal_angle_difference_deg=angle,normal_vector_difference=norm,
            native_geometry_ms=1000*(time.perf_counter()-tick),source_structure=r['structure']))
        if i%30==0:print(args.recording,i,'count mismatches',counts,'mode mismatches',modes,'max normal angle',maxangle,flush=True)
    configure_native(None)
    (OUT/'geometry_control.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf-8')
    result=dict(frames=len(rows),rotor_count_mismatches=counts,mode_mismatches=modes,max_normal_difference_deg=maxangle,
        max_normal_vector_difference=maxnorm,NN_used=False,GPU_used=False,RAW_FPS_measured=False,warp_enabled=False,
        control_must_pass_before_accuracy_trial=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)
