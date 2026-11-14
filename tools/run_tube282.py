"""One conditional recovery, original geometry coordinates and baseline kept."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from audit_continuity266 import model
from run_rotor_only import serial,SOURCE_FILES
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from motion_normalized_input import MotionNormalizedInput
from motion_tube_fallback import backproject,qualified
from probe_motion_filter279 import cache_path
from replay_joint236 import source
from score_rotor164 import score
from moving_event_proposal import propose
from rotor_spatial_bridge import recover as spatial_bridge

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['1','13','15','16'],required=True);p.add_argument('--motion-prior',action='store_true');p.add_argument('--relocalize',action='store_true');p.add_argument('--spatial-bridge',action='store_true');args=p.parse_args();n=args.recording
    if args.spatial_bridge and not args.motion_prior:p.error('Spatial bridge requires inertial match search')
    if args.relocalize and not args.motion_prior:p.error('Relocalization profile requires inertial match search')
    stage='tube285' if args.spatial_bridge else 'tube284' if args.relocalize else 'tube283' if args.motion_prior else 'tube282'
    OUT=ROOT.parent/f'output/{stage}_20261108/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    files=sorted(set(list(SOURCE_FILES)+['tools/run_tube282.py','motion_tube_fallback.py','motion_normalized_input.py','moving_event_proposal.py','native_event_alignment.py','rotor_spatial_bridge.py']))
    files=[f for f in files if (ROOT/f).is_file()];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(recording=n,source_sha256=hashes,change='only failed raw entry: aligned periodic support backprojected to original event coordinates',
        original_count_planes_and_frequency_fields_retained=True,original_qualified_entry_unchanged=True,
        virtual_pixel_frequency_not_used_as_mechanical_RPM=True,all_recovered_thrust_in_unknown_force=True,
        no_threshold_or_force_parameter_tuning=True,no_recording_specific_rules=True,
        NN_used=False,GPU_used=True,RAW_FPS_measured=False,labels_read_during_inference=False,
        recording1_is_development=True,known_limit='single dominant moving activity region',
        existing_native_inertia_search_enabled=args.motion_prior,current_region_relock_kept=True,
        early_half_relocalization_after_failed_past_motion_proof=args.relocalize,
        recent_period_identity_current_spatial_bridge=args.spatial_bridge)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    models={arm:model(worker) for arm in ['control','tube']};normalizer=MotionNormalizedInput();outputs={arm:[] for arm in models};delta=0.;used=0
    models['tube'].front.motion_prior=args.motion_prior
    last_period_t=None
    oldpath=ROOT.parent/'output/frozen_unseen278_20261108/control.jsonl' if n=='1' else source(n)
    previous=[json.loads(s) for s in oldpath.read_text('utf8').splitlines()];cache=cache_path(n)
    try:
        for i,old in enumerate(previous):
            t=old['t'];events=window(cache,t);raw=worker.prepare_uncached(events,t);candidate=raw;info=dict(used=False,reason='original_period_entry_already_qualified')
            if not qualified(raw):
                aligned,proposal=normalizer.prepare(events,t,raw[0],raw[1]);info=dict(used=False,reason='no_motion_proposal',proposal=proposal)
                if aligned is not None:
                    normalized=worker.prepare_uncached(aligned,t)
                    candidate,info=backproject(raw,normalized,events,t,proposal['velocity_px_s']);info['proposal']=proposal
                if args.relocalize and not info['used'] and proposal.get('velocity_source')=='past_confirmed_native_displacement':
                    # A failed past-velocity proof may indicate acceleration.
                    # Use the independent CURRENT EARLY half as a relock
                    # hypothesis, with the same unchanged late period gates.
                    early,relocation=propose(events,t,raw[0],raw[1],active=True)
                    if early is not None:
                        normalized=worker.prepare_uncached(early,t)
                        candidate,newinfo=backproject(raw,normalized,events,t,relocation['velocity_px_s'])
                        newinfo.update(proposal=relocation,past_motion_attempt=info,current_early_half_relocalization=True)
                        info=newinfo
            current_period=qualified(raw) or info['used']
            bridge=dict(used=False)
            if args.spatial_bridge and not current_period:
                candidate,bridge=spatial_bridge(raw,models['tube'].front,t,last_period_t)
                if bridge['used']:info=bridge
            used+=info['used']
            for arm,m in models.items():
                if arm=='tube':m.front.cycle_motion=bool(bridge['used'])
                worker.cached=(t,raw if arm=='control' else candidate);r,frame=m.step(events[:0],t);frame=None
                if arm=='control' and r['coefficient']:
                    for h,pred in r['physical'].items():delta=max(delta,float(np.max(abs(np.asarray(pred)-old['physical'][h]))))
                if arm=='tube':r['motion_tube_recovery']=info;normalizer.observe(r)
                outputs[arm].append(r)
            if current_period and outputs['tube'][-1]['body'].get('detected'):last_period_t=t
            if i%30==0:print(n,i,'control delta',delta,'recovered',used,'valid',sum(r['valid'] for r in outputs['tube']),flush=True)
        assert delta<1e-6,delta
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    for arm,rs in outputs.items():(OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf8')
    if n=='1':ref=ROOT.parent/'output/frozen_unseen278_20261108/references.json'
    else:
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n];ref=ROOT.parent/f'output/{folder}/references.json'
    origins=np.round(np.arange(10.8,15.6001,.2),1);scores={arm:score(OUT/f'{arm}.jsonl',ref,origins) for arm in outputs}
    diagnostics={arm:dict(frames=len(rs),current_target=sum(r['body'].get('detected',False) for r in rs),valid=sum(r['valid'] for r in rs)) for arm,rs in outputs.items()}
    (OUT/'results.json').write_text(json.dumps(dict(scores=scores,diagnostics=diagnostics,recovered=used,control_max_px=delta,protocol=protocol),indent=2),'utf8')
    print(n,{arm:s['error'] for arm,s in scores.items()},diagnostics,flush=True)
