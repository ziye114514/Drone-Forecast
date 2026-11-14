"""Complete forecast after translation-normalized rotor-only filtering.

Uniform policy on all packets and recordings. It reuses189's activity ROI and
earlier-half velocity, original period gates and current force observer. The
warped representation is in the camera's cutoff-time coordinates, not a new
camera or reconstructed gray channel. Original RAW/arrays are never edited.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from audit_continuity266 import model
from run_rotor_only import serial,SOURCE_FILES
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from moving_event_proposal import propose
from probe_motion_filter279 import cache_path,evidence
from replay_joint236 import source
from score_rotor164 import score
from motion_normalized_input import MotionNormalizedInput

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['1','13','15','16'],required=True);p.add_argument('--velocity-policy',choices=['early','tracked'],default='early');args=p.parse_args();n=args.recording
    stage='motion280' if args.velocity_policy=='early' else 'motion281'
    OUT=ROOT.parent/f'output/{stage}_20261108/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    files=sorted(set(list(SOURCE_FILES)+['tools/run_motion280.py','moving_event_proposal.py','native_event_alignment.py','tools/probe_motion_filter279.py','motion_normalized_input.py']))
    files=[f for f in files if (ROOT/f).is_file()];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(recording=n,source_sha256=hashes,
        change='translation normalization BEFORE periodic filter, on every packet',velocity_policy=args.velocity_policy,
        no_threshold_or_force_parameter_tuning=True,no_recording_specific_rules=True,NN_used=False,GPU_used=True,
        RAW_FPS_measured=False,labels_read_during_inference=False,recording1_development_after_failure278=True,
        geometry276_and_rate277_not_used=True,coordinate_representation='camera image coordinates at cutoff; rounded event-time translation; cropped to activity ROI',
        known_limit='single moving high-activity target, not an all-scene or multi-target detector')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    models={arm:model(worker) for arm in ['control','normalized']};outputs={arm:[] for arm in models};traces=[];delta=0.
    normalizer=MotionNormalizedInput()
    oldpath=ROOT.parent/'output/frozen_unseen278_20261108/control.jsonl' if n=='1' else source(n)
    previous=[json.loads(s) for s in oldpath.read_text('utf8').splitlines()];cache=cache_path(n)
    try:
        for i,old in enumerate(previous):
            t=old['t'];events=window(cache,t);raw=worker.prepare_uncached(events,t);raw_evidence=evidence(raw)
            if args.velocity_policy=='tracked':warped,proposal=normalizer.prepare(events,t,raw[0],raw[1])
            else:warped,proposal=propose(events,t,raw[0],raw[1],active=True)
            normalized=raw if warped is None else worker.prepare_uncached(warped,t)
            aligned_evidence=evidence(normalized)
            for arm,m in models.items():
                worker.cached=(t,raw if arm=='control' else normalized);r,frame=m.step(events[:0],t)
                if arm=='control' and r['coefficient']:
                    for h,pred in r['physical'].items():delta=max(delta,float(np.max(abs(np.asarray(pred)-old['physical'][h]))))
                if arm=='normalized':
                    r.update(moving_period_filter=dict(proposal,enabled=warped is not None,
                        original_native_timestamps_and_polarities=True,spatial_event_coordinates_translated=warped is not None,
                        raw_qualification=raw_evidence,normalized_qualification=aligned_evidence),
                        geometry_uses_original_event_pixels=warped is None,
                        geometry_uses_transformed_original_event_coordinates=warped is not None)
                outputs[arm].append(r);frame=None
                if arm=='normalized' and args.velocity_policy=='tracked':normalizer.observe(r)
            traces.append(dict(t=t,proposal=proposal,raw=raw_evidence,normalized=aligned_evidence))
            if i%30==0:print(n,i,'control delta',delta,'valid',sum(r['valid'] for r in outputs['control']),sum(r['valid'] for r in outputs['normalized']),flush=True)
        assert delta<1e-6,delta
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    for arm,rs in outputs.items():(OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf8')
    if n=='1':ref=ROOT.parent/'output/frozen_unseen278_20261108/references.json'
    else:
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json'
    origins=np.round(np.arange(10.8,15.6001,.2),1);scores={arm:score(OUT/f'{arm}.jsonl',ref,origins) for arm in outputs}
    diagnostics={arm:dict(frames=len(rs),current_target=sum(r['body'].get('detected',False) for r in rs),valid=sum(r['valid'] for r in rs)) for arm,rs in outputs.items()}
    (OUT/'results.json').write_text(json.dumps(dict(scores=scores,diagnostics=diagnostics,control_max_px=delta,traces=traces,protocol=protocol),indent=2),'utf8')
    print(n,{arm:s['error'] for arm,s in scores.items()},diagnostics,flush=True)
