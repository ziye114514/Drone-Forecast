"""Current strict-pixel confirmed track continuation -> actual prediction.

All per-pixel periodic thresholds retained. Native event features identical
for both branches. Current matching is computed once and checked both ways.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from check_inverse322 import model
from periodic_track_continuation import PeriodicTrackContinuation
from run_rotor_only import SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from extend_inverse321 import sources,references
from physical_publication import evaluate
from score_rotor164 import score
from compare_final_pipeline import stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['4','13','15','16','11','3'],default='4');p.add_argument('--batch',default='continuation338_20261109');p.add_argument('--appearance-uncertainty',action='store_true');p.add_argument('--side-line-reference',action='store_true');p.add_argument('--structural-motion',action='store_true');p.add_argument('--reference-uncertainty',action='store_true');p.add_argument('--keyframe-rotor-motion',action='store_true');p.add_argument('--occupancy-motion',action='store_true');p.add_argument('--public-continuation',action='store_true');a=p.parse_args();n=a.recording
    if a.occupancy_motion and (a.appearance_uncertainty or a.structural_motion or a.keyframe_rotor_motion):raise ValueError('Occupancy representation is a separate mechanism; do not mix log-count tests')
    out=ROOT.parent/f'output/{a.batch}/recording{n}';out.mkdir(parents=True,exist_ok=False)
    src=ROOT.parent/'output/frozen324_20261109/control.jsonl' if n=='4' else sources(n);rows=list(map(json.loads,src.read_text('utf8').splitlines()))
    expected=list(map(json.loads,(ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl').read_text('utf8').splitlines()))
    cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/frozen_tangent287_20261108/cache' if n=='3' else 'output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}')
    manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames'] if n in ('3','4') else None
    files=sorted(set(SOURCE_FILES+['keyframe_rotor_motion.py','keyframe_force_observer.py','tools/run_continuation338.py','tools/check_inverse322.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}
    protocol=dict(source_sha256=hashes,source=str(src),reference326_control=True,NN_used=False,GPU_used=True,RAW_FPS_measured=False,
        current_recovery_ONOFF_disagreement_covariance=a.appearance_uncertainty,
        current_side_line_midpoint_bias_reference=a.side_line_reference,
        current_shared_rotor_motion_for_recovered_measurements=a.structural_motion,
        reference_point_leverage_uncertainty=a.reference_uncertainty,direct_keyframe_rotor_motion=a.keyframe_rotor_motion,native_occupancy_motion=a.occupancy_motion,
        pixel_period_tests_unchanged=True,cold_entry_requirement_unchanged=True,known_target_requires_current_strict_period_and_bidirectional_image_match=True,
        no_recording_specific_rules=True,no_parameter_search=True,no_reference_before_inference_complete=True,
        no_new_motor_RPM_or_known_thrust_created=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in hashes:
        q=out/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    models={arm:model(worker,inverse_normal_response=True,periodic_continuation=bool(a.public_continuation and arm=='continued')) for arm in ('control','continued')}
    if not a.public_continuation:models['continued'].front.periodic_continuation=PeriodicTrackContinuation(appearance_uncertainty=a.appearance_uncertainty,structural_motion=a.structural_motion,reference_uncertainty=a.reference_uncertainty)
    protocol['actual_public_continuation_constructor_used']=a.public_continuation
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    outputs={a:[] for a in models};parity=0.;scored_prefix_parity=0.;used=0
    models['continued'].front.occupancy_motion=a.occupancy_motion
    if a.keyframe_rotor_motion:
        from keyframe_force_observer import KeyframeAdaptiveObserver
        from keyframe_rotor_motion import install
        models['continued'].force=KeyframeAdaptiveObserver();install(models['continued'].front)
    if a.side_line_reference:
        from functools import partial
        models['continued'].front.side_line_reference=True
        models['continued'].front.geometry.measure_body=partial(models['continued'].front.geometry.measure_body,center_uncertainty=True)
    try:
        for i,(old,wanted) in enumerate(zip(rows,expected)):
            t=old['t'];e=np.load(cache/manifest[i]['file']) if manifest is not None else window(cache,t);features=worker.prepare_uncached(e,t)
            for arm,m in models.items():
                worker.cached=(t,features);r,frame=m.step(e[:0],t);frame=None
                assert r['counts']['events']==old['counts']['events'],(arm,i,'wrong native events')
                if arm=='control':
                    assert r['valid']==wanted['valid'];assert bool(r['coefficient'])==bool(wanted['coefficient'])
                    if r['coefficient']:
                        for h,pred in r['physical'].items():
                            d=float(np.max(abs(np.asarray(pred)-wanted['physical'][h])));parity=max(parity,d)
                            if t<=15.600001:scored_prefix_parity=max(scored_prefix_parity,d)
                else:used+=bool((r.get('periodic_track_continuation') or {}).get('used'))
                outputs[arm].append(r)
                with (out/f'{arm}.jsonl').open('a',encoding='utf8') as fh:fh.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
            if i%30==0:print(n,i,'control parity',parity,'recovered',used,'valid',[sum(r['valid'] for r in rr) for rr in outputs.values()],flush=True)
        assert scored_prefix_parity<1e-7,scored_prefix_parity
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
    scores={arm:score(out/f'{arm}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1)) for arm in outputs}
    common=[(l,r) for l,r in zip(scores['control']['records'],scores['continued']['records']) if l['valid'] and r['valid']]
    extra=[r for l,r in zip(scores['control']['records'],scores['continued']['records']) if not l['valid'] and r['valid']]
    diagnostics={arm:dict(current_detections=sum(bool(r['body'].get('detected')) for r in rr),valid_frames=sum(r['valid'] for r in rr)) for arm,rr in outputs.items()}
    result=dict(cases=scores,diagnostics=diagnostics,control_forecast_parity=parity,all_scored_source_prefix_parity=scored_prefix_parity,
        full_source_suffix_not_reproduced=bool(parity>=1e-7),suffix_provenance_note='Legacy212 source11 differs from current projectedFOVguard at15.633s+, after every fixed query; retained and disclosed, not used to claim algorithm gain.',recovered_frames=used,
        common_origins=dict(control=stats([l['error_px'] for l,r in common]),continued=stats([r['error_px'] for l,r in common])),
        added_origins=extra,all_parameters_same_for_every_recording=True,protocol=protocol)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(n,{arm:s['error'] for arm,s in scores.items()},diagnostics,result['common_origins'],flush=True)
