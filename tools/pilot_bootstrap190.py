"""Frozen generic bootstrap on the newly diagnosed missing-detection input."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from moving_event_proposal import propose
from pure_visibility170 import window
BASE=ROOT.parent/'output/bootstrap190_20261107'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['18','19','17'],default='18');args=p.parse_args()
    BASE.mkdir(exist_ok=True);OUT=BASE/f'recording{args.recording}';OUT.mkdir(exist_ok=False);cv2.setNumThreads(2)
    parent=ROOT.parent/'output/motion181_20261107' if args.recording=='19' else ROOT.parent/'output/phase185_20261107'
    cache=parent/f'cache_recording{args.recording}'
    names=list(dict.fromkeys(SOURCE_FILES+['moving_event_proposal.py','native_event_alignment.py','tools/pilot_bootstrap190.py']))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    for n in names:
        p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(NN_used=False,labels_during_inference=False,
        scene_specific_parameters=False,parameters_swept=False,coordinate_warp_not_gray_reconstruction=True,
        candidate_is_not_detection=True,source_sha256=hashes),indent=2),'utf-8')
    def guard(event,items):
        if event=='open' and items and isinstance(items[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(items[0]))
            if p.name=='labels.json' or 'evaluation_data' in p.parts:raise RuntimeError('No labels during inference')
    sys.addaudithook(guard)
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True)
    recovery_worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,weak_regional=True)
    models={};rows={};traces=[];prepared=None;recovered=None;frame=None;active=False
    for arm in ['raw176','compensated_free','compensated_plane']:
        m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
            relative_observation=True,relock_support_guard=True,border_motion=True,legacy_visibility=False,
            gap_prediction='short',reference_mode='fixed',unknown_force_model='plane' if arm.endswith('plane') else 'free')
        m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer()
        models[arm]=m;rows[arm]=[]
    try:
        manifest=json.loads((cache/'manifest.json').read_text('utf-8'))
        for i,item in enumerate(manifest['frames']):
            t=item['t'];events=window(cache,t);prepared=worker.prepare_uncached(events,t)
            worker.cached=(t,prepared);r,frame=models['raw176'].step(events[:0],t);rows['raw176'].append(r);frame=None
            raw_valid=r['body'].get('detected',False)
            use=prepared;proposal=dict(qualified=False,reason='raw_observation_available');corrected=False
            if active or not raw_valid:
                warped,proposal=propose(events,t,prepared[0],prepared[1],active=active)
                if warped is not None:
                    recovered=recovery_worker.prepare_uncached(warped,t);accepted=int(recovered[-1]['accepted_pixels'])
                    native_mask,diagnostic=recovered[2]['native_precomputed']
                    collective=diagnostic.get('regional_seed_groups',0)>0 and diagnostic.get('regional_seed_pixels',0)>=3
                    period_pass=bool(np.count_nonzero(native_mask)>=12 and (accepted>=12 or collective))
                    proposal['validated_regional_groups']=diagnostic.get('regional_seed_groups',0)
                    proposal['validated_regional_pixels']=diagnostic.get('regional_seed_pixels',0)
                    proposal['validated_aligned_period_pixels']=accepted
                    if active or period_pass:
                        if not active and models['compensated_free'].front.center is not None:
                            proposal['activation_rejected_existing_raw_reference']=True
                        else:
                            active=True;use=recovered;corrected=True
            if active and not corrected:
                use=recovery_worker.prepare_uncached(events[:0],t);corrected=True
                proposal['missing_compensation_is_not_a_raw_reference_switch']=True
            for arm in ['compensated_free','compensated_plane']:
                worker.cached=(t,use);r,frame=models[arm].step(events[:0],t);frame=None
                r['moving_bootstrap']=dict(proposal,compensated_coordinates_used=corrected,
                    source_polarities_timestamps_unchanged=True,ROI_hypothesis_alone_is_not_detection=True)
                if corrected:r['geometry_uses_original_event_pixels']=False;r['geometry_uses_transformed_original_event_coordinates']=True
                rows[arm].append(r)
            traces.append(dict(t=t,active=active,proposal=proposal))
            if i%30==0:print(args.recording,i,raw_valid,proposal.get('validated_aligned_period_pixels'),rows['compensated_plane'][-1]['body'].get('detected',False),flush=True)
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        prepared=None;recovered=None;worker.cached=None;worker.close();recovery_worker.close()
    summary={}
    for arm,rs in rows.items():
        (OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
        summary[arm]=dict(frames=len(rs),regions=sum(r['body'].get('detected',False) for r in rs),valid=sum(r['valid'] for r in rs))
    (OUT/'proposals.json').write_text(json.dumps(traces,indent=2),'utf-8')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    (OUT/'completion.json').write_text(json.dumps(dict(summary=summary,source_changed=False,NN_used=False,RAW_FPS_measured=False),indent=2),'utf-8');print(json.dumps(summary),flush=True)
