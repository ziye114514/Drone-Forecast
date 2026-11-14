"""RAW current point-pair replay with frozen bodies; no extra NN/flow pass."""
from pathlib import Path
import sys,json,time,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_fast_structure import track,PhysicalPositionState,js
from shared_image_motion import SharedImageMotion
from flow_force_balance import FlowBalancedForce
from native_render import occupancy_frame
from live_decode import native_batches
from compare_final_pipeline import score


def main(run,label,output):
    import cv2
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    cv2.setNumThreads(2)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);source=Path(run)
    protocol=json.loads((source/'protocol.json').read_text('utf-8'))
    rows=[json.loads(s) for s in (source/'predictions.jsonl').read_text('utf-8').splitlines()]
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer()
    state=PhysicalPositionState();helpers=[SharedImageMotion(),SharedImageMotion()];forces=[FlowBalancedForce(),FlowBalancedForce()]
    values=[[],[]];previous=None;oldbody=None;oldstructure=None;costs=[];delta_max=0.;rigid_count=0
    for (t,e),r in zip(native_batches(protocol['raw'],protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us']),rows):
        assert abs(t-r['t'])<1e-8 and len(e)==r['raw_events']
        noise.process_events(e,buffer);clean=buffer.numpy();clean=clean[clean['t']>=round(t*1e6)-20000]
        gray=occupancy_frame(clean);hint=None
        if oldstructure is not None:
            cov=np.asarray(oldstructure.get('model_conditioned_center_covariance',np.eye(2)*1e9))
            if oldstructure.get('mode') in ('four','three','partial_two') and cov.shape==(2,2) and np.linalg.eigvalsh(cov).max()<=36.:
                hint=oldstructure.get('model_conditioned_center')
        start=time.perf_counter();_,tracking=track(previous,gray,oldbody,rigid_motion=True,center_hint=hint)
        costs.append((time.perf_counter()-start)*1000)
        if 'delta' in tracking and 'delta' in r['tracking']:
            delta_max=max(delta_max,float(abs(np.asarray(tracking['delta'])-r['tracking']['delta']).max()))
        assert delta_max<1e-5,delta_max
        rigid_count+=bool(tracking.get('similarity_motion',{}).get('valid'))
        body=copy.deepcopy(r['body']);bridge=r['center_bridge']
        if bridge.get('applied'):body['center']=bridge['fused_position']
        state.kf.noise_scale=bridge.get('noise_scale',1.);c,valid,_=state.step(t,body)
        assert valid==r['valid']
        if c is not None:assert np.allclose(c['position'],r['coefficient']['position'],atol=1e-8,rtol=0)
        for j in [0,1]:
            info_track=dict(tracking)
            if j==0:info_track.pop('similarity_motion',None)
            coefficient,info=helpers[j].step(t,c,state.kf.P[2:4,2:4] if c else np.eye(2),body,info_track)
            forces[j].input_learning_current=(r.get('axis_amplitude_bridge') or {}).get('learning_input_current',True)
            forces[j].observe_flow(info['observation'],info['acceleration'].get('applied',False))
            p,f=forces[j].step(t,coefficient,r.get('force_observed_position'),r['force']['input_u'],.35)
            values[j].append(dict(r,tracking=info_track,coefficient=coefficient,image_motion=info,force=f,physical=p))
        previous=gray;oldbody=r['body'];oldstructure=r['structure']
    assert len(values[0])==len(rows)
    for name,data in zip(['same_tracks_median','same_tracks_rigid'],values):
        (out/(name+'.jsonl')).write_text('\n'.join(js(r) for r in data)+'\n','utf-8')
    # Only now read manual references. Bodies/position/filter eligibility above
    # are the existing causal detector output, not annotations or interpolation.
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    result=dict(source_frames=len(rows),frozen_causal_body_inputs=True,position_and_validity_unchanged=True,
        additional_LK_passes=0,RAW_end_to_end_FPS_not_measured=True,accepted_current_similarities=rigid_count,
        original_KLT_displacement_max_difference=delta_max,mean_existing_flow_and_added_fit_ms=float(np.mean(costs)),
        median_added_fit_ms=float(np.median([r['tracking'].get('similarity_motion',{}).get('CPU_ms',0) for r in values[1]])),
        scores={n:score(v,truth)['physical_future_error_px'] for n,v in zip(['same_tracks_median','same_tracks_rigid'],values)})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(*sys.argv[1:])
