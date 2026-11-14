"""Causal short holdout on one automatically refitted native rotor proposal.

Warm geometry/phase use only events <=seed cutoff. Next source packets test
frozen pre-packet phase before updating. Both directions are retained. No GT.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from helixtrack_adapter import HelixCore
from pure_visibility170 import window
OUT=ROOT.parent/'output/helix_bridge180_20261107'


def transformed(events,ellipse,body_delta,velocity,cutoff):
    xy=np.c_[events['x'],events['y']].astype(float)+(cutoff-events['t']/1e6)[:,None]*velocity-body_delta
    # Preserve sensor ABI; this particular ROI is away from boundaries.
    good=(xy[:,0]>=0)&(xy[:,0]<1280)&(xy[:,1]>=0)&(xy[:,1]<720)
    e=events[good].copy();e['x']=np.rint(xy[good,0]).astype(np.uint16);e['y']=np.rint(xy[good,1]).astype(np.uint16)
    a=np.deg2rad(ellipse['angle']);d=np.c_[e['x'],e['y']]-[ellipse['cx'],ellipse['cy']]
    u=(d[:,0]*np.cos(a)+d[:,1]*np.sin(a))/(ellipse['major']/2)
    v=(-d[:,0]*np.sin(a)+d[:,1]*np.cos(a))/(ellipse['minor']/2)
    radius=np.hypot(u,v);return e[(radius>=.2)&(radius<=1.5)]


def phase_geometry(e,ellipse,state,direction):
    xy=np.c_[e['x'],e['y']]-[ellipse['cx'],ellipse['cy']]
    H=np.asarray(state['H']);u=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(H).T;u=u[:,:2]/u[:,2:]
    return np.arctan2(direction*u[:,1],u[:,0])


if __name__=='__main__':
    reviewed_C='C' in sys.argv
    if reviewed_C:
        review=json.loads((OUT/'review_C_ring_phases.json').read_text('utf-8'))
        candidate=next(r for r in review['rotors'] if r['phase']['qualified'])
        ellipse=candidate['ellipse'];prior=candidate['phase']['phase_hz'];seed_t=review['t']
        rows=[json.loads(s) for s in (ROOT.parent/'output/layout176_20261107/integrated/side_fixed.jsonl').read_text('utf-8').splitlines()]
        seed=max((r for r in rows if r['t']<=seed_t),key=lambda r:r['t'])
        selected=[]
        for t in seed_t+np.arange(4)*.033333:
            r=max((r for r in rows if r['t']<=t),key=lambda r:r['t'])
            selected.append(dict(r,t=float(t),native_body_source_t=r['t']))
        cache=ROOT.parent/'output/side_temporal164_20261101/cache_side_warm'
        prior_source='current_single_ring_spatial_phase_earlier_discovery_later_validation'
    else:
        rows=[json.loads(s) for s in (ROOT.parent/'output/layout176_20261107/integrated/train2_fixed.jsonl').read_text('utf-8').splitlines()]
        record=next(r for r in json.loads((ROOT.parent/'output/isolated177_20261107/mechanism.json').read_text('utf-8'))['records'] if abs(r['t']-43.6)<.01)
        ellipse=next(r for r in record['new'] if r.get('native_rim_refit'))
        seed=min(rows,key=lambda r:abs(r['t']-record['t']));seed_t=seed['t']
        prior=seed['local_temporal_phase']['frequency_hz']
        cache=ROOT.parent/'output/native_plane165_20261104/cache_train2_inside'
        selected=[r for r in rows if seed_t-1e-9<=r['t']<=seed_t+.101]
        prior_source='ambiguous_existing_local_pixel_temporal_phase'
    seed_body=np.asarray(seed['body']['center']);assert prior and prior>0;out=[]
    for solve_pose,direction in [(False,1),(False,-1),(True,1),(True,-1)]:
        tracker=HelixCore(ellipse,prior,direction=direction);branch=[];last=None;t0=None
        try:
            for i,row in enumerate(selected):
                t=row['t'];old=max((r for r in rows if r['t']<=t-.033333),key=lambda r:r['t']);motion=old['motion'];shift=motion.get('estimate')
                if shift is None or shift.get('peak_at_search_boundary') or not row['motion']['current_observation']:
                    branch.append(dict(t=t,skipped='missing_qualified_native_motion'));break
                velocity=np.array([shift['dx'],shift['dy']])/motion['dt_s'];e=window(cache,t)
                if last is not None:e=e[e['t']>=round(last*1e6)]
                e=transformed(e,ellipse,np.asarray(row['body']['center'])-seed_body,velocity,t)
                if not len(e):branch.append(dict(t=t,skipped='no_ring_events'));break
                if i==0:
                    t0=int(e['t'][0]);state=tracker.state();theta=phase_geometry(e,ellipse,state,direction)
                    residual=2*theta+2*state['pose_q'][1]-2*np.pi*prior*(e['t']-t0)/1e6
                    phi=float(np.angle(np.mean(np.exp(1j*residual))));warm_R=float(abs(np.mean(np.exp(1j*residual))))
                    tracker.close();tracker=HelixCore(ellipse,prior,phase_rad=phi,direction=direction)
                    heldout=None
                else:
                    state=tracker.state();theta=phase_geometry(e,ellipse,state,direction);local=(e['t']-t0)/1e6
                    predicted=state['phase0_rad']+2*np.pi*state['phase_hz']*local+np.pi*state['phase_acceleration_hz_s']*local*local-2*state['pose_q'][1]
                    residual=predicted-2*theta
                    heldout=dict(events=len(e),concentration=float(abs(np.mean(np.exp(1j*residual)))),
                        mean_cosine=float(np.mean(np.cos(residual))),
                        sectors=int(len(np.unique(np.floor((theta+np.pi)*12/(2*np.pi)).astype(int)%12))),
                        predicted_before_current_event_updates=True)
                begin=time.perf_counter();state=tracker.process(e,cutoff_s=t,solve_pose=solve_pose);elapsed=(time.perf_counter()-begin)*1000
                branch.append(dict(t=t,ring_events=len(e),elapsed_ms=elapsed,heldout=heldout,state=state,
                    velocity_source_t=old['t'],geometry_translation_source='current_native_body_no_coordinate_labels',
                    native_body_source_t=row.get('native_body_source_t',row['t']),
                    pose_refinement=solve_pose));last=t
        finally:tracker.close()
        out.append(dict(direction=direction,pose_refinement=solve_pose,phase_prior_hz=prior,prior_source=prior_source,
            prior_is_independent_RPM_truth=False,warm_phase_concentration=warm_R,records=branch))
    files=[ROOT/'native/helixtrack_bridge.cpp',ROOT/'helixtrack_adapter.py',Path('D:/drone forecast/helixtrack-main/helixtrack_tracker.h')]
    result=dict(seed_t=seed_t,ellipse=ellipse,branches=out,NN_used=False,coordinate_labels_used=False,
        author_GT_initialization_used=False,author_code_modified=False,force_feedback_enabled=False,
        mechanical_RPM_validated=False,forecast_improvement_claimed=False,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (OUT/('native_pilot_C.json' if reviewed_C else 'native_pilot.json')).write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps([dict(direction=b['direction'],pose=b['pose_refinement'],warm_R=b['warm_phase_concentration'],records=[dict(t=r['t'],hz=r.get('state',{}).get('phase_hz'),holdout=r.get('heldout')) for r in b['records']]) for b in out]))
