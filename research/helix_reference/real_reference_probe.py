"""Short real-RAW author-core probe, initialized from saved CURRENT geometry.

This is an observation probe, not a new RAW-only NN/end-to-end run. No future
geometry, future boxes, manual labels or flight logs are used for estimation.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'src'),str(ROOT/'research/stage79')]
from reference_api import AuthorTracker,OUT
from live_decode import native_batches
import native79

frozen=ROOT.parent/'output/stage156_parallel_geometry_20261024/clear/predictions.jsonl'
anchor=None
for s in frozen.read_text(encoding='utf-8').splitlines():
    row=json.loads(s);rotors=row['geometry']['count'].get('rotors',[])
    if row['models']['count'].get('current') and len(rotors)==4 and all(r.get('mode')=='ellipse' and not r.get('inferred_shape') for r in rotors):
        anchor=row;break
assert anchor is not None
t0=anchor['t'];rotors=anchor['geometry']['count']['rotors'];velocity=np.asarray(anchor['coefficients']['matched_body']['velocity'])
engine=native79.engine(1280,720,'strict7');buf=engine.get_empty_output_buffer()
trackers={};seeds={};records=[];began=time.perf_counter()
protocol=dict(scope='Real RAW observation probe; frozen current conics/velocity initialize ellipse rectification. Not a new NN/end-to-end run.',
    raw='E:/dataset/train/6/media/event.raw',anchor_t=t0,duration_s=.4,
    seed_from='SDK strict7 per-pixel frequency mode within each current ellipse bounding box; still harmonic ambiguous',
    B_assumed=2,rotation_rate_is_not_ground_truth=True,current_geometry_source=str(frozen),
    source_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),velocity_only_at_anchor=velocity.tolist(),
    concentration_gate=.5,minimum_azimuth_sectors=9,thresholds_frozen_before_RAW=True,
    author_core_pose_solve_every_s=.04,author_config='config_tracker.yaml defaults, heatmap display disabled')
(OUT/'real_probe_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
try:
    for t,e in native_batches(protocol['raw'],round(t0-.12,2),.52):
        q=native79.checked_process(engine,e,buf,1280,720)
        if t<=t0+1e-8:
            if abs(t-t0)<1e-7:
                for i,r in enumerate(rotors):
                    a,b=r['major']/2,r['minor']/2
                    bbox=[r['cx']-a,r['cy']-a,r['cx']+a,r['cy']+a]
                    result=native79.summarize(q,bbox,t,1280,720);seeds[str(i)]=result
                    if result.get('frequency_hz'):
                        trackers[i]=AuthorTracker(result['frequency_hz']*60,2)
            continue
        frame={}
        age=e['t']/1e6-t0
        xy=np.c_[e['x'],e['y']]-age[:,None]*velocity
        for i,tracker in trackers.items():
            r=rotors[i];angle=np.deg2rad(r['angle']);u=xy-[r['cx'],r['cy']]
            normalized=np.c_[u[:,0]*np.cos(angle)+u[:,1]*np.sin(angle),-u[:,0]*np.sin(angle)+u[:,1]*np.cos(angle)]
            normalized/=np.array([r['major']/2,r['minor']/2])
            radius=np.linalg.norm(normalized,axis=1);keep=(radius>=.2)&(radius<=1.5)
            patch=np.rint(128+80*normalized[keep]);ev=e[keep]
            good=np.all((patch>=0)&(patch<256),axis=1);patch=patch[good];ev=ev[good]
            solve=round((t-t0)*50)%2==0
            value=tracker.feed(patch[:,0],patch[:,1],ev['t'],ev['p'],solve_pose=solve)
            value['conditional_phase_supported']=value['predictive_phase_concentration']>=.5 and value['azimuth_sectors']>=9
            frame[str(i)]=value
        records.append(dict(t=t,rotors=frame,native_events=len(e)))
finally:
    for tracker in trackers.values():tracker.close()
assert len(records)==20,len(records)
summary={}
for i in trackers:
    series=[row['rotors'][str(i)] for row in records]
    summary[str(i)]=dict(initial_seed_hz=seeds[str(i)]['frequency_hz'],initial_pixel_period_qualified=seeds[str(i)]['qualified'],
        median_phase_hz=float(np.median([r['phase_hz'] for r in series])),
        median_phase_concentration=float(np.median([r['predictive_phase_concentration'] for r in series])),
        supported_ticks=sum(r['conditional_phase_supported'] for r in series),ticks=len(series),
        rotation_signs=sorted(set(r['rotation_sign'] for r in series)))
report=dict(protocol=protocol,seeds=seeds,summary=summary,records=records,wall_s=time.perf_counter()-began,
            true_mechanical_RPM_available=False,force_or_forecast_gain_proven=False)
(OUT/'real_probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(anchor=t0,summary=summary,wall_s=report['wall_s'])),flush=True)
