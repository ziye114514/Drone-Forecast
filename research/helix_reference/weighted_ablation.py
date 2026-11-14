"""Same RAW/event coordinates: fixed ellipse vs author's GN pose refinement."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT)]
from reference_api import AuthorTracker,OUT
from live_decode import native_batches
from metavision_sdk_core import BaseFrameGenerationAlgorithm
import cv2

PARENT=json.loads((OUT/'real_probe.json').read_text(encoding='utf-8'))
t0=PARENT['protocol']['anchor_t'];out=OUT/'weighted_ablation';out.mkdir(exist_ok=False)
frozen=Path(PARENT['protocol']['current_geometry_source'])
anchor=next(json.loads(s) for s in frozen.read_text(encoding='utf-8').splitlines() if abs(json.loads(s)['t']-t0)<1e-8)
rotors=anchor['geometry']['count']['rotors'];velocity=np.asarray(PARENT['protocol']['velocity_only_at_anchor'])
trackers={(i,mode):AuthorTracker(PARENT['seeds'][str(i)]['frequency_hz']*60,2) for i in range(4) for mode in ['fixed','refine']}
streams={i:[] for i in range(4)};records=[];max_delta=0.;first_image=None
try:
    for t,e in native_batches(PARENT['protocol']['raw'],round(t0-.02,2),.42):
        if t<=t0+1e-8:
            first_image=np.zeros((720,1280,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(e,first_image)
            continue
        age=e['t']/1e6-t0;xy=np.c_[e['x'],e['y']]-age[:,None]*velocity;frame={}
        for i,r in enumerate(rotors):
            a=np.deg2rad(r['angle']);u=xy-[r['cx'],r['cy']]
            normalized=np.c_[u[:,0]*np.cos(a)+u[:,1]*np.sin(a),-u[:,0]*np.sin(a)+u[:,1]*np.cos(a)]/np.array([r['major']/2,r['minor']/2])
            radius=np.linalg.norm(normalized,axis=1);keep=(radius>=.2)&(radius<=1.5)
            patch=np.rint(128+80*normalized[keep]);ev=e[keep];good=np.all((patch>=0)&(patch<256),axis=1);patch=patch[good];ev=ev[good]
            packed=np.empty(len(ev),dtype=[('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')])
            packed['x']=patch[:,0];packed['y']=patch[:,1];packed['p']=ev['p'];packed['t']=ev['t'];streams[i].append(packed)
            for mode in ['fixed','refine']:
                value=trackers[i,mode].feed(patch[:,0],patch[:,1],ev['t'],ev['p'],solve_pose=mode=='refine' and round((t-t0)*50)%2==0)
                frame[f'{i}:{mode}']=value
                if mode=='refine':
                    old=PARENT['records'][len(records)]['rotors'][str(i)]
                    max_delta=max(max_delta,abs(value['phase_hz']-old['phase_hz']))
        records.append(dict(t=t,results=frame))
finally:
    for tracker in trackers.values():tracker.close()
assert len(records)==20 and max_delta<1e-4,(len(records),max_delta)
np.savez_compressed(out/'rectified_events.npz',**{f'rotor{i}_tick{k}':events for i,rows in streams.items() for k,events in enumerate(rows)})
summary={}
for i in range(4):
    for mode in ['fixed','refine']:
        series=[row['results'][f'{i}:{mode}'] for row in records]
        summary[f'{i}:{mode}']={key:float(np.median([r[key] for r in series])) for key in ['phase_hz','predictive_phase_concentration','weighted_phase_concentration','polarity_split_weighted_concentration']}
        summary[f'{i}:{mode}']['raw_signed_omega_median']=float(np.median([r['angular_velocity_rad_s'] for r in series]))
        summary[f'{i}:{mode}']['informative_ticks']=sum(r['weighted_phase_concentration']>=.5 and r['informative_ring_sectors']>=9 for r in series)
for i,r in enumerate(rotors):
    color=(0,210,0) if summary[f'{i}:refine']['weighted_phase_concentration']>=.5 else (0,100,255)
    cv2.ellipse(first_image,((r['cx'],r['cy']),(r['major'],r['minor']),r['angle']),color,2)
    cv2.putText(first_image,f"{i}: weighted phase R={summary[f'{i}:refine']['weighted_phase_concentration']:.2f}",(round(r['cx'])-65,max(20,round(r['cy'])-20)),0,.4,color,1)
cv2.putText(first_image,f'Init t={t0:.2f}s; labels summarize following0.4s, not RPM ground truth',(15,700),0,.6,(0,200,255),1)
cv2.imwrite(str(out/'initial_rotors_with_phase_evidence.png'),first_image)
(out/'results.json').write_text(json.dumps(dict(scope='Same real RAW and initial ellipse rectification; fixed vs author pose GN, pre-update coherence.',
    unweighted_author_baseline_phase_max_change=max_delta,summary=summary,records=records,mechanical_RPM_truth=False),indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
