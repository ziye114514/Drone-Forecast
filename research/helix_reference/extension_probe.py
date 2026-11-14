"""Frozen mechanism: extend sign bank from original anchor to NEW following RAW.

Current conics and constant velocity at initialization only. Observation-only
author-core test, not a new NN/forecast run; no future boxes/labels/logs used.
"""
from pathlib import Path
import hashlib,json,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT)]
from reference_api import OUT
from sign_bank import SignBank
from live_decode import native_batches

parent=json.loads((OUT/'real_probe.json').read_text(encoding='utf-8'))
t0=parent['protocol']['anchor_t'];out=OUT/'causal_extension';out.mkdir(exist_ok=False)
frozen=Path(parent['protocol']['current_geometry_source'])
anchor=next(json.loads(s) for s in frozen.read_text(encoding='utf-8').splitlines() if abs(json.loads(s)['t']-t0)<1e-8)
rotors=anchor['geometry']['count']['rotors'];velocity=np.asarray(parent['protocol']['velocity_only_at_anchor'])
protocol=dict(scope='Observation-only real RAW extension, not end-to-end detection or trajectory scoring',
    raw=parent['protocol']['raw'],anchor_t=t0,duration_s=1.2,previous_exposed_s=.4,
    extension_is_unknown_recording=False,initial_conics_only=True,initial_velocity_only=True,
    no_future_boxes_or_labels=True,known_blade_count=False,assumed_B=2,
    quality_gate=.5,minimum_sectors=9,lock_current_ticks=3,ewma_s=.1,switch_margin=.1,
    frequency_disagreement_ratio=1.1,author_GN_interval_s=.04,parameters_frozen=True,
    author_core_DLL_sha256=hashlib.sha256((OUT/'build/Release/helix_reference.dll').read_bytes()).hexdigest())
source_dir=out/'source_snapshot';source_dir.mkdir()
for file in ['sign_bank.py','extension_probe.py','reference_api.py','helix_bridge.cpp']:
    shutil.copy2(Path(__file__).parent/file,source_dir/file)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
banks={i:SignBank(parent['seeds'][str(i)]['frequency_hz'],2) for i in range(4)}
records=[];wall=time.perf_counter()
try:
    for t,e in native_batches(protocol['raw'],t0,1.2):
        age=e['t']/1e6-t0;xy=np.c_[e['x'],e['y']]-age[:,None]*velocity;frame={};begin=time.perf_counter()
        for i,r in enumerate(rotors):
            a=np.deg2rad(r['angle']);u=xy-[r['cx'],r['cy']]
            rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            normalized=(u@rotation)/np.array([r['major']/2,r['minor']/2])
            radius=np.linalg.norm(normalized,axis=1);keep=(radius>=.2)&(radius<=1.5)
            patch=np.rint(128+80*normalized[keep]);ev=e[keep];good=np.all((patch>=0)&(patch<256),axis=1);patch=patch[good];ev=ev[good]
            packed=np.empty(len(ev),dtype=[('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')])
            packed['x']=patch[:,0];packed['y']=patch[:,1];packed['p']=ev['p'];packed['t']=ev['t']
            value=banks[i].feed(t,packed,solve_pose=round((t-t0)*50)%2==0)
            if value['selected']:
                H=np.asarray(value['selected']['H']);patch_center=H[:2,2]/H[2,2]
                center=np.array([r['cx'],r['cy']])+velocity*(t-t0)+rotation@((patch_center-128)*np.array([r['major']/2,r['minor']/2])/80)
                value['conditional_axis_image_xy']=center.tolist()
            frame[str(i)]=value
        records.append(dict(t=t,rotors=frame,native_events=len(e),tracking_ms=(time.perf_counter()-begin)*1000))
finally:
    for bank in banks.values():bank.close()
assert len(records)==60,len(records)
summary={}
for period,lo,hi in [('exposed',0,.4),('new_extension',.4,1.2)]:
    rows=[r for r in records if t0+lo+1e-8<r['t']<=t0+hi+1e-8]
    summary[period]=dict(ticks=len(rows),all_four_supported=sum(all(v['valid'] for v in r['rotors'].values()) for r in rows),rotors={})
    for i in range(4):
        series=[r['rotors'][str(i)] for r in rows];values=[r['selected'] for r in series if r['valid']]
        summary[period]['rotors'][str(i)]=dict(valid_ticks=len(values),
            median_phase_hz=float(np.median([v['phase_hz'] for v in values])) if values else None,
            median_weighted_R=float(np.median([v['weighted_phase_concentration'] for v in values])) if values else None,
            ambiguity_ticks=sum(r['reason']=='conflicting_phase_frequency_hypotheses' for r in series),
            direction_ambiguity_ticks=sum(r['direction_ambiguous'] for r in series))
result=dict(protocol=protocol,summary=summary,records=records,total_wall_s=time.perf_counter()-wall,
    tracking_ms_median=float(np.median([r['tracking_ms'] for r in records])),
    frequency_or_force_ground_truth=False,flight_prediction_improvement_proven=False)
(out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
