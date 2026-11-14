"""Matched old versus LM pose solver using one cached observation sequence."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import run_optimized_forecast as main
from lm153 import install,restore
main.install_shared_objective()
models={label:{n:main.PartialModel('affine3') for n in ['count','occupancy']} for label in ['old','lm']}
rows=[json.loads(x) for x in (ROOT.parent/'output/stage149_inertia_scale_20261024/clear/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
differences=[];gates=0;times={'old':[],'lm':[]};fallback=0;fits=0;worst=[];records=[]
for index,r in enumerate(rows):
    for n in ['count','occupancy']:
        rotors,_=main.bounded(r['geometry'][n].get('rotors',[]),1280,720);values={}
        for label in (['old','lm'] if index%2==0 else ['lm','old']):
            install() if label=='lm' else restore()
            model=models[label][n];start=time.perf_counter();baseline=main.baseline_for(model,r['t'],rotors,r['body'])
            _,d=model.step(r['t'],r['body'],rotors,dict(valid=False),baseline)
            times[label].append((time.perf_counter()-start)*1000);values[label]=d
            if label=='lm' and d.get('fixed_structure_used'):
                fits+=1;fallback+=int(d.get('measurement',{}).get('fit',{}).get('lm_fallback153',False))
        a,b=values['old'],values['lm'];gates+=int(a['current']!=b['current'])
        if a['current'] and b['current']:
            delta=float(np.linalg.norm(np.asarray(a['model_geometric_center'])-b['model_geometric_center']));differences.append(delta)
            if delta>.1:worst.append(dict(t=r['t'],source=n,center_difference=delta))
        records.append(dict(t=r['t'],source=n,old_current=a['current'],lm_current=b['current'],
            old_center=a.get('model_geometric_center'),lm_center=b.get('model_geometric_center'),
            old_rms=a.get('measurement',{}).get('fit',{}).get('rms_px'),
            lm_rms=b.get('measurement',{}).get('fit',{}).get('rms_px')))
    if index%100==99:print(json.dumps(dict(frames=index+1,gates_changed=gates)),flush=True)
restore();out=ROOT.parent/'output/stage153_LM_geometry_20261024';out.mkdir(exist_ok=True)
report=dict(frames=len(rows),gates_changed=gates,center_difference_median=float(np.median(differences)),
    center_difference_max=float(max(differences)),worst=worst,LM_fallbacks=fallback,fixed_calls=fits,
    old_median_ms=float(np.median(times['old'])),LM_median_ms=float(np.median(times['lm'])),
    old_total_s=sum(times['old'])/1000,LM_total_s=sum(times['lm'])/1000,observation_replay_only=True)
(out/'replay153.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(out/'paired_geometry.jsonl').write_text('\n'.join(json.dumps(x) for x in records)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)
