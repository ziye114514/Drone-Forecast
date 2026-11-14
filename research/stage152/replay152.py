"""One matched geometry replay; old observations are diagnostic input only."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import run_optimized_forecast as main
main.install_shared_objective()
from warm152 import install,restore
old={n:main.PartialModel('affine3') for n in ['count','occupancy']}
install();new={n:main.PartialModel('affine3') for n in old};restore()
rows=[json.loads(x) for x in (ROOT.parent/'output/stage149_inertia_scale_20261024/clear/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
differences=[];gates=0;times={'old':[],'warm':[]};warm=0;fits=0;worst=[]
for index,r in enumerate(rows):
    for n in old:
        rotors,_=main.bounded(r['geometry'][n].get('rotors',[]),1280,720);values={}
        # Alternate order so the first solver is not always the cold CPU call.
        for label in (['old','warm'] if index%2==0 else ['warm','old']):
            model=old[n] if label=='old' else new[n]
            start=time.perf_counter();baseline=main.baseline_for(model,r['t'],rotors,r['body'])
            _,d=model.step(r['t'],r['body'],rotors,dict(valid=False),baseline)
            times[label].append((time.perf_counter()-start)*1000);values[label]=d
            if label=='warm' and d.get('fixed_structure_used'):
                fits+=1;warm+=int(d.get('measurement',{}).get('fit',{}).get('warm_start152',False))
        a,b=values['old'],values['warm'];gates+=int(a['current']!=b['current'])
        if a['current'] and b['current']:
            delta=float(np.linalg.norm(np.asarray(a['model_geometric_center'])-b['model_geometric_center']))
            differences.append(delta)
            if delta>.1:worst.append(dict(t=r['t'],source=n,center_difference=delta))
    if index%100==99:print(json.dumps(dict(frames=index+1,warm=warm,gates_changed=gates)),flush=True)
out=ROOT.parent/'output/stage152_warm_geometry_20261024';out.mkdir(exist_ok=True)
report=dict(frames=len(rows),matched_geometry_calls=len(times['old']),warm_selected=warm,
    fixed_calls=fits,gates_changed=gates,center_difference_median=float(np.median(differences)),
    center_difference_max=float(max(differences)),worst=worst,
    old_median_ms=float(np.median(times['old'])),warm_median_ms=float(np.median(times['warm'])),
    old_total_s=sum(times['old'])/1000,warm_total_s=sum(times['warm'])/1000,
    observation_replay_only=True,not_new_RAW_or_NN=True)
(out/'replay152.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
