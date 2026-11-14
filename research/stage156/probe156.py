"""One same-input replay for parallel scheduling, not additional precision data."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
import run_optimized_forecast as main
from parallel156 import fronts,Pool,one
main.install_shared_objective();sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage153'))
from lm153 import install
install()
OUT=ROOT.parent/'output/stage156_parallel_geometry_20261024';OUT.mkdir(exist_ok=True)
saved=[json.loads(x) for x in (ROOT.parent/'output/stage149_inertia_scale_20261024/clear/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
rows={round(r['t'],6):r for r in saved};reader=main.open_raw('E:/dataset/train/6/media/event.raw')
serial=main.tile_fronts(1280,720);parallel=fronts(1280,720)
models={k:{name:main.PartialModel('affine3') for name in serial} for k in ['serial','parallel']}
pool=Pool();timing={'serial':[],'parallel':[]};differences=[];mismatches=[];count=0
for t,e in main.raw_chunks(reader,22.,2.,20000):
    r=rows[round(t,6)];prepared=main.counts(e,1280,720);prepared['on']/=2;prepared['off']/=2
    data={}
    for label in (['serial','parallel'] if count%2==0 else ['parallel','serial']):
        fs=serial if label=='serial' else parallel;fs['occupancy'].set_hint(e,r['body'],t);tick=time.perf_counter()
        data[label]=({name:one(f,models[label][name],e,t,prepared,r['body'],1280,720,True) for name,f in fs.items()}
            if label=='serial' else pool.step(fs,models[label],e,t,prepared,r['body'],1280,720,True))
        timing[label].append((time.perf_counter()-tick)*1000)
    for name in serial:
        a,b=data['serial'][name],data['parallel'][name]
        if a['observation']!=b['observation']:mismatches.append(dict(t=t,source=name,field='observation'))
        da,db=a['diagnostic'],b['diagnostic']
        if da['current']!=db['current']:mismatches.append(dict(t=t,source=name,field='current'))
        if da['current'] and db['current']:differences.append(float(np.max(abs(np.asarray(da['model_geometric_center'])-db['model_geometric_center']))))
    count+=1
pool.close();result=dict(frames=count,mismatches=mismatches,maximum_center_difference=max(differences,default=0.),
    serial_median_ms=float(np.median(timing['serial'])),parallel_median_ms=float(np.median(timing['parallel'])),
    serial_total_s=sum(timing['serial'])/1000,parallel_total_s=sum(timing['parallel'])/1000,
    cached_boxes_for_diagnostic_only=True)
(OUT/'probe156.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
