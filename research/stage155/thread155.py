"""Small same-input OpenCV threading probe; no forecast or label tuning."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
import run_optimized_forecast as main
import cv2,numpy as np
main.install_shared_objective();OUT=ROOT.parent/'output/stage155_frontend_threads_20261024';OUT.mkdir(exist_ok=True)
saved=cv2.getNumThreads();report={}
for name,t in [('clear',26.84),('side',118.12)]:
    rows=[json.loads(s) for s in (ROOT.parent/'output/stage149_inertia_scale_20261024'/name/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    row=min(rows,key=lambda r:abs(r['t']-t));reader=main.open_raw('E:/dataset/train/6/media/event.raw')
    _,e=next(iter(main.raw_chunks(reader,t-.02,.02,20000)));prepared=main.counts(e,1280,720)
    prepared['on']/=2;prepared['off']/=2
    durations={1:[],saved:[]};digests={1:[],saved:[]}
    for repeat in range(4):
        for threads in ([saved,1] if repeat%2==0 else [1,saved]):
            cv2.setNumThreads(threads);fronts=main.tile_fronts(1280,720)
            fronts['occupancy'].set_hint(e,row['body'],t);tick=time.perf_counter();observed={}
            for source,front in fronts.items():observed[source]=front.detect_prepared(e,t,prepared)[0]['rotors']
            durations[threads].append((time.perf_counter()-tick)*1000)
            digests[threads].append(hashlib.sha256(main.js(observed).encode()).hexdigest())
    report[name]=dict(default_threads=saved,default_median_ms=float(np.median(durations[saved])),
        one_thread_median_ms=float(np.median(durations[1])),rotors_exact=set(digests[1])==set(digests[saved]),
        all_times_ms=durations)
cv2.setNumThreads(saved);(OUT/'thread_probe155.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
