"""Development contour replay timing, no RAW inference or accuracy claim."""
from pathlib import Path
import json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from curvature147 import split_contour
from probe147 import OUT,controls,js
controls(); saved=json.loads((OUT/'probe147_prototype.json').read_text()); result={}
for name,case in saved.items():
    timings=[]
    for repeat in range(5):
        tic=time.perf_counter()
        splits=[split_contour(np.array(p,np.int32).reshape(-1,1,2))[1] for p in case['contours']]
        timings.append((time.perf_counter()-tic)*1000)
    result[name]=dict(median_ms=float(np.median(timings)),splits=splits)
    print(js(dict(case=name,median_ms=result[name]['median_ms'],accepted=sum(d['accepted'] for d in splits))),flush=True)
(OUT/'bench147.json').write_text(js(result),encoding='utf-8')
