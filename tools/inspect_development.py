"""Development-only diagnostic snapshots; never selects held-out thresholds."""
from pathlib import Path
import sys,json
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.raw_io import open_raw
from event_drone_forecast.perception import RotorPerception
from event_drone_forecast.config import Config

out=Path(__file__).resolve().parents[2]/'output/raw_only_project/development_snapshots'
out.mkdir(exist_ok=True,parents=True)
reader=open_raw('E:/dataset/train/6/media/event.raw')
times=[17.,25.,35.,40.8,100.8,104.8,110.,112.]
records=[];tiles=[]
for t in times:
    path=out/f'{t:.1f}.npy'
    if path.exists():e=np.load(path)
    else:
        reader.seek_time(int((t-.02)*1e6));e=reader.load_delta_t(20000).copy();np.save(path,e)
    row=[]
    for b in (4,2,1):
        cfg=Config(spatial_bin=b,min_blob_area=7 if b==4 else 12)
        p=RotorPerception(1280,720,cfg);r,total=p.detect(e,t)
        image=cv2.cvtColor(np.uint8(np.clip(np.log1p(total)/4,0,1)*255),cv2.COLOR_GRAY2BGR)
        for q in r['rotors']:cv2.ellipse(image,((q['cx'],q['cy']),(q['major'],q['minor']),q['angle']),(0,255,0),2)
        cv2.putText(image,f'train6 {t:.1f}s bin={b} N={len(r["rotors"])}',(10,35),0,.9,(255,255,255),2)
        row.append(cv2.resize(image,(640,360)))
        records.append(dict(t=t,bin=b,detected=r['detected'],rotors=len(r['rotors']),center=r.get('center')))
    tiles.append(np.hstack(row))
cv2.imwrite(str(out/'resolution_comparison.png'),np.vstack(tiles))
(out/'diagnostics.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records))
