"""Bounded current-window profile; locate costs, not an accuracy/FPS benchmark."""
from pathlib import Path
import os,sys,json,cProfile,pstats,io
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from fast_structure_front import FastStructure


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);prof=cProfile.Profile();cv2.setNumThreads(2)
    source=ROOT.parent/'output/predictive_harmonics_20261027/native_pilot'
    dtype=np.dtype([('x','u2'),('y','u2'),('t','i8'),('frequency','f4')]);samples=[]
    for name,folder in [('clear','raw_release_clear'),('partial','raw_release_clear'),('side','raw_release_side')]:
        p=np.load(source/name/'native_window.npz');t=float(p['t']);e=p['events']
        rows=[json.loads(s) for s in (ROOT.parent/'output/three_continuation_20261026'/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        row=min(rows,key=lambda r:abs(r['t']-t));model=FastStructure()
        prof.enable();model.step(e,t,row['body'],np.zeros(0,dtype),row['coefficient']['velocity'],frequency_pixels_unique=True);prof.disable()
    prof.dump_stats(str(out/'geometry.pstats'));buf=io.StringIO();pstats.Stats(prof,stream=buf).sort_stats('cumulative').print_stats(23)
    (out/'profile.txt').write_text(buf.getvalue(),'utf-8');print(buf.getvalue())
    inventory=[]
    for directory in (ROOT.parent/'output').glob('*label*'):
        p=directory/'labels.json'
        if not p.is_file():continue
        data=json.loads(p.read_text('utf-8'));records=data.get('records',[])
        if isinstance(records,dict):records=list(records.values())
        times=[float(r['t']) for r in records if isinstance(r,dict) and 't' in r]
        inventory.append(dict(name=directory.name,count=len(records),min_t=min(times,default=None),max_t=max(times,default=None),
            first_record=records[:1]))
    (out/'labels_inventory.json').write_text(json.dumps(inventory,indent=2),'utf-8');print(json.dumps(inventory))


if __name__=='__main__':main(sys.argv[1])
