"""One fixed-input geometry profile; detector observations are frozen.

This is a component profile, never an end-to-end FPS claim. Native RAW remains
read-only. The same public SDK filter, counts and five-circle state are used.
"""
from pathlib import Path
import os,sys,json,cProfile,pstats,io,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import cv2
from fast_structure_front import FastStructure
from live_decode import native_batches
from sdk_activity_backend import SDKActivityAVX2

def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2)
    rows=[json.loads(s) for s in (ROOT.parent/'output/target25_continuity_20261031/RAW_clear_center4s/predictions.jsonl').read_text('utf-8').splitlines()][:60]
    model=FastStructure(optimized_compute=True,side_axis_budget=True,jit_counts=True,jit_conics=True)
    noise=SDKActivityAVX2(1280,720,10000);buffer=noise.get_empty_output_buffer();prof=cProfile.Profile();values=[]
    begin=(26000000//33333)*33333/1e6
    for i,(t,events) in enumerate(native_batches(Path('E:/dataset/train/6/media/event.raw'),begin,len(rows)*33333/1e6,False,33333)):
        r=rows[i];assert abs(r['t']-t)<1e-7,(r['t'],t)
        noise.process_events(events,buffer);selected=buffer.numpy()
        old=rows[i-1]['coefficient'] if i else None;v=np.zeros(2) if old is None else np.asarray(old['velocity'])
        prof.enable();s,rates,g,density=model.step(selected,t,r['body'],None,v,frequency_pixels_unique=True,defer_rates=True);prof.disable()
        values.append(dict(t=t,body=r['body'],structure=s,geometry=g))
    (out/'geometry.jsonl').write_text('\n'.join(json.dumps(r,default=lambda x:x.tolist() if hasattr(x,'tolist') else x) for r in values)+'\n','utf-8')
    prof.dump_stats(str(out/'geometry.prof'));stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumtime').print_stats(32)
    (out/'profile.txt').write_text(stream.getvalue(),'utf-8');print(stream.getvalue())
if __name__=='__main__':main(sys.argv[1])
