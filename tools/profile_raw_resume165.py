"""Read-only SDK pause/resume timing, isolate the native decoder from CUDA."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,json,cProfile,pstats,io
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches

if __name__=='__main__':
    raw,out=sys.argv[1:];records=[]
    source=iter(native_batches(Path(raw),110,1,False,33333,reader_backend='iterator'))
    profile=cProfile.Profile()
    try:
        for i in range(12):
            tick=time.perf_counter();cpu=time.process_time();profile.enable()
            t,e=next(source);profile.disable()
            record=dict(index=i,t=t,events=len(e),ms=(time.perf_counter()-tick)*1000,cpu_ms=(time.process_time()-cpu)*1000)
            records.append(record);print(json.dumps(record),flush=True)
            if i==0:time.sleep(7.)
    finally:source.close()
    output=Path(out);output.write_text(json.dumps(records,indent=2),'utf-8')
    stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(20)
    output.with_suffix('.profile.txt').write_text(stream.getvalue(),'utf-8');print(stream.getvalue())
