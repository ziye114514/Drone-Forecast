"""Native public-reader comparison, unchanged native events; no GPU/model."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,cProfile,pstats,io,hashlib,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reverse',action='store_true');parser.add_argument('--cooperative',action='store_true');args=parser.parse_args()
    OUT=ROOT.parent/f'output/{"decode251" if args.cooperative else "decode247_reverse" if args.reverse else "decode247"}_20261108';OUT.mkdir(exist_ok=False);results={}
    backends=['iterator','cooperative'] if args.cooperative else ['iterator','rolling'] if args.reverse else ['rolling','iterator']
    for backend in backends:
        iterator=iter(native_batches(Path('E:/dataset/experiment/13/media/event.raw'),10.,6.,False,33333,reader_backend=backend))
        rows=[];profile=cProfile.Profile();profile.enable();tick=time.perf_counter()
        while True:
            start=time.perf_counter();cpu=time.thread_time();packet=next(iterator,None);wall=(time.perf_counter()-start)*1000;cpu=(time.thread_time()-cpu)*1000
            if packet is None:break
            t,e=packet;canonical=np.empty(len(e),dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')])
            for field in canonical.dtype.names:canonical[field]=e[field]
            rows.append(dict(t=t,events=len(e),wall_ms=wall,cpu_ms=cpu,native_fields_sha256=hashlib.sha256(canonical.tobytes()).hexdigest()))
        profile.disable();buffer=io.StringIO();pstats.Stats(profile,stream=buffer).sort_stats('cumtime').print_stats(25)
        (OUT/f'{backend}_profile.txt').write_text(buffer.getvalue(),'utf-8')
        result=dict(frames=len(rows),events=sum(r['events'] for r in rows),rows=rows,wall_s=time.perf_counter()-tick,
            read_wall_ms_after_seek_mean=float(np.mean([r['wall_ms'] for r in rows[1:]])),
            read_wall_ms_after_seek_max=float(max(r['wall_ms'] for r in rows[1:])),
            read_CPU_ms_after_seek_max=float(max(r['cpu_ms'] for r in rows[1:])))
        results[backend]=result;print(backend,{k:v for k,v in result.items() if k!='rows'},flush=True)
        (OUT/'partial.json').write_text(json.dumps(results,indent=2),'utf-8')
    aa=results[backends[0]]['rows'];bb=results[backends[1]]['rows']
    same=len(aa)==len(bb) and all(a['t']==b['t'] and a['events']==b['events'] and a['native_fields_sha256']==b['native_fields_sha256'] for a,b in zip(aa,bb))
    (OUT/'results.json').write_text(json.dumps(dict(results=results,all_native_fields_and_cutoffs_identical=same,
        GPU_used=False,RAW_pipeline_FPS_measured=False,SDK_modified=False),indent=2),'utf-8');print('same_native_sequence',same,flush=True)
