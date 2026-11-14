"""Instrument a short existing CLI run's decoder thread; no SDK patch."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,cProfile,pstats,io,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
import live_decode
original=live_decode.native_batches
OUT=ROOT.parent/'output/read249_20261108';OUT.mkdir(exist_ok=False)
records=[]

def traced(*args,**kwargs):
    iterator=iter(original(*args,**kwargs))
    while True:
        p=cProfile.Profile();tick=time.perf_counter();p.enable()
        try:packet=next(iterator,None)
        finally:p.disable()
        ms=(time.perf_counter()-tick)*1000
        if packet is None:break
        t,e=packet
        if ms>100.:
            buffer=io.StringIO();pstats.Stats(p,stream=buffer).sort_stats('cumtime').print_stats(35)
            file=f'read_{t:.6f}.txt';(OUT/file).write_text(buffer.getvalue(),'utf-8')
            records.append(dict(t=t,wall_ms=ms,events=len(e),profile=file))
            (OUT/'slow_reads.json').write_text(json.dumps(records,indent=2),'utf-8')
        yield packet

live_decode.native_batches=traced
from run_rotor_candidate import main
if __name__=='__main__':
    (OUT/'instrumentation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        SDK_modified=False,decoder_events_or_timestamps_changed=False,only_decoder_thread_profiled=True,
        measurement_overhead_present=True,not_a_speed_certification=True),indent=2),'utf-8')
    main()
