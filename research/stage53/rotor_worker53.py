"""Current-cutoff RAW10 rotor worker; separate OpenCV state from RAW20 body."""
from pathlib import Path
from collections import deque
import argparse,hashlib,json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage9'),str(ROOT/'research/stage33')]
from recovery import RecoveryFrontend
from slots33 import ReplaceStale,PaperConfig
from event_drone_forecast.raw_io import open_raw,raw_chunks

def json_text(x):return json.dumps(x,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
def fingerprints():
    paths=[Path(__file__),ROOT/'research/stage33/slots33.py',ROOT/'research/stage9/recovery.py',ROOT/'research/stage8/polarity.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def main(raw,output,start,duration):
    raw=Path(raw).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else None
            writing=isinstance(mode,str) and any(x in mode for x in ['w','a','x','+'])
            if not writing and p.is_relative_to(ROOT.parent/'output') and not p.is_relative_to(output) and p.suffix.lower() in ['.json','.jsonl','.png','.mp4']:
                raise RuntimeError('Rotor worker cache/truth input forbidden '+str(p))
            if p.is_relative_to(ROOT.parent/'output/raw_only_project/evaluation_data'):raise RuntimeError('Truth forbidden')
    sys.addaudithook(guard);cfg=PaperConfig();hashes=fingerprints();reader=open_raw(raw);h,w=reader.get_size();front=RecoveryFrontend(w,h,'neighbor10');sensor=ReplaceStale(cfg);history=deque();stream=raw_chunks(reader,start,duration,10000);n=0;last=start;times=[];requests=0;event_dtype=None
    print(json_text(dict(ready=True,width=w,height=h,step_us=10000,source_hashes=hashes,inputs='RAW only; no sensing/geometry cache')),flush=True)
    try:
        with (output/'rotor_ticks.jsonl').open('x',encoding='utf-8') as f:
            for line in sys.stdin:
                request=json.loads(line)
                if request.get('stop'):break
                target=float(request['cutoff']);assert request['id']==requests and target>last+1e-8 and round(target*1e6)%20000==0;requests+=1
                last_record=None;request_begin=time.perf_counter()
                while last<target-1e-8:
                    padding=False
                    try:t,events=next(stream);event_dtype=events.dtype
                    except StopIteration:
                        # A20ms final bucket may end10ms after the10ms reader.
                        # Supply no invented events, only that final empty slice.
                        assert reader.is_done() and event_dtype is not None and target-last<=.01000001
                        t=target;events=np.empty(0,dtype=event_dtype);padding=True
                    assert t<=target+1e-8 and (not len(events) or int(events['t'].max())<round(t*1e6))
                    tick=time.perf_counter();obs,_=front.detect(events,t)
                    stride=max(1,int(np.ceil(len(events)/max(1,cfg.max_buffer_events//6))));history.append((t,events[::stride].copy()))
                    while history and history[0][0]<t-cfg.frequency_window_us/1e6-.03:history.popleft()
                    sensed=sensor.step(obs,history,t);elapsed=(time.perf_counter()-tick)*1000
                    last_record=dict(t=t,event_count=len(events),observation=obs,sensing=sensed,slot_audit=sensor.audit,processing_ms=elapsed,eof_empty_padding=padding)
                    f.write(json_text(last_record)+'\n');last=t;n+=1;times.append(elapsed)
                assert last_record is not None and abs(last-target)<1e-8
                print(json_text(dict(id=request['id'],cutoff=last,record=last_record,request_wall_ms=(time.perf_counter()-request_begin)*1000)),flush=True)
    finally:
        summary=dict(native10_ticks=n,last_cutoff=last,processing_median_ms=float(np.median(times)) if times else None,source_hashes_unchanged=hashes==fingerprints(),labels_read=False)
        (output/'rotor_completion.json').write_text(json_text(summary),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,required=True);a=p.parse_args();main(a.raw,a.output,a.start,a.duration)
