"""Same native event packets under two public SDK RAW read chunk sizes."""
from pathlib import Path
import sys,time,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_decode import native_batches


def collect(raw,start,duration,chunk):
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);clean=noise.get_empty_output_buffer()
    step=33333;begin=(round(start*1e6)//step)*step/1e6
    duration=(round(duration*1e6)//step)*step/1e6
    before=time.perf_counter();last=before;rows=[]
    for t,events in native_batches(raw,begin,duration,False,step,read_chunk=chunk):
        now=time.perf_counter();noise.process_events(events,clean)
        rows.append(dict(t=t,n=len(events),hash=hashlib.sha256(events.tobytes()).hexdigest(),
            clean_hash=hashlib.sha256(clean.numpy().tobytes()).hexdigest(),decode_ms=(now-last)*1000))
        last=time.perf_counter()
    return dict(frames=len(rows),rows=rows,total_s=time.perf_counter()-before,
        steady_decode_ms=float(np.mean([r['decode_ms'] for r in rows[1:]])),
        max_events=max(r['n'] for r in rows))


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);cases={}
    for name,start,duration,order in [('clear',26.,2.,[100000,1000000]),('burst',109.5,1.5,[1000000,100000])]:
        values={str(c):collect('E:/dataset/train/6/media/event.raw',start,duration,c) for c in order}
        a,b=values['100000']['rows'],values['1000000']['rows'];assert len(a)==len(b)
        for left,right in zip(a,b):
            for key in ['t','n','hash','clean_hash']:assert left[key]==right[key],(name,key,left,right)
        cases[name]=dict(chunks=values,exact_native_and_filtered_bytes_and_time_windows=True)
    result=dict(cases=cases,component_only_not_full_pipeline_FPS=True,one_paired_trial_each=True,
        reverse_order_on_second_case=True,source_event_sampling_or_dropping=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:{c:{q:v[q] for q in ['frames','total_s','steady_decode_ms','max_events']} for c,v in d['chunks'].items()} for k,d in cases.items()}))


if __name__=='__main__':main(sys.argv[1])
