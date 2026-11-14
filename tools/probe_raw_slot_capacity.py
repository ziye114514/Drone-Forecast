"""Same native/filtered event fields and cutoffs; transport-only diagnostic.

Field hashing itself consumes CPU and can backpressure the producer. These
timings are NOT full-pipeline FPS or a clean isolated throughput benchmark.
"""
from pathlib import Path
import argparse,os,sys,json,hashlib,time,numpy as np
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from fast_raw_reader import ProcessRAW
OUT=ROOT.parent/'output/transport_stability_20261029'


def fields_hash(events):
    digest=hashlib.sha256()
    for name in ('x','y','p','t'):digest.update(events[name].tobytes())
    return digest.hexdigest()


def sample(capacity):
    step=33333;start=(108000000//step)*step/1e6
    reader=ProcessRAW(Path('E:/dataset/train/6/media/event.raw'),start,60*step/1e6,step,
        filter_events=True,avx2_filter=True,slot_events=capacity)
    answer=[];before=time.perf_counter()
    try:
        while True:
            began=time.perf_counter();packet=reader.read();read_ms=(time.perf_counter()-began)*1000
            if packet is None:break
            t,e,cost=packet;clean=reader.filtered_events;hash_started=time.perf_counter()
            answer.append(dict(t=t,raw_events=len(e),clean_events=len(clean),raw_fields_sha256=fields_hash(e),
                clean_fields_sha256=fields_hash(clean),consumer_wait_ms=read_ms,producer_service_ms=cost,
                producer_native_ms=reader.last_native_decode_ms,overflow=reader.packet_overflow))
            answer[-1]['hash_CPU_wall_ms']=(time.perf_counter()-hash_started)*1000
            packet=e=clean=None
    finally:reader.close()
    return dict(slot_events=capacity,allocated_shared_bytes=reader.allocated_shared_bytes,
        source_frames=len(answer),overflow_packets=reader.overflow_packets,wall_s=time.perf_counter()-before,
        steady_consumer_wait_total_ms=sum(r['consumer_wait_ms'] for r in answer[1:]),records=answer)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT/'slot_capacity')
    target=parser.parse_args().output.resolve();target.mkdir(parents=True,exist_ok=False)
    values={str(c):sample(c) for c in (2000000,3000000)}
    old,new=values.values();a,b=old['records'],new['records']
    keys=('t','raw_events','clean_events','raw_fields_sha256','clean_fields_sha256')
    exact=len(a)==len(b)==60 and all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b))
    assert exact,'Native/filtered sequence or timestamp windows changed'
    result=dict(native_and_filtered_fields_exact=exact,unspecified_struct_padding_not_compared=True,
        source_frames=60,capacity_only_changed=True,hashing_affects_timing=True,full_pipeline_FPS_measured=False,
        variants=values)
    (target/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in values.items()},indent=2))


if __name__=='__main__':main()
