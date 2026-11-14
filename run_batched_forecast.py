"""RAW-only full49 with exact streamed input and batched future projection."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage84')]
import run_stream_forecast as previous
from batched84 import OnlineExtras
def build():
    ns=previous.build();ns['OnlineExtras']=OnlineExtras;old=ns['fingerprints']
    def fingerprints():
        result=old()
        for p in [Path(__file__),ROOT/'research/stage84/batched84.py']:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    ns['fingerprints']=fingerprints
    return ns
def run(raw,output,start=0.,duration=0.,preview=False,verify_frames=False):
    ns=build()
    def receive(worker,kind):
        line=worker.stdout.readline()
        if not line:raise RuntimeError(kind+' worker stopped; inspect log')
        try:return json.loads(line)
        except json.JSONDecodeError:
            Path(output,'ipc_failure.json').write_text(json.dumps(dict(kind=kind,raw_line=line,process_poll=worker.poll()),ensure_ascii=False),encoding='utf-8');raise
    ns['receive']=receive
    try:
        result=ns['run'](raw,output,start,duration,32,preview,verify_frames)
        p=Path(output,'protocol.json');q=json.loads(p.read_text('utf-8'))
        q.update(exact_compute_deployment_stage=85,streamed_window_stage=82,batched_projection_stage=84,numerical_math_stage=78,
            native_sdk_frequency79_enabled=False,
            optimization='Exact80 forecast envelopes/background memo,82 streamed sampling,84 batched future projection. Full49.')
        p.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    finally:ns.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
