"""RAW-only complete forecast with past-body candidate gating, not event clipping."""
from pathlib import Path
import argparse, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage57'),str(ROOT/'research/stage55')]
import run_roi_forecast as previous
from roi55 import SemanticPublisher

def pipeline_source():
    source=previous.pipeline_source();old="str(ROOT/'research/stage55/rotor_worker55.py')";assert source.count(old)==1
    return source.replace(old,"str(ROOT/'research/stage57/rotor_worker57.py')")

def build(mode,capacity_mib):
    assert mode in ['off','body']
    path=ROOT/'core/run_joint_forecast.py';namespace={'__name__':'candidate57','__file__':str(path),'event_transport':'shared',
        'event_capacity_bytes':capacity_mib*1024*1024,'Publisher':SemanticPublisher,'geometry_roi_mode':mode}
    exec(compile(pipeline_source(),str(path),'exec'),namespace);namespace['METHODS']=['delayed_feedback'];original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),ROOT/'core/run_roi_forecast.py',ROOT/'core/run_deploy_forecast.py',ROOT/'research/stage57/gating57.py',ROOT/'research/stage57/rotor_worker57.py',ROOT/'research/stage55/roi55.py',ROOT/'research/stage55/rotor_worker55.py']:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace['fingerprints']=fingerprints
    return namespace

def run(raw,output,start=0.,duration=0.,precision=32,preview=False,verify_frames=False,geometry_roi='body',event_capacity_mib=32):
    assert 1<=event_capacity_mib<=256
    namespace=build(geometry_roi,event_capacity_mib)
    def receive(worker,kind):
        line=worker.stdout.readline()
        if not line:raise RuntimeError(kind+' worker stopped; inspect its log')
        try:return json.loads(line)
        except json.JSONDecodeError:
            Path(output,'ipc_failure.json').write_text(json.dumps(dict(kind=kind,raw_line=line,process_poll=worker.poll()),ensure_ascii=False),encoding='utf-8')
            raise
    namespace['receive']=receive
    try:return namespace['run'](raw,output,start,duration,precision,preview,verify_frames)
    finally:namespace.clear()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--precision',type=int,choices=[16,32],default=32);p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true');p.add_argument('--geometry-roi',choices=['off','body'],default='body');p.add_argument('--event-capacity-mib',type=int,default=32)
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.precision,a.preview,a.verify_frames,a.geometry_roi,a.event_capacity_mib)
