"""Experimental RAW-only force-increment forecast; all previous26 retained."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage73')]
import run_event_force_forecast as previous
from online73 import OnlineExtras

def build():
    ns=previous.build();ns['OnlineExtras']=OnlineExtras
    old=ns['fingerprints']
    def fingerprints():
        result=old()
        for p in [Path(__file__),ROOT/'research/stage73/online73.py',ROOT/'research/stage72/increment72.py']:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    ns['fingerprints']=fingerprints
    # Preview the actual new frequency-increment branch, not the old model.
    source=previous.pipeline_source()
    source=source.replace("('force_periodic',(190,80,210))","('increment_periodic',(190,80,210))")
    source=source.replace('purple=local-periodic-force','purple=force-increment')
    source=source.replace('q["event_force_diagnostic"]["models"]["force_periodic"]["active"]','q["increment_diagnostic"]["models"]["increment_periodic"]["active"]')
    source=source.replace('q["event_force_diagnostic"]["models"]["force_periodic"]["updates"]','q["increment_diagnostic"]["models"]["increment_periodic"]["updates"]')
    source=source.replace('event_force_stage=71','event_force_stage=73').replace('additional_forecasts=11','additional_forecasts=18')
    managed_factory=ns['Managed52'];old_run_globals=ns
    exec(compile(source,str(ROOT/'run_joint_forecast.py'),'exec'),old_run_globals)
    # Source exec defines the original fingerprint function; restore wrapper.
    ns['fingerprints']=fingerprints;ns['METHODS']=['delayed_feedback'];ns['OnlineExtras']=OnlineExtras;ns['Managed52']=managed_factory
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
    try:return ns['run'](raw,output,start,duration,32,preview,verify_frames)
    finally:ns.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
