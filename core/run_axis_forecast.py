"""Experimental RAW-only forecasts with independently admitted geometry axis."""
from pathlib import Path
import argparse, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage62')]
import run_stable_paired_forecast as previous
from independent62 import Managed62, METHODS as ADMISSIONS
from roi55 import SemanticPublisher
import runtime_full61


def pipeline_source():
    source=previous.pipeline_source()
    substitutions={"('adaptive_axis_freq',(190,80,210))":"('adaptive_axis_only',(190,80,210))",
        'purple=axis+frequency':'purple=axis-only',
        'q["feature_diagnostic"]["active"]':'q["feature_diagnostic"]["models"]["adaptive_axis_only"]["active"]',
        'stats["delayed_feedback"]["actual_updates"]["adaptive_axis_freq"]':'stats["delayed_feedback"]["actual_updates"]["adaptive_axis_only"]'}
    for old,new in substitutions.items():
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new)
    return source


def build(axis_mode,capacity_mib):
    assert axis_mode in ADMISSIONS;runtime_full61.verify()
    path=ROOT/'core/run_joint_forecast.py'
    namespace={'__name__':'axis62','__file__':str(path),'event_transport':'shared',
        'event_capacity_bytes':capacity_mib*1024*1024,'Publisher':SemanticPublisher,'geometry_roi_mode':'off'}
    exec(compile(pipeline_source(),str(path),'exec'),namespace)
    namespace['METHODS']=['delayed_feedback'];namespace['Managed52']=lambda:Managed62(axis_mode)
    original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),ROOT/'core/run_stable_paired_forecast.py',ROOT/'core/run_paired_forecast.py',
                  ROOT/'research/stage62/independent62.py',*[ROOT/'research/stage61'/n for n in ['preline61.py','runtime_full61.py','rotor_worker61.py']]]:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace['fingerprints']=fingerprints
    return namespace


def run(raw,output,start=0.,duration=0.,precision=32,preview=False,verify_frames=False,axis_mode='axis_learn',event_capacity_mib=32):
    assert 1<=event_capacity_mib<=256
    namespace=build(axis_mode,event_capacity_mib)
    def receive(worker,kind):
        line=worker.stdout.readline()
        if not line:raise RuntimeError(kind+' worker stopped; inspect log')
        try:return json.loads(line)
        except json.JSONDecodeError:
            Path(output,'ipc_failure.json').write_text(json.dumps(dict(kind=kind,raw_line=line,process_poll=worker.poll()),ensure_ascii=False),encoding='utf-8');raise
    namespace['receive']=receive
    try:return namespace['run'](raw,output,start,duration,precision,preview,verify_frames)
    finally:namespace.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--axis-mode',choices=ADMISSIONS,default='axis_learn')
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,32,a.preview,a.verify_frames,a.axis_mode)
