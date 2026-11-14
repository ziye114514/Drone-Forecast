"""Experimental RAW-only paired side-line/ellipse observations, shared15 models."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage60')]
import run_candidate_forecast as previous
from roi55 import SemanticPublisher

def pipeline_source():
    source=previous.pipeline_source();old="str(ROOT/'research/stage57/rotor_worker57.py')";assert source.count(old)==1
    return source.replace(old,"str(ROOT/'research/stage60/rotor_worker60.py')")

def build(line_mode,capacity_mib):
    assert line_mode in ['off','paired'];path=ROOT/'core/run_joint_forecast.py'
    namespace={'__name__':'paired60','__file__':str(path),'event_transport':'shared','event_capacity_bytes':capacity_mib*1024*1024,
       'Publisher':SemanticPublisher,'geometry_roi_mode':'off' if line_mode=='off' else 'body'}
    exec(compile(pipeline_source(),str(path),'exec'),namespace);namespace['METHODS']=['delayed_feedback'];original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),*[ROOT/'research/stage60'/n for n in ['rotor_worker60.py','frontend60.py','pairs60.py','profile60.py']]]:result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace['fingerprints']=fingerprints;return namespace

def run(raw,output,start=0.,duration=0.,precision=32,preview=False,verify_frames=False,line_mode='paired',event_capacity_mib=32):
    namespace=build(line_mode,event_capacity_mib)
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--line-mode',choices=['off','paired'],default='paired');p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,32,a.preview,a.verify_frames,a.line_mode)
