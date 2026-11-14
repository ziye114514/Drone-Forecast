"""Experimental RAW-only55: original49 plus fixed complementary history candidates."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage93')]
import run_batched_forecast as previous
from online93 import OnlineExtras


def build():
    ns=previous.build();ns['OnlineExtras']=OnlineExtras;old=ns['fingerprints']
    def fingerprints():
        result=old()
        for p in [Path(__file__),ROOT/'research/stage93/online93.py',ROOT/'research/stage92/router92.py']:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    ns['fingerprints']=fingerprints;return ns


def run(raw,output,start=0.,duration=0.,preview=False,verify_frames=False):
    ns=build()
    def receive(worker,kind):
        line=worker.stdout.readline()
        if not line:raise RuntimeError(kind+' worker stopped; inspect log')
        return json.loads(line)
    ns['receive']=receive
    try:
        result=ns['run'](raw,output,start,duration,32,preview,verify_frames)
        p=Path(output,'protocol.json');q=json.loads(p.read_text('utf-8'))
        q.update(complementary_deployment_stage=93,original_numerical_math_stage=78,base_compute_stage=85,
            original49_retained=True,total_research_forecasts=55,new_names=['kinematic_train6400','complement_history','complement_history_fixed','complement_fallback','complement_full','complement_full_fixed'],
            native_sdk_frequency79_enabled=False,histogram89_thrust_enabled=False,
            history_parameter_source='90 TRAIN0/4/6/10 only; sigma6400 fixed in source. No labels read during RAW inference.',
            video='Original forecasts.mp4 still shows original49 selection; study93 creates a separate cyan complementary overlay.',default_promoted=False)
        p.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');return result
    finally:ns.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
