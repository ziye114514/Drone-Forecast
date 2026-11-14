"""Experimental RAW-only planar-center branch104; original93 stays unchanged."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage104')]
import run_complementary_forecast as previous
from online104 import OnlineExtras


def build():
    ns=previous.build();context={};associate=ns['associate']
    # Current body boxes in the old pipeline do not all carry t.
    def capture_time(sensed,position):
        result=associate(sensed,position);context.update(sensed=result[0]);return result
    class Extras(OnlineExtras):
        def __init__(self,w,h):super().__init__(w,h,context)
        def measure(self,t,events,body):context['t']=t;return super().measure(t,events,body)
    ns['associate']=capture_time;ns['OnlineExtras']=Extras;old=ns['fingerprints']
    def fingerprints():
        result=old()
        for p in [Path(__file__),ROOT/'research/stage104/rigid104.py',ROOT/'research/stage104/online104.py',
                  ROOT/'research/stage58/geometry58.py',ROOT/'research/stage49/rigid49.py']:
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
        q.update(planar_center_stage=104,original55_retained=True,default_promoted=False,
            scope104='RAW current four-conic center/layout, five-frame lock, side/partial center fallback; no metric pose/COM truth. Separate view-dependent force gain trained on corrected center with matched constant/zero controls.')
        p.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');return result
    finally:ns.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
