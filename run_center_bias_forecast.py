"""Experimental RAW107: previous60 preserved, four center/bias controls added."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'research/stage107')]
import run_planar_center_forecast as previous
from center107 import CenterBank,NAMES,VENDOR


def build():
    ns=previous.build();Previous=ns['OnlineExtras']
    class Extras(Previous):
        def __init__(self,w,h):super().__init__(w,h);self.centers107=CenterBank();self.body107=None
        def measure(self,t,events,body):self.body107=body;return super().measure(t,events,body)
        def forecast(self,q,obs,rotor_quality):
            q=super().forecast(q,obs,rotor_quality);v=self.centers107.step(q['t'],self.body107,q['rigid_center_diagnostic'],self.current['body'])
            q=dict(q,coefficients=dict(q['coefficients']),predictions=dict(q['predictions']))
            if q['valid']:
                for name,c in v['coefficients'].items():
                    q['coefficients'][name]=c;q['predictions'][name]={str(h):(np.asarray(c['position'])+np.asarray(c['velocity'])*h).tolist() for h in [.1,.2,.4]}
            q['center_bias_diagnostic']=v['diagnostics'];return q
    import numpy as np
    ns['OnlineExtras']=Extras;old=ns['fingerprints']
    def fingerprints():
        result=old()
        for p in [Path(__file__),ROOT/'research/stage107/center107.py',*VENDOR.glob('*')]:
            if p.is_file():result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
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
        q.update(center_bias_stage=107,previous60_retained=True,new4=NAMES,default_promoted=False,
            scope107='Unified image center/velocity/appearance-bias KF with uncalibrated geometry and side covariance; no new force parameters.')
        p.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');return result
    finally:ns.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
