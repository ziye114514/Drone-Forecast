"""Experimental RAW118: preserve107's64 branches, add3 fixed-structure controls."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage118')]
import run_center_bias_forecast as previous
NAMES118=['model_free20_kf118','model_fixed_kf118','model_fixed_guarded_kf118']

def build(worker=None):
    ns=previous.build();Previous=ns['OnlineExtras']
    class Extras(Previous):
        def measure(self,t,events,body):
            result=super().measure(t,events,body)
            if worker is None:raise RuntimeError('118 requires isolatedRAW worker')
            ns['send'](worker,dict(id=worker.request_id,cutoff=t,body=body,axis=self.current['body']));worker.request_id+=1
            line=worker.stdout.readline()
            if not line:raise RuntimeError('RAW118 geometry worker stopped; inspect worker log')
            self.structure118=json.loads(line);assert self.structure118['id']==worker.request_id-1 and abs(self.structure118['cutoff']-t)<1e-8
            return result
        def forecast(self,q,obs,rotor_quality):
            q=super().forecast(q,obs,rotor_quality);q=dict(q,coefficients=dict(q['coefficients']),predictions=dict(q['predictions']))
            if q['valid']:
                for name,c in self.structure118['coefficients'].items():
                    q['coefficients'][name]=c;q['predictions'][name]={str(h):(np.asarray(c['position'])+h*np.asarray(c['velocity'])).tolist() for h in [.1,.2,.4]}
            assert self.structure118['body_control']==q['coefficients'].get('center_body_kf107') or not q['valid']
            q['learned_structure118']=self.structure118;return q
    ns['OnlineExtras']=Extras;old=ns['fingerprints']
    def fingerprints():
        hashes=old()
        for path in [Path(__file__),ROOT/'research/stage118/worker118.py',ROOT/'research/stage117/fixed117.py',ROOT/'research/stage113/support113.py',ROOT/'research/stage110/merge110.py']:
            hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes
    ns['fingerprints']=fingerprints;return ns

def run(raw,output,start=0.,duration=0.,preview=False,verify_frames=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    tmp=ROOT.parent/'tmp';tmp.mkdir(exist_ok=True)
    log=tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='geometry118_',suffix='.log',dir=tmp,delete=False)
    worker=None;ns=None
    try:
        worker=subprocess.Popen([sys.executable,str(ROOT/'research/stage118/worker118.py'),'--raw',str(raw),'--start',str(start),'--duration',str(duration)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        line=worker.stdout.readline()
        if not line:raise RuntimeError('RAW118 worker initialization failed: '+log.name)
        ready=json.loads(line);assert ready['ready'];worker.request_id=0;ns=build(worker)
        def receive(child,kind):
            line=child.stdout.readline()
            if not line:raise RuntimeError(kind+' worker stopped; inspect logs')
            return json.loads(line)
        ns['receive']=receive
        result=ns['run'](raw,output,start,duration,32,preview,verify_frames)
        protocol=output/'protocol.json';p=json.loads(protocol.read_text('utf-8'));p.update(learned_structure_stage=118,original64_retained=True,new3=NAMES118,
            isolated_worker=ready,isolated_worker_log=log.name,default_promoted=False,
            scope118='ActualRAW and fixed NN weights only. Separate RAW geometry worker preserves old OpenCV RNG state; no caches/targets. Five-frame learned structure then H8 current pose; same body/axis and matched free-shape control. No trueCOM/RPM or realtime guarantee.')
        protocol.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8');return result
    finally:
        if worker is not None:
            if worker.poll() is None:
                try:worker.stdin.write(json.dumps(dict(stop=True))+'\n');worker.stdin.flush();worker.wait(timeout=15)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):worker.terminate();worker.wait(timeout=10)
            for stream in [worker.stdin,worker.stdout]:
                if stream is not None:stream.close()
        log.close()
        if ns is not None:ns.clear()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
