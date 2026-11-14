"""Experimental RAW120: original64 forecasts plus four partial-structure controls."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage120')]
import run_learned_structure_forecast as previous
NAMES120=['partial_'+m+'_kf120' for m in ['body_hold','affine3','conic3','guarded3']]

def build(worker=None):
    ns=previous.build(worker);old=ns['fingerprints']
    def fingerprints():
        hashes=old()
        for p in [Path(__file__),ROOT/'research/stage120/worker120.py',ROOT/'research/stage119/partial119.py']:
            hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return hashes
    ns['fingerprints']=fingerprints;return ns

def run(raw,output,start=0.,duration=0.,preview=False,verify_frames=False,dropout_mask=-1):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    tmp=ROOT.parent/'tmp';tmp.mkdir(exist_ok=True)
    log=tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='partial120_',suffix='.log',dir=tmp,delete=False);worker=None;ns=None
    try:
        worker=subprocess.Popen([sys.executable,str(ROOT/'research/stage120/worker120.py'),'--raw',str(raw),'--start',str(start),'--duration',str(duration),'--dropout-mask',str(dropout_mask)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        line=worker.stdout.readline()
        if not line:raise RuntimeError('RAW120 initialization failed: '+log.name)
        ready=json.loads(line);assert ready['ready'];worker.request_id=0;ns=build(worker)
        def receive(child,kind):
            line=child.stdout.readline()
            if not line:raise RuntimeError(kind+' worker stopped; inspect logs')
            return json.loads(line)
        ns['receive']=receive;result=ns['run'](raw,output,start,duration,32,preview,verify_frames)
        protocol=output/'protocol.json';p=json.loads(protocol.read_text('utf-8'));p.update(partial_structure_stage=120,original64_retained=True,new4=NAMES120,
            isolated_worker=ready,isolated_worker_log=log.name,default_promoted=False,physical_pixel_occlusion=False,
            scope120='Actual RAW execution, no current feature caches/targets. Optional deterministic one-observation dropout is a module intervention, not physical event occlusion. Default -1 applies no forced loss. Three-conic model update only after original full-four initialization; extreme lines retain original body fallback. Not realtime/trueCOM/RPM proof.')
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true');p.add_argument('--dropout-mask',type=int,choices=[-1,0,1,2,3],default=-1)
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames,a.dropout_mask)
