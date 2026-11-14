"""Experimental126: original69 RAW forecasts plus complementary support KF.

Run as a CLI subprocess. Default stable interface is intentionally unchanged.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
import run_validated_structure_forecast as previous
def build(worker=None):
    ns=previous.build(worker);old=ns['fingerprints']
    def fingerprints():
        hashes=old()
        for p in [Path(__file__),ROOT/'research/stage126/worker126.py',ROOT/'research/stage124/occupancy124.py',ROOT/'research/stage125/hybrid125.py',ROOT/'research/stage55/roi55.py']:
            hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return hashes
    ns['fingerprints']=fingerprints;return ns
def run(raw,output,start=0.,duration=0.,preview=False,verify_frames=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    tmp=ROOT.parent/'tmp';tmp.mkdir(exist_ok=True);log=tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='support126_',suffix='.log',dir=tmp,delete=False);worker=None;ns=None
    try:
        worker=subprocess.Popen([sys.executable,str(ROOT/'research/stage126/worker126.py'),'--raw',str(raw),'--start',str(start),'--duration',str(duration)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        line=worker.stdout.readline()
        if not line:raise RuntimeError('RAW126 initialization failed: '+log.name)
        ready=json.loads(line);assert ready['ready'];worker.request_id=0;ns=build(worker)
        def receive(child,kind):
            line=child.stdout.readline()
            if not line:raise RuntimeError(kind+' worker stopped; inspect logs')
            return json.loads(line)
        ns['receive']=receive;result=ns['run'](raw,output,start,duration,32,preview,verify_frames)
        protocol=output/'protocol.json';p=json.loads(protocol.read_text('utf-8'));p.update(support_stage=126,original69_retained=True,new_method='support_hybrid_kf126',isolated_worker=ready,isolated_worker_log=log.name,
            default_promoted=False,scope126='RAW and existing NN weights only; current whole-object hint gates separately measured bipolar occupancy. Original104 conic checks choose fallback. Original122 runs isolated, no label/cache reads. No new thrust/COM/real-time claim.')
        protocol.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8');return result
    finally:
        if worker is not None:
            if worker.poll() is None:
                try:worker.stdin.write(json.dumps(dict(stop=True))+'\n');worker.stdin.flush();worker.wait(timeout=15)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):worker.terminate();worker.wait(timeout=10)
            for s in [worker.stdin,worker.stdout]:
                if s is not None:s.close()
        log.close()
        if ns is not None:ns.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true');a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
