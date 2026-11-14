"""Isolated occupancy augmentation of unmodified RAW122 worker.

Old122 remains a separate process so extra OpenCV work cannot change its RNG.
Both workers read RAW only, and advance exactly to each requested cutoff.
"""
from pathlib import Path
import argparse,json,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage124'),str(ROOT/'research/stage125')]
from occupancy124 import SupportFrontend
from hybrid125 import Hybrid,NAMES
from worker122 import guard,Observer,open_raw,raw_chunks

def main(raw,start,duration):
    sys.addaudithook(guard)
    log=tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='original122_for126_',suffix='.log',dir=ROOT.parent/'tmp',delete=False)
    old=None
    try:
        old=subprocess.Popen([sys.executable,str(ROOT/'research/stage122/worker122.py'),'--raw',raw,'--start',str(start),'--duration',str(duration),'--dropout-mask','-1'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ready=json.loads(old.stdout.readline());assert ready['ready']
        reader=open_raw(raw);h,w=reader.get_size();front=Observer(w,h,'native20');front.geometry=SupportFrontend(w,h,'occupancy_roi');hybrid=Hybrid()
        begin=max(0.,start-.1);stream=iter(raw_chunks(reader,begin,duration+start-begin if duration else 0.,20000));counter=0
        print(json.dumps(dict(ready=True,width=w,height=h,stage=126,raw_only=True,cache_and_truth_reads_forbidden=True,old122_isolated=True,old122_log=log.name)),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            assert request['id']==counter;counter+=1;cutoff=request['cutoff']
            old.stdin.write(json.dumps(request)+'\n');old.stdin.flush();oldline=old.stdout.readline()
            if not oldline:raise RuntimeError('Original122 stopped; inspect '+log.name)
            result=json.loads(oldline);assert result['id']==request['id'] and abs(result['cutoff']-cutoff)<1e-8
            while True:
                t,e=next(stream);assert t<=cutoff+1e-8
                hint=request['body'] if abs(t-cutoff)<1e-8 else dict(t=t,detected=False)
                front.geometry.set_hint(e,hint,t)
                obs=front.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
                if abs(t-cutoff)<1e-8:break
                assert t<=start+1e-8
            out=hybrid.step(t,request['body'],request['axis'],result['input_rotors'],obs.get('rotors',[]))
            assert out['forecast']['coefficients'].get(NAMES[0])==result['body_control']
            c=out['forecast']['coefficients'].get(NAMES[2])
            if c is not None:result['coefficients']['support_hybrid_kf126']=c
            result['support126']=dict(occupancy_geometry=obs,hybrid=out,raw_only=True,event_count=len(e),max_event_t=int(e['t'].max()) if len(e) else None,
                roi=front.geometry.current_roi,threshold=front.geometry.support_threshold,no_future_hint=True,cache_read=False)
            print(json.dumps(result,allow_nan=False),flush=True)
    finally:
        if old is not None:
            if old.poll() is None:
                try:old.stdin.write(json.dumps(dict(stop=True))+'\n');old.stdin.flush();old.wait(timeout=15)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):old.terminate();old.wait(timeout=10)
            for s in [old.stdin,old.stdout]:
                if s is not None:s.close()
        log.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,required=True);a=p.parse_args();main(a.raw,a.start,a.duration)
