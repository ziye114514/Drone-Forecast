"""Current RAW126 observations -> independent occupancy model/shared filter."""
from pathlib import Path
import argparse,json,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage127')]
from independent127 import MeasurementFusion,PartialModel,geom,NAMES
from worker122 import guard
def main(raw,start,duration):
    sys.addaudithook(guard);log=tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='original126_for128_',suffix='.log',dir=ROOT.parent/'tmp',delete=False);old=None
    try:
        old=subprocess.Popen([sys.executable,str(ROOT/'research/stage126/worker126.py'),'--raw',raw,'--start',str(start),'--duration',str(duration)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        line=old.stdout.readline()
        if not line:raise RuntimeError('Original126 failed: '+log.name)
        ready=json.loads(line);assert ready['ready'];model=PartialModel('affine3');fusion=MeasurementFusion();counter=0
        print(json.dumps(dict(ready=True,width=ready['width'],height=ready['height'],stage=128,raw_only=True,cache_and_truth_reads_forbidden=True,old126_isolated=True,old126_log=log.name)),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            assert request['id']==counter;counter+=1;old.stdin.write(json.dumps(request)+'\n');old.stdin.flush();response=old.stdout.readline()
            if not response:raise RuntimeError('Original126 stopped: '+log.name)
            result=json.loads(response);t=result['cutoff'];assert result['id']==request['id'] and abs(t-request['cutoff'])<1e-8
            occupancy=result['support126']['occupancy_geometry'].get('rotors',[]);m=geom.rigid104.measure_four(occupancy,request['body']);_,d=model.step(t,request['body'],occupancy,request['axis'],m)
            out=fusion.step(t,request['body'],request['axis'],result['outputs']['affine3']['center'],d)
            assert out['forecast']['coefficients'].get(NAMES[0])==result['body_control']
            c=out['forecast']['coefficients'].get(NAMES[2])
            if c is not None:result['coefficients']['support_independent_kf128']=c
            result['support128']=dict(output=out,occupancy_model=d,no_cache_reads=True,only_current126_observations=True)
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
