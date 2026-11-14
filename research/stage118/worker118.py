"""Isolated RAW-only current20 geometry and learned-structure worker."""
from pathlib import Path
import argparse,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage117'),str(ROOT/'research/stage110'),str(ROOT/'research/stage107'),str(ROOT/'research/stage21')]
from fixed117 import FixedModel,geom
from merge110 import MergeRecovery
from center107 import CenterBank,NAMES
from windows21 import Observer
from event_drone_forecast.raw_io import open_raw,raw_chunks

NAMES118={'free_structure':'model_free20_kf118','fixed_structure':'model_fixed_kf118','fixed_guarded':'model_fixed_guarded_kf118'}

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else None;write=isinstance(mode,str) and any(c in mode for c in 'wax+')
        if not write and p.is_relative_to(ROOT.parent/'output') and p.suffix.lower() in ['.json','.jsonl','.png','.mp4','.npz']:
            raise RuntimeError('No cache/truth input allowed in RAW118 worker: '+str(p))

def main(raw,start,duration):
    sys.addaudithook(guard);reader=open_raw(raw);h,w=reader.get_size();front=Observer(w,h,'native20');front.geometry=MergeRecovery(w,h,'merge_safe')
    models={'free_structure':geom.MeasuredCenter(),'fixed_structure':FixedModel(),'fixed_guarded':FixedModel(True)}
    banks={name:CenterBank() for name in models};begin=max(0.,start-.1);stream=iter(raw_chunks(reader,begin,duration+start-begin if duration else 0.,20000));counter=0
    print(json.dumps(dict(ready=True,width=w,height=h,raw_only=True,cache_and_truth_reads_forbidden=True)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        assert request['id']==counter;counter+=1;cutoff=request['cutoff']
        while True:
            t,e=next(stream);obs=front.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
            assert t<=cutoff+1e-8
            if abs(t-cutoff)<1e-8:break
            assert t<=start+1e-8,'Worker skipped a requested model update'
        body=request['body'];axis=request['axis'];rotors=obs.get('rotors',[]);baseline=geom.rigid104.measure_four(rotors,body);outputs={};coeff={}
        for name,model in models.items():
            if name=='free_structure':model.input_measurement=baseline;_,d=model.step(t,body,rotors,axis)
            else:_,d=model.step(t,body,rotors,axis,baseline)
            q=banks[name].step(t,body,d,axis);outputs[name]=dict(center=d,forecast=q)
            if NAMES[2] in q['coefficients']:coeff[NAMES118[name]]=q['coefficients'][NAMES[2]]
        body_coeff=[q['forecast']['coefficients'].get(NAMES[0]) for q in outputs.values()];assert all(v==body_coeff[0] for v in body_coeff)
        print(json.dumps(dict(id=request['id'],cutoff=t,geometry=obs,outputs=outputs,coefficients=coeff,
            body_control=body_coeff[0],event_count=len(e),max_event_t=None if not len(e) else int(e['t'].max())),allow_nan=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,required=True)
    a=p.parse_args();main(a.raw,a.start,a.duration)
