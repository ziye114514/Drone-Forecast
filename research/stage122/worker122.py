"""RAW worker with causal full-view adequacy checks and shared-state forecast."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage120'),str(ROOT/'research/stage121')]
from worker120 import np,guard,PartialModel,geom,PARAMS,MergeRecovery,CenterBank,NAMES,Observer,open_raw,raw_chunks,NAMES120
from validation121 import ValidatedSelector

def main(raw,start,duration,mask):
    sys.addaudithook(guard);reader=open_raw(raw);h,w=reader.get_size();front=Observer(w,h,'native20');front.geometry=MergeRecovery(w,h,'merge_safe')
    models={m:PartialModel(m) for m in PARAMS['modes']};banks={m:CenterBank() for m in models};selector=ValidatedSelector()
    begin=max(0.,start-.1);stream=iter(raw_chunks(reader,begin,duration+start-begin if duration else 0.,20000));counter=0
    print(json.dumps(dict(ready=True,width=w,height=h,raw_only=True,cache_and_truth_reads_forbidden=True,feature_dropout_mask=mask,stage=122)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        index=request['id'];assert index==counter;counter+=1;cutoff=request['cutoff']
        while True:
            t,e=next(stream);obs=front.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
            assert t<=cutoff+1e-8
            if abs(t-cutoff)<1e-8:break
            assert t<=start+1e-8
        rotors=list(obs.get('rotors',[]));removed=False
        if mask>=0 and (65<=index<125 or 155<=index<185) and len(rotors)==4:
            centers=np.array([[r['cx'],r['cy']] for r in rotors]);order=np.argsort(np.arctan2(*(centers-centers.mean(0))[:,::-1].T))
            del rotors[int(order[mask])];removed=True
        body=request['body'];axis=request['axis'];baseline=geom.rigid104.measure_four(rotors,body);outputs={};coeff={}
        for name,model in models.items():
            _,d=model.step(t,body,rotors,axis,baseline);q=banks[name].step(t,body,d,axis);outputs[name]=dict(center=d,forecast=q)
            if NAMES[2] in q['coefficients']:coeff[NAMES120[name]]=q['coefficients'][NAMES[2]]
        validated=selector.step(dict(t=t,body=body,input_rotors=rotors,axis=axis,outputs=outputs))
        if NAMES[2] in validated['forecast']['coefficients']:coeff['partial_validated_kf122']=validated['forecast']['coefficients'][NAMES[2]]
        body_coeff=[v['forecast']['coefficients'].get(NAMES[0]) for v in outputs.values()];assert all(v==body_coeff[0] for v in body_coeff)
        assert validated['forecast']['coefficients'].get(NAMES[0])==body_coeff[0]
        print(json.dumps(dict(id=index,cutoff=t,input_rotors=rotors,removed_one_observation=removed,outputs=outputs,validated122=validated,coefficients=coeff,body_control=body_coeff[0],
            event_count=len(e),max_event_t=None if not len(e) else int(e['t'].max()),feature_cache_read=False,physical_pixel_occlusion=False),allow_nan=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,required=True)
    p.add_argument('--dropout-mask',type=int,choices=[-1,0,1,2,3],default=-1);a=p.parse_args();main(a.raw,a.start,a.duration,a.dropout_mask)
