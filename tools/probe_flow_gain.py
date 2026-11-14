"""One defined local force-gain mechanism, no constant sweep or truth feedback."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from flow_identifiable_gain import fit_gain,FlowIdentifiableForce
from compare_final_pipeline import score


def controlled():
    t=np.arange(16)*.033333;unknown=np.array([40.,-100.]);gain=700.;answers=[]
    for changing in [False,True]:
        u=np.column_stack((np.sin(9*t),np.cos(7*t))) if changing else np.tile([0.,-1.],(len(t),1))
        I=np.vstack((np.zeros(2),np.cumsum(.5*(u[1:]+u[:-1])*np.diff(t)[:,None],axis=0)))
        v=np.array([10.,20.])+gain*I+t[:,None]*unknown
        hist=[(s,V,np.eye(2)) for s,V in zip(t,v)]
        k,info=fit_gain(hist,list(zip(t,u)))
        if changing:assert k is not None and abs(k-gain)<1e-8,(k,info)
        else:assert k is None,(k,info)
        answers.append(info)
    return answers


def main(run,label,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);checks=controlled()
    rows=[json.loads(s) for s in (Path(run)/'predictions.jsonl').read_text('utf-8').splitlines()]
    model=FlowIdentifiableForce();values=[];times=[]
    for r in rows:
        motion=r.get('image_motion') or {};model.observe_flow(motion.get('observation',dict(valid=False)),motion.get('acceleration',{}).get('applied',False))
        start=time.perf_counter();p,f=model.step(r['t'],r['coefficient'],r.get('force_observed_position'),r['force']['input_u'],.35)
        times.append((time.perf_counter()-start)*1000);values.append(dict(r,physical=p,force=f))
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
    truths=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    result=dict(controlled=checks,same_inputs_and_prediction_slots=True,RAW_FPS_not_measured=True,
        local_gain_updates=sum(r['force']['local_flow_gain'].get('updated',False) for r in values),
        final_gain=model.gain,mean_force_CPU_ms=float(np.mean(times)),
        scores={n:score(v,truths)['physical_future_error_px'] for n,v in [('control',rows),('identifiable_gain',values)]})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(*sys.argv[1:])
