"""Fixed-input causal force-residual pilot; annotations read after output freeze."""
from pathlib import Path
import sys,json,copy,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from flow_force_balance import FlowBalancedForce,residual_history
from compare_final_pipeline import score


def controlled():
    # Known time-varying input plus a constant unknown acceleration. The exact
    # trapezoidal integral must remove the known command before slope fitting.
    t=np.arange(16)*.033333;u=np.column_stack((np.sin(3*t),np.cos(2*t)))
    I=np.vstack((np.zeros(2),np.cumsum(.5*(u[1:]+u[:-1])*np.diff(t)[:,None],axis=0)))
    unknown=np.array([40.,-100.]);gain=700.;v=np.array([10.,20.])+gain*I+t[:,None]*unknown
    hist=[(s,V,np.eye(2)) for s,V in zip(t,v)]
    residual,info=residual_history(hist,list(zip(t,u)),gain)
    slope=np.linalg.lstsq(np.column_stack((np.ones(len(t)),t)),np.array([r[1] for r in residual]),rcond=None)[0][1]
    assert np.allclose(slope,unknown,atol=1e-8),slope
    missing=list(zip(t,u));missing[4]=(t[4],None)
    assert residual_history(hist,missing,gain)[0] is None
    return dict(constant_unknown_max_error=float(abs(slope-unknown).max()),missing_known_input_rejected=True)


def main(run,label,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);checks=controlled()
    rows=[json.loads(s) for s in (Path(run)/'predictions.jsonl').read_text('utf-8').splitlines()]
    model=FlowBalancedForce();values=[];cost=[]
    for r in rows:
        motion=r.get('image_motion') or {};obs=motion.get('observation',dict(valid=False))
        model.observe_flow(obs,motion.get('acceleration',{}).get('applied',False))
        start=time.perf_counter();p,f=model.step(r['t'],r['coefficient'],r.get('force_observed_position'),r['force']['input_u'],.35)
        cost.append((time.perf_counter()-start)*1000);values.append(dict(r,physical=p,force=f))
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    result=dict(controlled=checks,same_inputs_and_valid_prediction_slots=True,RAW_FPS_not_measured=True,
        updates=sum(r['force']['flow_force_balance'].get('updated',False) for r in values),
        mean_force_CPU_ms=float(np.mean(cost)),scores={n:score(v,truth)['physical_future_error_px'] for n,v in [('control',rows),('flow_balance',values)]})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(*sys.argv[1:])
