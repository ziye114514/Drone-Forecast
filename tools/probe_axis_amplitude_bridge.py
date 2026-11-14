"""Bridge current pose and recent amplitude on fixed causal RAW observations."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from axis_amplitude_bridge import AxisAmplitudeBridge
from flow_force_balance import FlowBalancedForce
from compare_final_pipeline import score


def controlled():
    bridge=AxisAmplitudeBridge();s=dict(geometry_epoch=1,axis_xy=[0.,-1.],axis_support=dict(valid=True,confidence=.8))
    bridge.step(0.,s,np.array([0.,-1.]),.35,dict(relative_f2=1.))
    s['axis_xy']=[.6,-.8]
    u,q,d=bridge.step(.033333,s,None,0.,dict(relative_f2=None))
    assert np.allclose(u,[.6,-.8]) and not d['learning_input_current']
    assert bridge.step(.133333,s,None,0.,dict(relative_f2=None))[0] is None
    return dict(current_axis_response=True,amplitude_not_current_or_learning_evidence=True,period_expires_without_self_refresh=True)


def main(run,label,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);checks=controlled()
    rows=[json.loads(s) for s in (Path(run)/'predictions.jsonl').read_text('utf-8').splitlines()]
    model=FlowBalancedForce();bridge=AxisAmplitudeBridge();values=[];times=[]
    for r in rows:
        motion=r.get('image_motion') or {};model.observe_flow(motion.get('observation',dict(valid=False)),motion.get('acceleration',{}).get('applied',False))
        start=time.perf_counter();u,q,d=bridge.step(r['t'],r['structure'],r['force']['input_u'],.35,r['period_selection'])
        model.input_learning_current=d['learning_input_current']
        p,f=model.step(r['t'],r['coefficient'],r.get('force_observed_position'),u,q)
        times.append((time.perf_counter()-start)*1000);values.append(dict(r,physical=p,force=f,axis_amplitude_bridge=d))
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    result=dict(controlled=checks,same_current_positions_velocity_geometry_and_prediction_slots=True,RAW_FPS_not_measured=True,
        bridged_ticks=sum(r['axis_amplitude_bridge']['applied'] for r in values),
        mean_force_CPU_ms=float(np.mean(times)),scores={n:score(v,truth)['physical_future_error_px'] for n,v in [('control',rows),('axis_amplitude_bridge',values)]})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(*sys.argv[1:])
