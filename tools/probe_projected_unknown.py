"""Fixed causal image observations, rank1 unknown-force projection; labels last."""
from pathlib import Path
import sys,json,time,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from projected_unknown_force import ProjectedResidualForce
from compare_final_pipeline import score


def controlled():
    m=ProjectedResidualForce();structure=dict(mode='axis_only',axis_xy=[0.,-1.],axis_support=dict(valid=True,confidence=.9))
    normals=[]
    for t in np.arange(12)/30:
        a=np.array([100.,0.]);c=dict(position=(.5*t*t*a).tolist(),velocity=(t*a).tolist(),acceleration=[0.,0.])
        motion=dict(acceleration=dict(applied=True,shrunk_acceleration=a.tolist(),standard_error_proxy=[10.,10.]))
        m.observe_flow(dict(valid=False),False);m.observe_axis_motion(structure,motion)
        _,d=m.step(float(t),c,c['position'],np.array([0.,-1.]),.8)
        if d['projected_unknown_force']['updated']:normals.append(abs(d['projected_unknown_force']['normal_component_of_correction']))
    assert normals and max(normals)<1e-10 and m.d[0]>0.
    return dict(projected_feedback_updates=len(normals),normal_component_max_error=max(normals),
        normal_force_not_identified=True,constant_wind_tangent_responds=True)


def main(run,label,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);checks=controlled()
    snapshot=out/'source_snapshot';sources={}
    for name in ['projected_unknown_force.py','axis_amplitude_bridge.py','flow_force_balance.py','tools/probe_projected_unknown.py']:
        data=(ROOT/name).read_bytes();p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        sources[name]=hashlib.sha256(data).hexdigest()
    source=Path(run);file=source/'predictions.jsonl'
    if not file.exists():file=source/'same_tracks_rigid.jsonl'
    rows=[json.loads(s) for s in file.read_text('utf-8').splitlines()]
    model=ProjectedResidualForce();values=[];cost=[]
    for r in rows:
        motion=r.get('image_motion') or {};model.observe_flow(motion.get('observation',dict(valid=False)),motion.get('acceleration',{}).get('applied',False))
        model.observe_axis_motion(r['structure'],motion)
        model.input_learning_current=(r.get('axis_amplitude_bridge') or {}).get('learning_input_current',True)
        start=time.perf_counter();p,f=model.step(r['t'],r['coefficient'],r.get('force_observed_position'),r['force']['input_u'],.35)
        cost.append((time.perf_counter()-start)*1000);values.append(dict(r,physical=p,force=f))
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    result=dict(controlled=checks,same_inputs_positions_and_prediction_slots=True,RAW_FPS_not_measured=True,
        mechanism='current_tangent_scalar_CI_no_second_EWMA',sources=sources,
        updates=sum(r['force']['projected_unknown_force']['updated'] for r in values),mean_force_CPU_ms=float(np.mean(cost)),
        scores={n:score(v,truth)['physical_future_error_px'] for n,v in [('control',rows),('projected_unknown',values)]})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(*sys.argv[1:])
