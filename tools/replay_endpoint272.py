"""One normal-rate lag correction. Positions, force states and periods frozen."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from endpoint_normal_trend import EndpointNormalTrend,rotation_kernel
from adaptive_force_memory import kernel
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score
from run_rotor_only import serial


def modify(r,trend):
    if (r.get('normal_turn_forecast') or {}).get('applied'):
        r['normal_turn_forecast']['trend']=trend
    for component in r['force'].get('response_model_mixture',[]):modify(component,trend)


if __name__=='__main__':
    OUT=ROOT.parent/'output/endpoint272_20261108';OUT.mkdir(exist_ok=False)
    files=['endpoint_normal_trend.py','physical_publication.py','tools/replay_endpoint272.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(change='past-plane curvature corrects window-average normal-rate lag; same p/v/a and observed period',
        source_sha256=hashes,no_recording_specific_rules=True,no_parameter_search=True,development=['13','15','16'],
        heldout1_opened=False,NN_used=False,GPU_used=False,existing_angular_noise_and_sweep_bounds_retained=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    exact=max(abs(rotation_kernel(-1/.35,h,w,0.)-kernel(-1/.35+1j*w,h)) for h in [0.,.1,.4,1.] for w in [-1.,0.,1.])
    assert exact<1e-9,exact
    results={};checks={}
    for n in protocol['development']:
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];model=EndpointNormalTrend();outputs=[];maxdelta=0.;applied=0
        for old in rows:
            r=copy.deepcopy(old);t=r['t'];motion=r['motion'];estimate=motion.get('estimate') or {};dt=motion.get('dt_s');flow=dict(valid=False)
            if dt and motion.get('current_observation') and not estimate.get('peak_at_search_boundary') and estimate.get('score',0)>0:
                flow=dict(valid=True,velocity=[estimate['dx']/dt,estimate['dy']/dt])
            trend=model.observe(t,r['structure'],r['coefficient'],dict(observation=flow))
            control=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {}
            for h,p in control.items():maxdelta=max(maxdelta,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
            if trend.get('qualified'):
                modify(r,trend);applied+=bool((r.get('normal_turn_forecast') or {}).get('applied'))
            r['endpoint_plane_trend']=trend;r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs.append(r)
        assert maxdelta<1e-7,maxdelta
        path=OUT/f'{n}_endpoint.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf8')
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]=dict(control=score(source(n),ref,origins),endpoint=score(path,ref,origins))
        checks[n]=dict(control_max_px=maxdelta,changed_trend_frames=applied)
        print(n,checks[n],{k:v['error'] for k,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(scores=results,checks=checks,protocol=protocol,quadrature_check=exact),indent=2),'utf8')
