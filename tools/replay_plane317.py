"""Current rigid-plane angular response vs old window-average direction."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from plane_state_forecast import PlaneStateForecast
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score

if __name__=='__main__':
    OUT=ROOT.parent/'output/plane317_20261109';OUT.mkdir(exist_ok=False)
    files=['plane_state_forecast.py','conic_direction_forecast.py','physical_publication.py','tools/replay_plane317.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        no_parameter_search=True,no_recording_specific_rules=True,current_translation_state_frozen=True,
        unknown_force_parts_not_reassigned=True,same30degree_future_sweep_bound=True,all_inference_before_reference_reads=True,
        process_prior_is_same_as_existing_angular_observer=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8');results={};parities={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];models=dict(original_noise=PlaneStateForecast(conic_noise=False),conic_noise=PlaneStateForecast());outputs={a:[] for a in models};delta=0.
        for row in rows:
            if row['coefficient']:
                for h,p in evaluate(row,[.1,.2,.4]).items():delta=max(delta,float(np.max(abs(np.asarray(p)-row['physical'][h]))))
            for arm,model in models.items():
                r=model.step(row);r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs[arm].append(r)
        assert delta<1e-7,(n,delta);parities[n]=delta
        for a,rr in outputs.items():(OUT/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        print(n,'current-state preserved, baseline parity',delta,flush=True)
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={a:score(OUT/f'{n}_{a}.jsonl',ref,origins) for a in models}
        print(n,{a:v['error'] for a,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,control_parity=parities,protocol=protocol),indent=2),'utf8')
