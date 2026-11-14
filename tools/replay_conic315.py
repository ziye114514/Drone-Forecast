"""Same source state; only uncertain ellipse-plane forecast evolves."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from conic_direction_forecast import ConicDirectionForecast
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',default='conic316_20261109');args=p.parse_args()
    OUT=ROOT.parent/'output'/args.batch;OUT.mkdir(exist_ok=False)
    files=['conic_direction_forecast.py','normal_turn_forecast.py','endpoint_normal_trend.py','tools/replay_conic315.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        no_parameter_search=True,no_recording_specific_rules=True,current_p_v_a_and_force_state_frozen=True,
        ellipse_uncertainty_not_divided_by_child_count=True,all_inference_before_reference_reads=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8');results={};diagnostics={};parity={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        models=dict(parity=ConicDirectionForecast(override=False),conic=ConicDirectionForecast(),endpoint=ConicDirectionForecast(endpoint=True))
        outputs={a:[] for a in models};delta=0.
        for old in rows:
            for arm,m in models.items():
                r=m.step(old);r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs[arm].append(r)
                if arm=='parity':
                    for h,p in r['physical'].items():delta=max(delta,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
        assert delta<1e-7,(n,delta);parity[n]=delta
        for a,rs in outputs.items():(OUT/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rs),'utf8')
        ns=[r['conic_direction_forecast']['current_uncertainty'] for r in outputs['conic']]
        sig=[r['sigma_deg'] for r in ns if r['qualified']]
        diagnostics[n]=dict(qualified_conic_noise_frames=len(sig),median_sigma_deg=None if not sig else float(np.median(sig)),
            minimum_sigma_deg=None if not sig else min(sig),maximum_sigma_deg=None if not sig else max(sig))
        print(n,'parity',delta,diagnostics[n],flush=True)
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={a:score(OUT/f'{n}_{a}.jsonl',ref,origins) for a in models}
        print(n,{a:v['error'] for a,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,diagnostics=diagnostics,control_parity=parity,protocol=protocol),indent=2),'utf8')
