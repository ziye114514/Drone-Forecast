"""Current measured rotor changes, frozen states, no per-video choices."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from visible_force_rate import VisibleForceRate
from physical_publication import evaluate
from separated_force_response import rate_kernel
from continuous_phase_force import torque_period
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='visible_rate355_20261109');a=parser.parse_args()
    out=ROOT.parent/'output'/a.batch;out.mkdir(exist_ok=False)
    files=['visible_force_rate.py','anisotropic_response.py','physical_publication.py','rotor_frequency_drive.py','tools/probe_visible_rate355.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    # Independent constant-normal exponential-rate force integral.
    c=dict(position=[30.,80.],velocity=[200.,-50.],acceleration=[100.,-20.])
    f=dict(acceleration_px_s2=c['acceleration'],acceleration_correlation_s=.35,visible_force_rate=dict(amplitude_rate_px_s3=300.,normal_xy=[0.,-1.],correlation_s=.35))
    r=dict(coefficient=c,force=f);actual=evaluate(r,[0.,.1,.4]);control=copy.deepcopy(r);control['force'].pop('visible_force_rate');expected=evaluate(control,[0.,.1,.4])
    err=max(float(np.max(abs(np.asarray(actual[k])-v-rate_kernel(-1/.35,-1/.35,float(k))*np.array([0.,-300.])))) for k,v in expected.items());assert err<1e-7,err
    counts={};arms={}
    for n in ('13','15','16','11','3','4'):
        rows=[json.loads(s) for s in (ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl').read_text('utf8').splitlines()]
        phases=[json.loads(s) for s in (ROOT.parent/f'output/bootstrap311_20261109/recording{n}/periods.jsonl').read_text('utf8').splitlines()] if n in ('13','15','16') else None
        arms[n]={};counts[n]={}
        for arm in ('native','joint_phase') if phases is not None else ('native',):
            model=VisibleForceRate();outputs=[];used=0
            for i,row in enumerate(rows):
                period=torque_period(row,phases[i]) if arm=='joint_phase' else None
                new=model.step(row,period);assert new['coefficient']==row['coefficient'] and new['valid']==row['valid'];used+=new['visible_force_rate']['applied'];new['physical']=evaluate(new,[.1,.2,.4]) if new['coefficient'] else {};outputs.append(new)
            path=out/f'{n}_{arm}.jsonl';path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8');arms[n][arm]=path;counts[n][arm]=used
        print(n,'frozen',counts[n],flush=True)
    cases={}
    for n,paths in arms.items():
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl',ref,origins),**{arm:score(path,ref,origins) for arm,path in paths.items()});print(n,{k:v['error'] for k,v in cases[n].items()},flush=True)
    result=dict(cases=cases,applied_frames=counts,analytic_force_rate_integral_max_error=err,source_sha256=hashes,
        current_position_velocity_acceleration_unchanged=True,all_cases_development=True,no_parameter_search=True,no_new_RAW_FPS=True,
        hidden_frequencies_not_imputed=True,joint_phase_arm_is_saved_slow_CPU_evidence_not_fast_RAW=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8')
