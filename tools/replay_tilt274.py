"""One missing physical projection term, fixed source states and protocol."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from projected_tilt_trend import ProjectedTiltTrend,projected_normal_kernel
from replay_joint236 import source
from physical_publication import evaluate
from adaptive_force_memory import kernel
from score_rotor164 import score
from run_rotor_only import serial


def apply(row,trend):
    turn=row.get('normal_turn_forecast') or {}
    if turn.get('applied'):turn['projected_tilt_trend']=trend
    for component in row['force'].get('response_model_mixture',[]):apply(component,trend)


if __name__=='__main__':
    OUT=ROOT.parent/'output/tilt274_20261108';OUT.mkdir(exist_ok=False)
    files=['projected_tilt_trend.py','physical_publication.py','tools/replay_tilt274.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(change='future rotor normal projection length from current/past qualified conic inclination',
        no_recording_specific_rules=True,no_parameter_search=True,source_sha256=hashes,
        development=['13','15','16'],heldout1_opened=False,NN_used=False,GPU_used=False,
        current_positions_velocities_forces_and_periods_unchanged=True,
        body_axis_not_used_as_inclination=True,default_unchanged=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    zero=max(abs(projected_normal_kernel(-1/.35,h,w,.6,0)-kernel(-1/.35+1j*w,h)) for h in [0.,.1,.4,1.] for w in [-1.,0.,1.])
    assert zero<1e-9,zero
    results={};checks={}
    for n in protocol['development']:
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];model=ProjectedTiltTrend();outputs=[];changed=0;maxdelta=0
        for old in rows:
            r=copy.deepcopy(old);trend=model.observe(r['t'],r['structure'],r['geometry'])
            control=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {}
            for h,p in control.items():maxdelta=max(maxdelta,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
            if trend['qualified']:
                apply(r,trend);changed+=bool((r.get('normal_turn_forecast') or {}).get('applied'))
            r['projected_tilt_trend']=trend;r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs.append(r)
        assert maxdelta<1e-7,maxdelta
        path=OUT/f'{n}_tilt.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf8')
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]=dict(control=score(source(n),ref,origins),tilt=score(path,ref,origins));checks[n]=dict(control_max_px=maxdelta,current_tilt_rate_frames=changed)
        print(n,checks[n],{k:v['error'] for k,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(scores=results,checks=checks,protocol=protocol,zero_rate_kernel_error=zero),indent=2),'utf8')
