"""Frozen observability correction, then score development and unseen3 once."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from identifiable_tangent_memory import IdentifiableTangentMemory,adapt,fit_samples
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score
from run_rotor_only import serial


def mechanisms():
    samples=[]
    for i in range(30):
        t=i*.033333;n=np.array([.2*np.sin(t*3),-1.]);n/=np.linalg.norm(n);d=np.array([-n[1],n[0]])
        v=np.array([400.,-100.]);X=np.r_[d,-d@v];samples.append((t,X,float(d@np.array([120.,900.])),4.))
    constant=fit_samples(samples);assert constant['information_fraction']<1e-12,constant
    assert abs(constant['effective_response_s_inv']-1/.35)<1e-12,constant
    samples=[];offset=[];rotated=[];angle=.7;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    for i in range(60):
        t=i*.033333;n=np.array([.3*np.sin(3*t),-1.]);n/=np.linalg.norm(n);d=np.array([-n[1],n[0]])
        v=np.array([400*np.sin(t*4),200*np.cos(t*3)]);b=np.array([120.,900.]);a=1500*np.cos(t)*n+b-1.2*v
        X=np.r_[d,-d@v];samples.append((t,X,float(d@a),4.))
        offset.append((t,X,float(d@(a+np.array([700.,-450.]))),4.))
        dr=R@d;vr=R@v;rotated.append((t,np.r_[dr,-dr@vr],float(dr@(R@a)),4.))
    base=fit_samples(samples);shift=fit_samples(offset);turn=fit_samples(rotated)
    e1=abs(base['effective_response_s_inv']-shift['effective_response_s_inv']);e2=abs(base['effective_response_s_inv']-turn['effective_response_s_inv'])
    assert max(e1,e2)<1e-10,(e1,e2)
    return dict(constant_velocity_data_credit=constant['information_fraction'],
        arbitrary_constant_unknown_force_offset_changes_response_by=e1,
        coordinate_rotation_changes_response_by=e2,synthetic_true_response=1.2,
        synthetic_estimate=base['effective_response_s_inv'],physical_real_data_identification_not_proved=True)


if __name__=='__main__':
    OUT=ROOT.parent/'output/identifiable289_20261108';OUT.mkdir(exist_ok=False)
    files=['identifiable_tangent_memory.py','tangential_response_memory.py','tools/replay_identifiable289.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(source_sha256=hashes,change='data-only Schur nuisance elimination, no finite gravity/wind prior used to claim c identifiable',
        correction_specified_before_new_recording3_outcome_inspection=True,
        no_recording_specific_rules=True,no_parameter_search=True,NN_used=False,GPU_used=False,
        development=['13','15','16','11'],new_outcome='3',do_not_retune_after_test=True,
        mechanical_drag_identification_not_claimed=True,old_default_not_replaced=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    checks=mechanisms();result={}
    for n in ['13','15','16','11','3']:
        if n=='3':
            src=ROOT.parent/'output/frozen_tangent287_20261108/control.jsonl'
            if not src.exists():
                (OUT/'development_partial.json').write_text(json.dumps(dict(scores=result,checks=checks,protocol=protocol),indent=2),'utf8')
                print('New3 inference still running; frozen candidate not changed; rerun separate scoring after completion.',flush=True);break
            ref=ROOT.parent/'output/frozen_tangent287_20261108/references.json'
        else:
            src=source(n);folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16','11':'regional212_20261108/reference11'}[n];ref=ROOT.parent/f'output/{folder}/references.json'
        rows=[json.loads(s) for s in src.read_text('utf8').splitlines()];memory=IdentifiableTangentMemory();outputs=[];parity=0.;active=0
        for old in rows:
            info=memory.observe(old);r=adapt(old,info);active+=bool(info.get('qualified'))
            control=evaluate(old,[.1,.2,.4]) if old['coefficient'] else {}
            for h,p in control.items():parity=max(parity,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
            r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs.append(r)
        assert parity<1e-7,parity
        path=OUT/f'{n}_tangent.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf8')
        origins=np.round(np.arange(10.8,15.6001,.2),1)
        result[n]=dict(control=score(src,ref,origins),tangent=score(path,ref,origins))
        print(n,{k:v['error'] for k,v in result[n].items()},'last response',outputs[-1]['tangential_response_memory'].get('effective_response_s_inv'),flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(scores=result,checks=checks,protocol=protocol),indent=2),'utf8')
