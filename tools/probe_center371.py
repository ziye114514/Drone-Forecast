"""Check projected-body-centre readout from the existing correlated layout state.

No trajectory/model fit changes: x contains p and b with layout=p+b. Output
that modeled centre, using each hypothesis's own b and cross covariance.
All old origins remain eligible; no known future layout is supplied.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from adaptive_response_observer import AdaptiveResponseObserver
from inverse_normal_response import InverseNormalResponse
from anisotropic_response import AnisotropicResponse
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


def centered(row,observer):
    r=copy.deepcopy(row);mixture=r['force'].get('response_model_mixture') or [];total=np.zeros(2);infos=[]
    leaves=mixture if mixture else [dict(weight=1.,coefficient=r['coefficient'],force=r['force'])]
    for component,m in zip(leaves,(observer,observer.alternative)):
        if component['coefficient'] is None:continue
        offset=m.x[9:11].copy() if m.layout_initialized else np.zeros(2)
        component['coefficient']['position']=(np.asarray(component['coefficient']['position'])+offset).tolist();total+=component.get('weight',1.)*offset
        J=np.zeros((2,m.state_dim));J[:,:2]=np.eye(2)
        if m.layout_initialized:J[:,9:11]=np.eye(2)
        covariance=J@m.P@J.T
        assert np.linalg.eigvalsh((covariance+covariance.T)/2).min()>-1e-6
        infos.append(dict(offset=offset.tolist(),layout_initialized=m.layout_initialized,layout_age_s=None if m.layout_t is None else row['t']-m.layout_t,
            covariance=((covariance+covariance.T)/2).tolist()))
    if mixture and r['coefficient']:r['coefficient']['position']=(np.asarray(r['coefficient']['position'])+total).tolist()
    r['body_center_readout']=dict(mode='existing_correlated_p_plus_b',offset=total.tolist(),branches=infos,
        independent_current_body_or_COM_truth=False,force_velocity_and_acceleration_unchanged=True,
        current_or_future_labels_used=False,missing_current_layout_does_not_drop_forecasts=True)
    r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {}
    return r


if __name__=='__main__':
    out=ROOT.parent/'output/center371_20261109';out.mkdir(exist_ok=False)
    files=['tools/probe_center371.py','adaptive_response_observer.py','layout_reference_observer.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    parity={};counts={}
    for n in ('13','15','16','11','3','4'):
        rows=[json.loads(s) for s in (ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl').read_text('utf8').splitlines()]
        expected=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        m=AdaptiveResponseObserver();inv=InverseNormalResponse();disk=AnisotropicResponse();outputs=[];d=0.;used=0
        for old,wanted in zip(rows,expected):
            r=copy.deepcopy(old);u=r['force'].get('input_u');u=None if u is None else np.asarray(u);meta=r['period_selection']
            fresh=m.x is None or m.seen is not None and r['t']-m.seen>.5
            if fresh and r['body'].get('detected') and not r['body'].get('center_provisional'):m.two_point_init=bool(r['state'].get('two_point_initialization_selected_by_partial_entry'))
            c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'],structure=r['structure'])
            r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'));r=disk.step(inv.step(r))
            if c:
                for h,pred in evaluate(r,[.1,.2,.4]).items():d=max(d,float(np.max(abs(np.asarray(pred)-wanted['physical'][h]))))
            assert bool(c)==bool(wanted['coefficient']);assert v==wanted['valid']
            new=centered(r,m);used+=bool(np.linalg.norm(new['body_center_readout']['offset'])>1e-8);outputs.append(new)
        assert d<1e-7,(n,d);parity[n]=d;counts[n]=used
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8');print(n,'frozen parity',d,'centre updates',used,flush=True)
    cases={}
    for n in counts:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),centered=score(out/f'{n}.jsonl',ref,origins));print(n,{k:v['error'] for k,v in cases[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,old_force_replay_parity=parity,initialized_center_output_frames=counts,source_sha256=hashes,one_global_rule=True,no_parameter_search=True,all_cases_development=True,NN_used=False),indent=2),'utf8')
