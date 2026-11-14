"""Compose anonymous measured regional input with the force-scale prior.

Regions have no motor head/tail IDs. Missing rotors remain entirely unknown.
Compare one global policy with the exact retained463 baseline, no refit.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from anonymous_rotor_group import AnonymousRotorGroup
from equilibrium_gain_prior import EquilibriumGainObserver,anchor_metadata
from replay_endpoint432 import run
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate

if __name__=='__main__':
    out=ROOT.parent/'output/partial_gain467_20261111';out.mkdir(exist_ok=False)
    files=['anonymous_rotor_group.py','equilibrium_gain_prior.py','tools/evaluate_partial467.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,anonymous_group_not_total_force=True,
        missing_rotors_owned_by_unknown=True,old_named_bank_inputs_preserved=True,no_fit_or_parameter_search=True,
        all6_uniform=True,all_inference_before_references=True),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');parity={};evidence={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/positive_normal457_20261110/{n}_positive.jsonl').read_text('utf8').splitlines()];bank=AnonymousRotorGroup();modified=[];control=[];count=0
        for old in rows:
            r=copy.deepcopy(old);p=r['period_selection'];f=r.get('period',{}).get('selected_frequency_hz');oldmeta=anchor_metadata(p,f);c=copy.deepcopy(r);c['period_selection']=oldmeta;control.append(c)
            if p['source']=='continued_rotor_support_without_force_update':modified.append(c);continue
            pooled=dict(qualified=bool(old['force'].get('input_u') is not None),frequency_hz=f,selection_source=p.get('pooled_estimator_source'))
            u,q,meta=bank.step(r['t'],r['structure'],r['geometry'].get('rotors',[]),r['rates'],pooled)
            if p['source']!='regional_support_without_motor_frequency' and meta['source']=='anonymous_current_disjoint_partial_f2_sum':
                r['force']['input_u']=u.tolist();r['period_selection']=anchor_metadata(meta,f);count+=1
            else:r['period_selection']=oldmeta
            modified.append(r)
        evidence[n]=dict(anonymous_partial_frames=count)
        for arm,data in [('control',control),('partial',modified)]:
            rr=run(data,EquilibriumGainObserver,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation();vv=[win.step(vis.step(boot.step(r))) for r in rr]
            assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
            if arm=='control':
                expected=[json.loads(s) for s in (ROOT.parent/f'output/equilibrium463_20261111/{n}_equilibrium.jsonl').read_text('utf8').splitlines()]
                delta=max([np.max(abs(np.array(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,expected) if r['valid']] or [0.]);assert delta<1e-7,(n,delta);parity[n]=float(delta)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',evidence[n],flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('control','partial')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            s=score(out/f'{n}_{a}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within31=sum(x<=31 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,control_parity=parity,evidence=evidence,original102_preserved=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
