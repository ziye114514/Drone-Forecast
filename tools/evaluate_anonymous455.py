"""One current regional-input change; no fit, labels or per-video switches."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from anonymous_rotor_group import AnonymousRotorGroup
from diffuse_rotor_startup import DiffuseStartupObserver
from replay_endpoint432 import run
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate

if __name__=='__main__':
    out=ROOT.parent/'output/anonymous_group456_20261110';out.mkdir(exist_ok=False)
    files=['anonymous_rotor_group.py','tools/evaluate_anonymous455.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,no_fit_or_threshold_search=True,
        old_individual_slots_priority=True,hidden_rotors_not_filled=True,all_inference_before_references=True,
        anonymous_groups_not_verified_motor_count=True,old_model_unchanged=True),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');evidence={};parity={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/accuracy451_20261110/{n}.jsonl').read_text('utf8').splitlines()];bank=AnonymousRotorGroup();modified=[];count=0
        control=run(rows,DiffuseStartupObserver,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation()
        control=[win.step(vis.step(boot.step(r))) for r in control]
        delta=max([np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(control,rows) if r['valid']] or [0.]);assert delta<1e-7,(n,delta);parity[n]=float(delta)
        for row in rows:
            r=copy.deepcopy(row);p=r['period_selection'];pooled=dict(qualified=bool(row['force'].get('input_u') is not None),frequency_hz=row.get('period',{}).get('selected_frequency_hz'),selection_source=p.get('pooled_estimator_source'))
            if p['source']=='continued_rotor_support_without_force_update':modified.append(r);continue
            u,q,meta=bank.step(r['t'],r['structure'],r['geometry'].get('rotors',[]),r['rates'],pooled)
            if p['source']!='regional_support_without_motor_frequency' and meta['source']=='anonymous_current_disjoint_partial_f2_sum':r['force']['input_u']=u.tolist();r['period_selection']=meta;count+=1
            # Existing input exactly preserved whenever the new branch abstains.
            modified.append(r)
        rr=run(modified,DiffuseStartupObserver,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation();vv=[win.step(vis.step(boot.step(r))) for r in rr]
        assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        evidence[n]=dict(anonymous_partial_input_frames=count,gain_updates=sum(r['force'].get('gain_learned',False) for r in vv));print(n,'inference frozen',evidence[n],flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ['accuracy451','anonymous455']}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('accuracy451',ROOT.parent/f'output/accuracy451_20261110/{n}.jsonl'),('anonymous455',out/f'{n}.jsonl')]:
            s=score(p,ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,evidence=evidence,control_parity=parity,original102_preserved=True,new_video_validated=False,FPS_work=False,source_sha256=hashes,not_actual_RAW_factory_yet=True),indent=2),'utf8');print('POOLED',pooled,flush=True)
