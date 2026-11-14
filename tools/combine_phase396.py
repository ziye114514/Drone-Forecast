"""Cumulative accuracy test: current phase-validated geometry, native frequency.

Reuses existing238/244 normal refinement, not a new idea. Native spectral
amplitudes/identities remain; only a qualified common plane rotates their
vectors. A transported311 shape is NOT substituted for a current phase refit.
All6 sources, original102 origins; source-excluded learned response is optional
matched arm, never selected by recording. No new thresholds or parameter scan.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from adaptive_response_observer import AdaptiveResponseObserver
from inverse_normal_response import InverseNormalResponse
from anisotropic_response import AnisotropicResponse
from conditional_unknown_force import teachers,fit,ConditionalUnknownForce
from joint_force_adapter import adapt
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    source=ROOT.parent/'output/phase395_20261110';assert (source/'completion.json').exists(),'Wait for all6 current phase passes to finish'
    out=ROOT.parent/'output/combined396_20261110';out.mkdir(exist_ok=False)
    files=['joint_force_adapter.py','conditional_unknown_force.py','tools/combine_phase396.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    data={};counts={}
    for n in ('13','15','16','11','3','4'):
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()];periods=[json.loads(s) for s in (source/f'{n}_periods.jsonl').read_text('utf8').splitlines()];assert len(rr)==len(periods)
        m=AdaptiveResponseObserver();inv=InverseNormalResponse();disk=AnisotropicResponse();values=[];refined=0
        for old,period in zip(rr,periods):
            assert abs(old['t']-period['t'])<1e-9
            r=copy.deepcopy(old);_,_,st=adapt(r,period,refine_normal=True);r['structure']=st;pm=copy.deepcopy(r['period_selection']);u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
            if st.get('normal_refinement_source'):
                refined+=1;n0=np.asarray(old['structure'].get('axis_xy') or [],float);n1=np.asarray(st['axis_xy'])
                if n0.shape==(2,) and n0@n0>1e-8 and u is not None:
                    if n0[1]>0:n0=-n0
                    parts={k:(float(n0@np.asarray(v)/(n0@n0))*n1).tolist() for k,v in pm.get('observed_components',{}).items()}
                    if parts:pm['observed_components']=parts;u=np.sum(list(parts.values()),axis=0)
                st['phase_geometry_is_model_conditioned_not_independent_COM_GT']=True
            fresh=m.x is None or m.seen is not None and r['t']-m.seen>.5
            if fresh and r['body'].get('detected') and not r['body'].get('center_provisional'):m.two_point_init=bool(r['state'].get('two_point_initialization_selected_by_partial_entry'))
            c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=st)
            r.update(coefficient=c,valid=v,force=f,state=s,period_selection=pm,normal_turn_forecast=s.get('normal_turn_forecast'));r=disk.step(inv.step(r));r['physical']=evaluate(r,[.1,.2,.4]) if c else {};values.append(r);assert v==old['valid']
        data[n]=values;counts[n]=refined;(out/f'{n}_geometry.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values),'utf8');print(n,'geometry inference frozen',refined,flush=True)
    samples={n:teachers(rr) for n,rr in data.items()};models={}
    for n,rr in data.items():
        model=fit([s for k,s in samples.items() if k!=n]);models[n]=model;pred=ConditionalUnknownForce(model);values=[pred.step(r) for r in rr]
        (out/f'{n}_learned.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values),'utf8')
    (out/'models.json').write_text(json.dumps(dict(LOO=models,shared_for_future=fit(list(samples.values())),teacher_counts={n:len(s) for n,s in samples.items()}),indent=2),'utf8')
    cases={};pool={a:[] for a in ('control','learned382','geometry','learned')}
    for n in data:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),learned382=score(ROOT.parent/f'output/unknown382_20261110/{n}_disk.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in ('geometry','learned')})
        for a,c in cases[n].items():pool[a].extend(r['error_px'] for r in c['records'] if r['valid'])
        print(n,{a:c['error'] for a,c in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,refined_direction_frames=counts,source_sha256=hashes,all_sources_exposed_development=True,manual_references_not_used_for_training=True,CPU_accuracy_only_not30FPS=True,one_algorithm_all_sources=True),indent=2),'utf8');print(pooled,flush=True)
