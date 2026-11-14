"""One known support-time conversion, never tune delay against these videos."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from window_position_observation import WindowPositionObservation
from physical_publication import evaluate,freeze
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/window_observation449_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/window_observation449_predeclared_20261110.json').read_bytes())
    files=['window_position_observation.py','physical_publication.py','tools/evaluate_window449.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    toy=dict(t=1.,valid=True,coefficient=dict(position=[100.,200.],velocity=[300.,-60.],acceleration=[0.,0.]),
        force=dict(active=False,input_u=None,effective_gain_px_s2=200.,acceleration_correlation_s=.35,acceleration_px_s2=[0.,0.]),normal_turn_forecast=dict(applied=False))
    equal=WindowPositionObservation(33333,33333).step(toy);new=WindowPositionObservation().step(toy)
    assert np.max(abs(np.asarray(evaluate(equal,[.4])['0.4'])-evaluate(toy,[.4])['0.4']))<1e-10
    expected=np.asarray(toy['coefficient']['position'])+np.asarray(toy['coefficient']['velocity'])*(.4+.0066665)
    assert np.max(abs(np.asarray(evaluate(new,[.4])['0.4'])-expected))<1e-10
    frozen=freeze(new,1.);assert np.max(abs(np.asarray(evaluate(frozen,[.5])['0.5'])-evaluate(new,[.5])['0.5']))<1e-10
    ns=('13','15','16','11','3','4');paths={}
    for mode in ('cached','actual','complete'):
        for n in ns:
            p=ROOT.parent/(f'output/projection436_fixed_20261110/{n}_qualified.jsonl' if mode=='cached' else f'output/raw_projection437_20261110/recording{n}/predictions.jsonl' if mode=='actual' else f'output/reference_bootstrap447_20261110/temporal_{n}.jsonl');paths[mode,n]=p
            rr=[json.loads(s) for s in p.read_text('utf8').splitlines()];observer=WindowPositionObservation();vv=[observer.step(r) for r in rr]
            assert all(all(r[k]==s[k] for k in ('t','coefficient','force','valid','body')) for r,s in zip(vv,rr))
            (out/f'{mode}_{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(mode,'all inference frozen',flush=True)
    cases={};pooled={};original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    for mode in ('cached','actual','complete'):
        pool={a:[] for a in ('control','aligned')};cases[mode]={}
        for n in ns:
            ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[mode][n]={}
            for a,p in [('control',paths[mode,n]),('aligned',out/f'{mode}_{n}.jsonl')]:
                s=score(p,ref,origins);v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[mode][n][a]=s
            print(mode,n,{a:s['original_available'] for a,s in cases[mode][n].items()},flush=True)
        pooled[mode]={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,source_sha256=hashes,fixed_time_conversion_s=.0066665,
        controls=dict(equal_windows_exact=True,constant_velocity_exact=True,frozen_publication_consistent=True),
        reference_target_timestamps_unchanged=True,no_lag_fit_or_search=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
