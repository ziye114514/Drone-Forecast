"""One relative-depth projection cue, fixed observer/model and186 teachers."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_scale_response import RotorScaleTrend,increment,apply
from conditional_unknown_force import fit,HORIZON,TAU
from separated_force_response import rate_kernel
from adaptive_force_memory import kernel
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    out=ROOT.parent/'output/rotor_scale435_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/rotor_scale435_predeclared_20261110.json').read_bytes())
    files=['rotor_scale_response.py','tools/evaluate_scale435.py','physical_publication.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    base=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));groups=[];manifest={};counts={}
    for n in base['training_manifest']:
        assert n not in ('13','15','16','11','3','4')
        rows=[json.loads(s) for s in (ROOT.parent/f'output/training400_20261110/recording{n}/predictions.jsonl').read_text('utf8').splitlines()];trend=RotorScaleTrend();contexts={r['t']:trend.observe(r) for r in rows}
        teachers=json.loads((ROOT.parent/f'output/endpoint_training433_20261110/teachers{n}.json').read_text('utf8'));g=[];qualified=0
        for s in teachers:
            X=np.asarray(s['X']);factor=rate_kernel(-1/TAU,0.,s['h'])/rate_kernel(-1/TAU,0.,HORIZON)
            # Original camera basis exactly exposes a_x/a_y in these columns.
            a=np.array([X[0,3],X[1,2]])/kernel(-1/TAU,HORIZON)/factor
            z=contexts[s['t']];feature=increment(a,z['log_scale_rate_s_inv'],s['h']) if z['qualified'] else np.zeros(2);qualified+=bool(z['qualified'])
            augmented=np.column_stack((X,(s['visibility_weight']-base['visibility_context_mean'])*X,s['plane_X']))
            residual=np.asarray(s['y'])-augmented@np.asarray(base['weights'])-(base['future_drive_multiplier']-1)*np.asarray(s['drive_feature'])
            g.append(dict(t=s['t'],X=feature[:,None].tolist(),y=residual.tolist()))
        groups.append(g);manifest[n]=len(g);counts[n]=qualified
    assert sum(manifest.values())==186
    model=fit(groups,'camera',prior=np.zeros(1),bounds=[(0.,1.)]);coefficient=float(model['weights'][0]);(out/'model.json').write_text(json.dumps(dict(model,training_manifest=manifest,qualified_training_scale_origins=counts),indent=2),'utf8');print('COEFFICIENT FROZEN',coefficient,'supported',counts,flush=True)
    ns=('13','15','16','11','3','4');used={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/endpoint432_20261110/{n}_endpoint.jsonl').read_text('utf8').splitlines()];trend=RotorScaleTrend();vv=[];used[n]=0
        for r in rows:
            z=trend.observe(r);q=apply(r,z,coefficient);used[n]+=bool(z['qualified']);assert q['coefficient']==r['coefficient'] and q['force']==r['force'] and q['valid']==r['valid'];vv.append(q)
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');print(n,'inference frozen, qualified scale',used[n],flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('endpoint432','scale435')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('endpoint432',ROOT.parent/f'output/endpoint432_20261110/{n}_endpoint.jsonl'),('scale435',out/f'{n}.jsonl')]:
            r=score(p,ref,origins);v=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);r['same_original_origins']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=r
        print(n,{a:r['same_original_origins'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,source_sha256=hashes,training_manifest=manifest,
        qualified_scale_frames=used,one_global_coefficient=coefficient,all6_recordings_excluded_from_fit=True,FPS_work=False,new_video_validated=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
