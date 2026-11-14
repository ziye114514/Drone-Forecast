"""One causal amplitude fit across all existing sources, no parameter search."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from integral_normal_response import IntegralNormalResponse,fit_response
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='integral363_20261109');parser.add_argument('--span',type=float,default=.1);parser.add_argument('--min-factors',type=int,default=8);a=parser.parse_args()
    span=a.span;minimum=a.min_factors
    out=ROOT.parent/'output'/a.batch;out.mkdir(exist_ok=False)
    files=['integral_normal_response.py','tools/probe_integral363.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    # Known force, unknown external bias, varying normals, constant velocity
    # integral independent of normal. Identifiability and offset invariance.
    rng=np.random.default_rng(363);samples=[];A=1400.;b=np.array([130.,-210.]);beta=1/.35
    for i in range(16):
        dn=rng.normal(size=2)*.1;dv=rng.normal(size=2)*20;dt=.1
        samples.append((i*.2,dn,dv,dt,A*dn+b*dt-beta*dv,np.eye(2)))
    a=fit_response(samples);shift=np.array([-300.,70.]);shifted=[(t,n,v,dt,y+shift*dt,R) for t,n,v,dt,y,R in samples];z=fit_response(shifted)
    assert abs(a['amplitude']-A)<1e-6 and abs(z['amplitude']-A)<1e-6,(a,z)
    # Constant normal must not become collective-force information through a
    # gravity/wind zero-mean prior.
    fixed=[(t,np.array([0.,-dt]),v,dt,np.array([0.,-A*dt])+b*dt-beta*v,R) for t,n,v,dt,y,R in samples]
    degenerate=fit_response(fixed);assert not degenerate['qualified'],degenerate
    counts={};models={};results={}
    for n in ('13','15','16','11','3','4'):
        rr=[json.loads(s) for s in (ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl').read_text('utf8').splitlines()];m=IntegralNormalResponse(span,minimum);outputs=[]
        for row in rr:
            r=m.step(row);assert r['coefficient']==row['coefficient'];assert r['valid']==row['valid'];r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs.append(r)
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8');counts[n]=dict(applied=sum(r['integral_normal_response']['applied'] for r in outputs),max_factors=max(r['integral_normal_response'].get('factors',0) for r in outputs));print(n,'frozen',counts[n],flush=True)
    for n in counts:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]=dict(control=score(ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl',ref,origins),candidate=score(out/f'{n}.jsonl',ref,origins));print(n,{k:v['error'] for k,v in results[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=results,counts=counts,span_s=span,min_factors=minimum,known_amplitude_recovered=a,external_offset_invariant=z,constant_normal_not_identified=degenerate,source_sha256=hashes,NN_used=False,no_parameter_search=True,RAW_FPS_measured=False),indent=2),'utf8')
