"""No per-video selection: same causal learning update on every source."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from matured_unknown_force import MaturedUnknownForce
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/matured384_20261110';out.mkdir(exist_ok=False)
    files=['conditional_unknown_force.py','matured_unknown_force.py','physical_publication.py','tools/probe_matured384.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    models=json.loads((ROOT.parent/'output/unknown382_20261110/models.json').read_text('utf8'))['LOO'];arms=('disk','camera','no_future_plane');counts={}
    for n in models:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()];counts[n]={}
        for arm in arms:
            m=MaturedUnknownForce(models[n][arm]);outputs=[m.step(r) for r in rows]
            assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] for r,old in zip(outputs,rows))
            assert all((r.get('conditional_unknown_force') or {}).get('causal_native_adaptation',{}).get('newest_target_t',r['t']) is None or (r.get('conditional_unknown_force') or {}).get('causal_native_adaptation',{}).get('newest_target_t',r['t'])<=r['t'] for r in outputs)
            counts[n][arm]=len(m.samples)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8')
        print(n,'inference frozen',counts[n],flush=True)
    cases={}
    for n in models:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in arms});print(n,{a:v['error'] for a,v in cases[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,causal_samples=counts,source_sha256=hashes,all_inference_before_manual_reference_reads=True,all_sources_exposed_development=True,causal_adaptation_is_not_target_source_offline_training=True,no_NN=True),indent=2),'utf8')
