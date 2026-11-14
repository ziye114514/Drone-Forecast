"""Same498 hypothesis; correct the fixed-image readout coordinate contract."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_background498 import compensated
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--supported',action='store_true');parser.add_argument('--translation-only',action='store_true');parser.add_argument('--shared-endpoints',action='store_true');args=parser.parse_args()
    if args.shared_endpoints and not args.translation_only:parser.error('Shared-background test follows the fixed translation503 evidence')
    out=ROOT.parent/('output/background_endpoints506d_20261112' if args.shared_endpoints else 'output/background_translation504_20261112' if args.translation_only else 'output/background_supported502_20261112' if args.supported else 'output/background_image500_20261112');out.mkdir(exist_ok=False)
    evidence_folder='background_translation503_20261112' if args.translation_only else 'background_supported501_20261112' if args.supported else 'background_reference497_20261112'
    files=['background_relative_force.py','physical_publication.py','tools/replay_background498.py','tools/replay_background500.py','background_endpoint_observer.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,
        difference_from498='existing fixed-image calibration now consumes image-coordinate v/a; internal force remains background-relative',
        same_background_evidence_and_uncertainty=True,background_reference_support=args.supported,translation_only=args.translation_only,shared_endpoints=args.shared_endpoints,no_model_fit_or_parameter_search=True,
        unchanged_camera035_s=True,all_inference_before_references=True,reserved_video_unread=True),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    ns=('13','15','16','11','3','4');model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));parity={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl').read_text('utf8').splitlines()]
        ev=[json.loads(s) for s in (ROOT.parent/f'output/{evidence_folder}/{n}.jsonl').read_text('utf8').splitlines()]
        zero=[dict(t=r['t'],proof={},pair=None) for r in rows];zz=compensated(rows,zero,model,False,image_calibration=True,shared_endpoints=args.shared_endpoints)
        d=max(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(zz,rows) if r['valid'])
        assert d<1e-7,(n,d);parity[n]=float(d)
        for a,rr in [('control',rows),('noise_only',compensated(rows,ev,model,True,image_calibration=True,shared_endpoints=args.shared_endpoints)),
                     ('reference',compensated(rows,ev,model,False,image_calibration=True,shared_endpoints=args.shared_endpoints))]:
            (out/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        if args.shared_endpoints:
            rr=compensated(rows,ev,model,False,image_calibration=True,preserve_foreground_floor=True)
            (out/f'{n}_independent_floor.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        print(n,'inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in (('control','noise_only','reference','independent_floor') if args.shared_endpoints else ('control','noise_only','reference'))}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            s=score(out/f'{n}_{a}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1))
            v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids)
            pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30p5=sum(x<=30.5 for x in v)) for a,v in pool.items()}
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,no_background_control_parity=parity,
        original102_preserved=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
