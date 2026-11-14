"""One background-root continuation: evidence-based coordinate hypotheses."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from camera_reference_selection import ReferenceEvidenceWeight,ReferenceMemoryWeight,mix
from physical_publication import evaluate,freeze
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--memory',action='store_true');args=parser.parse_args()
    out=ROOT.parent/('output/background_memory516_20261113' if args.memory else 'output/background_evidence511_20261112');out.mkdir(exist_ok=False)
    files=['camera_reference_selection.py','physical_publication.py','tools/replay_camera_weight511.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,one_rule_all6=True,
        background_likelihood_only=True,equal_static_and_motion_prior_odds=True,
        existing1000_velocity_and_acceleration_priors_reused=True,no_hyperparameter_search_or_fit=True,
        both_branches_same_native_rotor_front=True,all_inference_before_references=True,
        reserved_video_unread=True,FPS_work=False,memory_reuses_old035_force_correlation=args.memory),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    ns=('13','15','16','11','3','4');evidence={}
    for n in ns:
        aa=[json.loads(s) for s in (ROOT.parent/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl').read_text('utf8').splitlines()]
        bb=[json.loads(s) for s in (ROOT.parent/f'output/raw_bg_endpoints509_20261112/recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
        ev=[json.loads(s) for s in (ROOT.parent/f'output/background_translation503_20261112/{n}.jsonl').read_text('utf8').splitlines()]
        gate=ReferenceMemoryWeight() if args.memory else ReferenceEvidenceWeight();rr=[];weights=[]
        for a,b,e in zip(aa,bb,ev):
            info=gate.step(e);r=mix(a,b,info);rr.append(r);weights.append(info['weight'])
            assert r['valid']==a['valid'] and r['body']==a['body'] and r['motion']==a['motion']
            if r['valid']:assert np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(freeze(r,r['t']),[.4])['0.4']))<1e-7
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        evidence[n]=dict(mean_weight=float(np.mean(weights)),median_weight=float(np.median(weights)),weights_gt05=sum(w>.5 for w in weights))
        print(n,'inference frozen',evidence[n],flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('baseline487','camera509','selection')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('baseline487',ROOT.parent/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl'),
                    ('camera509',ROOT.parent/f'output/raw_bg_endpoints509_20261112/recording{n}/predictions.jsonl'),('selection',out/f'{n}.jsonl')]:
            s=score(p,ref,np.round(np.arange(10.8,15.6001,.2),1));values=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids]
            assert len(values)==len(ids);pool[a].extend(values);s['original_available']=dict(n=len(values),mean=float(np.mean(values)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30p5=sum(x<=30.5 for x in v)) for a,v in pool.items()}
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,gate_evidence=evidence,
        original102_preserved=True,no_parameters_fitted=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
