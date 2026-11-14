"""Bound bootstrap to qualification state, not permanent temporal filtering."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from bootstrap_native_reference import BootstrapNativeReference
from score_rotor164 import score
from extend_inverse321 import references

if __name__=='__main__':
    out=ROOT.parent/'output/reference_bootstrap447_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/reference_bootstrap447_predeclared_20261110.json').read_bytes())
    files=['bootstrap_native_reference.py','tools/evaluate_bootstrap447.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    ns=('13','15','16','11','3','4');paths={};post_warmup_delta=0.
    for mode in ('cached','actual','temporal'):
        for n in ns:
            p=ROOT.parent/(f'output/projection436_fixed_20261110/{n}_qualified.jsonl' if mode=='cached' else f'output/raw_projection437_20261110/recording{n}/predictions.jsonl' if mode=='actual' else f'output/temporal444_20261110/{n}_temporal.jsonl')
            rr=[json.loads(s) for s in p.read_text('utf8').splitlines()];memory=BootstrapNativeReference();vv=[]
            for old in rr:
                r=memory.step(old);assert all(r[k]==old[k] for k in ('coefficient','force','valid','body','native_fov_observation') if k in old)
                if r.get('native_reference_bootstrap',{}).get('original_readout_exact_after_warmup'):
                    assert r['conditional_unknown_force']==old['conditional_unknown_force']
                vv.append(r)
            (out/f'{mode}_{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');paths[mode,n]=p
        print(mode,'all inference frozen',flush=True)
    seed=next(copy.deepcopy(r) for r in rr if r.get('coefficient') and r.get('conditional_unknown_force'))
    seed['body'].update(detected=True,center_provisional=False,native_reference_calibration_allowed=True,bbox=[450,280,562,332],center=[506.,306.],measurement_source='filtered_native_translation');seed['body'].pop('measurement_reference_shift_xy',None);seed['motion']['current_observation']=True;seed['coefficient']['position']=[501.,309.]
    memory=BootstrapNativeReference()
    for i in range(6):
        r=copy.deepcopy(seed);r['t']=100+i/30;r['state']['observations']=i+50
        if i==3:r['body']['bbox']=[450,280,512,320]
        q=memory.step(r)
        if i==3:assert np.max(abs(np.asarray(q['coefficient']['position'])+q['conditional_unknown_force']['position_offset']-[506.,306.]))<1e-8
        if i>=4:assert q['conditional_unknown_force']==r['conditional_unknown_force']
    cases={};pooled={};original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    for mode in ('cached','actual','temporal'):
        pool={a:[] for a in ('control','bootstrap')};cases[mode]={}
        for n in ns:
            ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[mode][n]={}
            for a,p in [('control',paths[mode,n]),('bootstrap',out/f'{mode}_{n}.jsonl')]:
                s=score(p,ref,origins);v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[mode][n][a]=s
            print(mode,n,{a:s['original_available'] for a,s in cases[mode][n].items()},flush=True)
        pooled[mode]={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,source_sha256=hashes,
        controls=dict(early_support_loss_preserves_current_position=True,exact_original_readout_after_five_qualified_refs=True),
        physical_detector_FOV_unchanged=True,no_model_fit=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
