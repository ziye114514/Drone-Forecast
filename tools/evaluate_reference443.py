"""Only geometrically demonstrated loss may withhold a reference update."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from material_reference_qualification import MaterialReferenceQualification
from physical_publication import evaluate
from score_rotor164 import score
from extend_inverse321 import references


if __name__=='__main__':
    out=ROOT.parent/'output/reference_contraction443_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/reference_contraction443_predeclared_20261110.json').read_bytes())
    files=['material_reference_qualification.py','tools/evaluate_reference443.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    ns=('13','15','16','11','3','4');paths={}
    for mode in ('cached','actual','birth441'):
        for n in (('11',) if mode=='birth441' else ns):
            p=ROOT.parent/(f'output/projection436_fixed_20261110/{n}_qualified.jsonl' if mode=='cached' else f'output/raw_projection437_20261110/recording{n}/predictions.jsonl' if mode=='actual' else 'output/temporal_search441_20261110/temporal.jsonl')
            rr=[json.loads(s) for s in p.read_text('utf8').splitlines()];memory=MaterialReferenceQualification();vv=[]
            for old in rr:
                r=memory.step(old);assert all(r[k]==old[k] for k in ('coefficient','force','valid','body','native_fov_observation') if k in old);vv.append(r)
            (out/f'{mode}_{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');paths[mode,n]=p
        print(mode,'all inference frozen',flush=True)
    # Mechanism uses the same interior fixed material point for all frames.
    seed=next(copy.deepcopy(r) for r in rr if r.get('coefficient') and r.get('conditional_unknown_force'))
    seed['body'].update(detected=True,center_provisional=False,native_reference_calibration_allowed=True,bbox=[450,280,562,332]);seed['body'].pop('measurement_reference_shift_xy',None);seed['coefficient']['position']=[506.,306.]
    memory=MaterialReferenceQualification();offsets=[];flags=[]
    for i in range(5):
        r=copy.deepcopy(seed);r['t']=100+i/30;r['state']['observations']=i+50
        if i==3:r['body']['bbox']=[450,280,512,320]
        q=memory.step(r);offsets.append(q['material_reference_qualification']['held_offset_xy']);flags.append(q['material_reference_qualification']['current_complete_reference_withheld_for_support_loss'])
    assert np.max(abs(np.asarray(offsets)))<1e-8 and flags==[False,False,False,True,False]
    r=copy.deepcopy(seed);r['t']=101.;r['state']['observations']=100;r['body']['native_reference_calibration_allowed']=False;r['body']['measurement_reference_shift_xy']=[20.,-10.];r['coefficient']['position']=[526.,296.]
    q=memory.step(r);assert np.max(abs(np.asarray(q['coefficient']['position'])+q['material_reference_qualification']['held_offset_xy']-[506.,306.]))<1e-8
    cases={};pooled={};original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    for mode in ('cached','actual'):
        pool={a:[] for a in ('control','qualified')};cases[mode]={}
        for n in ns:
            ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[mode][n]={}
            for a,p in [('control',paths[mode,n]),('qualified',out/f'{mode}_{n}.jsonl')]:
                s=score(p,ref,origins);v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[mode][n][a]=s
            print(mode,n,{a:s['original_available'] for a,s in cases[mode][n].items()},flush=True)
        pooled[mode]={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    birth={a:score(p,references('11'),np.round(np.arange(10.8,15.6001,.2),1)) for a,p in [('control441',paths['birth441','11']),('qualified443',out/'birth441_11.jsonl')]}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,birth441_diagnosis_separate=birth,source_sha256=hashes,
        controls=dict(support_loss_no_shift=True,restored_full_reference_accepted=True,reference_gauge_invariant=True),
        no_fit=True,current_physical_detector_FOV_unchanged=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,'BIRTH',{a:s['error'] for a,s in birth.items()},flush=True)
