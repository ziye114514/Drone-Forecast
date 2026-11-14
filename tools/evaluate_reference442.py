"""Fixed medoid of reference packets, current dynamics exactly unchanged."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from stable_native_reference import StableNativeReference,medoid
from physical_publication import evaluate
from score_rotor164 import score
from extend_inverse321 import references


if __name__=='__main__':
    out=ROOT.parent/'output/reference_consistency442_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/reference_consistency442_predeclared_20261110.json').read_bytes())
    files=['stable_native_reference.py','tools/evaluate_reference442.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    packets=[dict(offset=[0.,0.],extent=[100.,40.]),dict(offset=[.5,.1],extent=[101.,40.]),dict(offset=[-25.,-6.],extent=[50.,28.])]
    chosen=medoid(packets);assert chosen in (0,1)
    angle=.731;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    turned=[dict(p,offset=(R@p['offset']).tolist()) for p in packets]
    assert medoid(turned)==chosen
    ns=('13','15','16','11','3','4');arms={};used={}
    for mode in ('cached','actual'):
        for n in ns:
            p=ROOT.parent/(f'output/projection436_fixed_20261110/{n}_qualified.jsonl' if mode=='cached' else f'output/raw_projection437_20261110/recording{n}/predictions.jsonl')
            rr=[json.loads(s) for s in p.read_text('utf8').splitlines()];ref=StableNativeReference();vv=[]
            for old in rr:
                r=ref.step(old);assert r['coefficient']==old['coefficient'] and r['force']==old['force'] and r['valid']==old['valid'] and r['body']==old['body'];vv.append(r)
            (out/f'{mode}_{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');arms[mode,n]=p
        print(mode,'all inference frozen',flush=True)
    # The new-birth diagnosis is separate, not spliced into the aggregate.
    ref=StableNativeReference();rr=[json.loads(s) for s in (ROOT.parent/'output/temporal_search441_20261110/temporal.jsonl').read_text('utf8').splitlines()]
    vv=[ref.step(r) for r in rr];(out/'birth441_11.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
    seed=next(copy.deepcopy(r) for r in rr if r.get('coefficient') and r.get('conditional_unknown_force') and r['body'].get('bbox') and r['body']['bbox'][0]>0)
    seed['body']['native_reference_calibration_allowed']=True;seed['body'].pop('measurement_reference_shift_xy',None);seed['body']['bbox']=[450,280,562,332];seed['coefficient']['position']=[506.,306.];seed['coefficient']['velocity']=[0.,0.];seed['coefficient']['acceleration']=[0.,0.]
    memory=StableNativeReference();values=[]
    for i in range(5):
        r=copy.deepcopy(seed);r['t']=100+i/30;r['state']['observations']=i+50
        if i==3:r['body']['bbox']=[450,280,512,320]
        values.append(memory.step(r)['stable_native_reference']['offset'])
    assert np.max(abs(np.asarray(values)))<1e-8
    rebased=copy.deepcopy(r);rebased['t']+=1/30;rebased['body']['native_reference_calibration_allowed']=False;rebased['body']['measurement_reference_shift_xy']=[20.,-10.];rebased['coefficient']['position']=[526.,296.]
    q=memory.step(rebased);assert np.max(abs(np.asarray(q['coefficient']['position'])+q['stable_native_reference']['offset']-[506.,306.]))<1e-8
    cases={};pooled={};original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    for mode in ('cached','actual'):
        pool={a:[] for a in ('control','stable')};cases[mode]={}
        for n in ns:
            refpath=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[mode][n]={}
            for a,p in [('control',arms[mode,n]),('stable',out/f'{mode}_{n}.jsonl')]:
                scoredata=score(p,refpath,origins);v=[q['error_px'] for q in scoredata['records'] if q['valid'] and q['t'] in ids];pool[a].extend(v);scoredata['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[mode][n][a]=scoredata
            print(mode,n,{a:s['original_available'] for a,s in cases[mode][n].items()},flush=True)
        pooled[mode]={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    birth={a:score(p,references('11'),np.round(np.arange(10.8,15.6001,.2),1)) for a,p in [('control441',ROOT.parent/'output/temporal_search441_20261110/temporal.jsonl'),('stable442',out/'birth441_11.jsonl')]}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,birth441_diagnosis_separate=birth,
        source_sha256=hashes,controls=dict(isolated_support_loss_no_output_shift=True,image_rotation_selection_invariant=True,pure_reference_rebase_output_invariant=True),
        physical_state_force_detector_unchanged=True,no_fit=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,'BIRTH',{a:s['error'] for a,s in birth.items()},flush=True)
