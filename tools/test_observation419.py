"""Paired current-observation-role repair, one unchanged native model all six."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
from functools import partial
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from check_inverse322 import model
from run_rotor_only import SOURCE_FILES,serial
from learned_rotor_predictor import LearnedRotorPredictor
from observed_rotor_predictor import ObservedRotorPredictor,PriorConsistentObservedPredictor,VisibleObservedPredictor,BoundedObservedPredictor
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recordings',nargs='+',default=['13','4','15','16','11','3']);parser.add_argument('--batch',default='observation419_20261110');parser.add_argument('--consistent-observation-prior',action='store_true');parser.add_argument('--visible-envelope',action='store_true');parser.add_argument('--bounded-period-gap',action='store_true');parser.add_argument('--declaration',type=Path,default=ROOT.parent/'output/checkpoints/observation419_predeclared_20261110.json');args=parser.parse_args()
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False);files=sorted(set(SOURCE_FILES+['tools/test_observation419.py','tools/check_inverse322.py']))
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    declaration=args.declaration;(out/'protocol.json').write_bytes(declaration.read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    evidence={}
    try:
        for n in args.recordings:
            expected=[json.loads(s) for s in (ROOT.parent/f'output/combined410_20261110/{n}_joint.jsonl').read_text('utf8').splitlines()]
            cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/frozen_tangent287_20261108/cache' if n=='3' else 'output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}')
            manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames'];path=ROOT/'models/native_response_plane_v1.json'
            candidate=BoundedObservedPredictor if args.bounded_period_gap else VisibleObservedPredictor if args.visible_envelope else PriorConsistentObservedPredictor if args.consistent_observation_prior else ObservedRotorPredictor
            mm={a:model(worker,predictor_type=partial(cls,response_model=path),inverse_normal_response=True,anisotropic_response=True) for a,cls in [('control',LearnedRotorPredictor),('observed',candidate)]}
            parity=0.;suffix=0.;used=0;withheld=0;candidate_parity=0.
            candidate_expected=[json.loads(s) for s in (ROOT.parent/f'output/visible421_20261110/{n}_projected420.jsonl').read_text('utf8').splitlines()] if args.visible_envelope and not args.bounded_period_gap else None
            for i,old in enumerate(expected):
                t=old['t'];e=np.load(cache/manifest[i]['file']) if n in ('3','4') else window(cache,t);features=worker.prepare_uncached(e,t)
                for a,m in mm.items():
                    worker.cached=(t,features);r,frame=m.step(e[:0],t);frame=None;assert r['counts']['events']==old['counts']['events']
                    if a=='control' and t<=15.600001:
                        assert r['valid']==old['valid'] and bool(r['coefficient'])==bool(old['coefficient'])
                        if r['coefficient']:
                            d=max(float(np.max(abs(np.asarray(p)-old['physical'][h]))) for h,p in r['physical'].items());parity=max(parity,d)
                    elif a=='control' and r['coefficient'] and old['coefficient']:
                        suffix=max(suffix,max(float(np.max(abs(np.asarray(p)-old['physical'][h]))) for h,p in r['physical'].items()))
                    if a=='observed' and (r.get('periodic_track_continuation') or {}).get('used'):
                        used+=1;assert r['period_selection']['observed_components']=={} and r['force'].get('input_u') is None
                        assert r['period_selection']['motor_frequency_bank_updated'] is False;withheld+=1
                        gap=(r.get('periodic_track_continuation') or {}).get('period_gap_identity_age_s')
                        if gap is not None:
                            assert 0<gap<=.12+1e-8 and not r['periodic_track_continuation']['current_strict_period_refresh']
                            assert r['counts']['period_pixels']==0
                    if a=='observed' and candidate_expected is not None and r['coefficient']:
                        candidate_parity=max(candidate_parity,max(float(np.max(abs(np.asarray(p)-candidate_expected[i]['physical'][h]))) for h,p in r['physical'].items()))
                    with (out/f'{n}_{a}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
                if i%60==0:print(n,i,'baseline parity',parity,'continued_current',used,flush=True)
            assert parity<1e-7,(n,parity)
            assert candidate_parity<1e-7,(n,candidate_parity)
            evidence[n]=dict(scored_prefix_baseline_parity_px=parity,suffix_parity_px=suffix,current_observation_continuations=used,motor_bank_updates_withheld=withheld,integrated_candidate_parity_px=candidate_parity)
            (out/'progress.json').write_text(json.dumps(evidence,indent=2),'utf8')
            for m in mm.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
            worker.cached=None;print(n,'inference frozen',flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    finally:worker.cached=None;worker.close();configure_native(None)
    cases={};pool={a:[] for a in ('control','observed')};extra={a:[] for a in pool};diagnostics={}
    for n in args.recordings:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]={a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in pool};original={q['t'] for q in cases[n]['control']['records'] if q['valid']};diagnostics[n]={}
        for a in pool:
            sc=cases[n][a];common=[q['error_px'] for q in sc['records'] if q['valid'] and q['t'] in original];pool[a].extend(common);extra[a].extend(q['error_px'] for q in sc['records'] if q['valid'] and q['t'] not in original)
            rr=[json.loads(s) for s in (out/f'{n}_{a}.jsonl').read_text('utf8').splitlines()]
            diagnostics[n][a]=dict(current_regions=sum(bool(r['body'].get('detected')) for r in rr),current_observations=sum(bool(r['motion'].get('current_observation')) for r in rr),valid_forecasts=sum(bool(r['valid']) for r in rr),same_origins=dict(n=len(common),mean=float(np.mean(common)),p95=float(np.percentile(common,95))))
        print(n,diagnostics[n],flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within30=sum(x<=30 for x in v),extra_origins=len(extra[a]),extra_mean=float(np.mean(extra[a])) if extra[a] else None) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,diagnostics=diagnostics,factory_evidence=evidence,source_sha256=hashes,one_global_model_all_sources=True,new_model_fit=False,NN_used=False,FPS_work_performed=False,independent_new_video_validated=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
