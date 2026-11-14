"""One fixed temporal-birth observer across all6 exact actual packet streams."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
from functools import partial
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from rotor_cuda_worker import CudaRotorFrontWorker
from observed_rotor_predictor import ProjectionQualifiedObservedPredictor
from temporal_rotor_birth import TemporalBirthContinuation
from check_inverse322 import model
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score
from run_rotor_only import serial,SOURCE_FILES


if __name__=='__main__':
    out=ROOT.parent/'output/temporal444_20261110';out.mkdir(exist_ok=False);images=out/'birth_images';images.mkdir()
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/temporal444_predeclared_20261110.json').read_bytes())
    files=sorted(set(SOURCE_FILES+['temporal_rotor_birth.py','tools/evaluate_temporal444.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    ns=('13','15','16','11','3','4');progress={};births={}
    try:
        for n in ns:
            expected=[json.loads(s) for s in (ROOT.parent/f'output/raw_projection437_20261110/recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
            cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/frozen_tangent287_20261108/cache' if n=='3' else 'output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}')
            manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames'];births[n]=[];parity=0.;counts={a:0 for a in ('control','temporal')}
            mm={a:model(worker,predictor_type=partial(ProjectionQualifiedObservedPredictor,response_model=ROOT/'models/native_response_plane_v1.json'),inverse_normal_response=True,anisotropic_response=True) for a in counts}
            mm['temporal'].front.periodic_continuation=TemporalBirthContinuation(seed_motion_search=True)
            for i,old in enumerate(expected):
                t=old['t'];e=np.load(cache/manifest[i]['file']) if n in ('3','4') else window(cache,t)
                begin=expected[i-1]['t'] if i else 10.;e=e[e['t']>=round(begin*1e6)]
                features=worker.prepare_uncached(e,t)
                for a,m in mm.items():
                    worker.cached=(t,features);r,frame=m.step(e[:0],t);assert r['counts']['events']==old['counts']['events'],(n,i,a,'input packets differ')
                    counts[a]+=int(r['valid'])
                    if a=='control':
                        assert r['valid']==old['valid']
                        if r['valid']:
                            d=float(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(old,[.4])['0.4'])));parity=max(parity,d);assert d<1e-7,(n,i,d)
                    proof=(r.get('periodic_track_continuation') or {}).get('tentative_birth') or {}
                    if a=='temporal' and proof.get('used'):
                        assert not r['force']['active'] and r['period_selection']['observed_components']=={}
                        assert r['period_selection']['motor_frequency_bank_updated'] is False
                        if r['coefficient'] and r['state'].get('observations')==1:assert np.linalg.norm(r['coefficient']['velocity'])<1e-8
                        births[n].append(dict(t=t,**proof))
                        count=frame['on']+frame['off'];im=Image.fromarray(np.uint8(255-np.minimum(count*20,255))).convert('RGB');draw=ImageDraw.Draw(im)
                        if r['body'].get('bbox'):draw.rectangle(r['body']['bbox'],outline='#2166dc',width=3)
                        im.resize((640,360)).save(images/f'{n}_{t:.6f}.png')
                    with (out/f'{n}_{a}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
                if i%60==0:print(n,i,'exact control',parity,'births',len(births[n]),flush=True)
            progress[n]=dict(packets=len(expected),old_control_parity_px=parity,valid_forecasts=counts,births=len(births[n]))
            (out/'progress.json').write_text(json.dumps(progress,indent=2),'utf8');(out/'birth_proofs.json').write_text(json.dumps(births,indent=2),'utf8')
            for m in mm.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
            worker.cached=None;print(n,'all inference frozen',flush=True)
    finally:worker.cached=None;worker.close()
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('control','temporal')};common={a:[] for a in pool};coverage={};new_origins=[];missing=[]
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']}
        cases[n]={a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in pool};rr={a:{q['t']:q for q in s['records'] if q['valid']} for a,s in cases[n].items()};coverage[n]={a:sum(t in rr[a] for t in ids) for a in pool}
        for a in pool:pool[a].extend(rr[a][t]['error_px'] for t in ids if t in rr[a])
        for t in ids&rr['control'].keys()&rr['temporal'].keys():
            for a in common:common[a].append(rr[a][t]['error_px'])
        new_origins.extend(dict(recording=n,**rr['temporal'][t]) for t in ids&rr['temporal'].keys()-rr['control'].keys())
        missing.extend(dict(recording=n,t=t) for t in ids-rr['temporal'].keys())
        print(n,'original coverage',coverage[n],'all available mean',{a:s['error']['mean'] for a,s in cases[n].items()},flush=True)
    stats=lambda d:{a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in d.items()}
    result=dict(cases=cases,pooled_original_available=stats(pool),pooled_common_original=stats(common),original_coverage=coverage,
        newly_recovered_originals=new_origins,missing_originals=missing,births=births,source_sha256=hashes,
        exact_actualRAW_control=progress,no_model_fit=True,new_video_validated=False,FPS_work=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print('POOLED',result['pooled_original_available'],'COMMON',result['pooled_common_original'],flush=True)
