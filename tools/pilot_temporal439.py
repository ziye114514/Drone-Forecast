"""Single pain-point pilot on actual packet times, old control exact."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
from functools import partial
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_cuda_worker import CudaRotorFrontWorker
from observed_rotor_predictor import ProjectionQualifiedObservedPredictor
from temporal_rotor_birth import TemporalBirthContinuation,distinct_votes
from check_inverse322 import model
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score
from run_rotor_only import serial


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recording',default='11');parser.add_argument('--batch',default='temporal_birth439_fixed_20261110');parser.add_argument('--seed-motion-search',action='store_true');args=parser.parse_args();n=args.recording
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False)
    protocol=json.loads((ROOT.parent/'output/checkpoints/temporal_birth439_predeclared_20261110.json').read_text('utf8'))
    protocol.update(recording=n,seed_motion_search_only=args.seed_motion_search,physical_velocity_not_initialized_from_tentative_pair=True)
    if args.seed_motion_search:
        protocol['followup_declaration']=json.loads((ROOT.parent/'output/checkpoints/birth_search441_predeclared_20261110.json').read_text('utf8'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    files=['temporal_rotor_birth.py','tools/pilot_temporal439.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    mask=np.ones((20,30),bool);xy=np.array([[4,4],[5,4],[6,4],[7,4]])
    assert distinct_votes(xy,xy,[0,0],mask)==4
    assert distinct_votes(xy,xy+[1,0],[1,0],mask)==4
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    mm={a:model(worker,predictor_type=partial(ProjectionQualifiedObservedPredictor,response_model=ROOT/'models/native_response_plane_v1.json'),inverse_normal_response=True,anisotropic_response=True) for a in ('control','temporal')}
    mm['temporal'].front.periodic_continuation=TemporalBirthContinuation(seed_motion_search=args.seed_motion_search);expected=[json.loads(s) for s in (ROOT.parent/f'output/raw_projection437_20261110/recording{n}/predictions.jsonl').read_text('utf8').splitlines()];proofs=[];parity=0.
    cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/frozen_tangent287_20261108/cache' if n=='3' else 'output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}')
    manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames']
    try:
        for i,old in enumerate(expected):
            t=old['t'];e=np.load(cache/manifest[i]['file']) if n in ('3','4') else window(cache,t)
            # The cooperative RAW reader's final packet is only60us. Keep
            # exactly the actual packet interval, not a full rolling window.
            begin=expected[i-1]['t'] if i else 10.
            e=e[e['t']>=round(begin*1e6)];features=worker.prepare_uncached(e,t)
            for a,m in mm.items():
                worker.cached=(t,features);r,frame=m.step(e[:0],t);assert r['counts']['events']==old['counts']['events']
                if a=='control':assert r['valid']==old['valid']
                if a=='control' and r['valid']:
                    d=float(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(old,[.4])['0.4'])));parity=max(parity,d);assert d<1e-7
                proof=(r.get('periodic_track_continuation') or {}).get('tentative_birth')
                if a=='temporal' and proof:proofs.append(dict(t=t,**proof))
                if a=='temporal' and proof and proof.get('used'):
                    assert not r['force']['active'] and r['period_selection']['observed_components']=={}
                    assert r['period_selection']['motor_frequency_bank_updated'] is False
                    if r['coefficient'] and r['state'].get('observations')==1:assert np.linalg.norm(r['coefficient']['velocity'])<1e-8
                with (out/f'{a}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
            if i%60==0:print(i,'exactRAW control',parity,'newbirths',sum(q['used'] for q in proofs),flush=True)
    finally:
        for m in mm.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close()
    (out/'birth_proofs.json').write_text(json.dumps(proofs,indent=2),'utf8')
    ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
    scores={a:score(out/f'{a}.jsonl',ref,origins) for a in mm}
    (out/'results.json').write_text(json.dumps(dict(scores=scores,source_sha256=hashes,actualRAW_control_parity_px=parity,births=sum(q['used'] for q in proofs),
        distinct_vote_controls_passed=True,single_source_pilot_only=True,not_promoted=True,FPS_work=False),indent=2),'utf8');print('RESULT',{a:s['error'] for a,s in scores.items()},'births',sum(q['used'] for q in proofs),flush=True)
