"""Reuse stage22 exact endpoint scoring for predeclared robustness ablations."""
from pathlib import Path
import argparse,datetime,hashlib,json,importlib.util,time
import numpy as np,scipy
from robust23 import ROOT,PARAMS,MODELS,Predictor
from checks23 import run as checks

spec=importlib.util.spec_from_file_location('score22_reused',ROOT/'research/stage22/study22.py')
score22=importlib.util.module_from_spec(spec)
# The frozen scorer imports predict22 by module name.
import sys
sys.path.append(str(ROOT/'research/stage22'));spec.loader.exec_module(score22)
INPUT=score22.INPUT;PREVIOUS=ROOT.parent/'output/stage22_sparse_endpoint_20261018'
OUT=ROOT.parent/'output/stage23_robust_acceleration_20261018'
CASES=score22.CASES;FRONTENDS=score22.FRONTENDS;load=score22.load;save=score22.save;sha=score22.sha


def sources():
    ff=[Path(__file__),Path(__file__).with_name('robust23.py'),Path(__file__).with_name('checks23.py')]
    result={str(p.relative_to(ROOT)):sha(p) for p in ff}
    for p,v in load(PREVIOUS/'protocol.json')['source_hashes'].items():assert sha(ROOT/p)==v;result[p]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks());previous=load(PREVIOUS/'protocol.json')
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,
        scipy_version=scipy.__version__,inputs=previous['inputs'],label_hashes=previous['label_hashes'],frontends=FRONTENDS,models=MODELS,
        prior_forecast_hashes={c:sha(PREVIOUS/c/'forecasts.jsonl') for c in CASES},
        design='Same CV state/measurement inputs/validity as stage22. Keep hold/CV/original_history and add normcap3000, SciPy Huber fit, Huber covariance-shrink+cap. No labels in inference.',
        robustness='Huber delta1.345*existing6px noise; minimum6points and160ms support; covariance from weighted design/noise floor. Positive-part component shrink max(0,1-variance/a^2). Cap3000 from existing image-domain Config, not a physical acceleration truth.',
        score='Execute frozen stage22 scorer with output/model dependency injection only; exact target timestamps and H..H+25ms anchors unchanged. Check prior model forecasts exact.',
        selection='Parameters fixed before this robustness scoring; development data already used to find over-shooting, not blind. No post-result tuning.',
        prior_art=[dict(url='https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html',use='Executes installed SciPy Huber least_squares; no new robust-regression claim.')],
        scope='Position-prediction robustness only. No actual mass/wind/RPM/pose claim; no G or access to yolov10-main.'))
    print('Stage23 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def infer():
    protocol=verify()
    for case in CASES:
        assert sha(INPUT/case/'observations.jsonl')==protocol['inputs'][case]
        assert sha(PREVIOUS/case/'forecasts.jsonl')==protocol['prior_forecast_hashes'][case]
        cache=list(map(json.loads,(PREVIOUS/case/'forecasts.jsonl').read_text().splitlines()))
        rows=list(map(json.loads,(INPUT/case/'observations.jsonl').read_text().splitlines()));folder=OUT/case;folder.mkdir(parents=True,exist_ok=True)
        assert not (folder/'forecasts.jsonl').exists();models={f:Predictor() for f in FRONTENDS};equal=0;began=time.perf_counter()
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as stream:
            for index,r in enumerate(rows):
                result=dict(t=r['t'],frontends={})
                for f,p in models.items():
                    o=r['methods'][f]['observation'];q,baseline=p.step(r['t'],o['center'] if o['detected'] else None)
                    assert baseline==cache[index]['frontends'][f],(case,r['t'],f,'baseline forecasts changed')
                    equal+=1;result['frontends'][f]=q
                stream.write(json.dumps(result,allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(rows),baseline_frontend_records_exact=equal,wall_s=time.perf_counter()-began,
            labels_in_inference=False,sources_unchanged=sources()==protocol['source_hashes']))
        print('Completed '+case,flush=True)


def score():
    verify();score22.OUT=OUT;score22.MODELS=MODELS;score22.verify=verify
    score22.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','all','infer','score']);a=p.parse_args()
    if a.action=='all':infer();score()
    else:globals()[a.action]()
