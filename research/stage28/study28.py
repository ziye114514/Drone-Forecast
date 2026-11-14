from pathlib import Path
import argparse,datetime,importlib.util,json
from axis28 import ROOT,PARAMS,METHODS,Predictor,prior
from checks28 import run as checks
import study27

OUT=ROOT.parent/'output/stage28_axis_propagation_20261018';INPUT=study27.OUT
CASES=study27.CASES;MODELS=prior.MODELS;load=study27.load;save=study27.save;sha=study27.sha


def sources():
    result={}
    for p in [Path(__file__),Path(__file__).with_name('axis28.py'),Path(__file__).with_name('checks28.py')]:result[str(p.relative_to(ROOT))]=sha(p)
    for name,value in load(INPUT/'protocol.json')['source_hashes'].items():assert sha(ROOT/name)==value;result[name]=value
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    p=load(INPUT/'protocol.json');save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        parameters=PARAMS,models=MODELS,input_hashes=p['input_hashes'],label_hashes=p['label_hashes'],prior_forecasts={c:sha(INPUT/c/'forecasts.jsonl') for c in CASES},
        design='Only axis implementation changes. Original must exactly reproduce stage27 full forecast records; all variants keep same six historical models/CV state/validity. No thrust/adaptation parameter changes. All forecasts before scoring labels.',
        variants='original; no_rate zeros original estimated angular velocity before/after every step and future rollout; canonical_static uses original fresh-frame measurement consensus and same covariance without antipodal temporal flips/rate propagation; instantaneous same canonical measurement without temporal fusion.',
        prior_art=['https://www.jstatsoft.org/article/view/v089i04','https://users.aalto.fi/~ssarkka/pub/ctvmf-conf.pdf'],
        scope='Prior art directional filtering motivates separating axial/sign uncertainty; no libDirectional or vMF implementation executed. Simple ablation of existing local weak-normal/KF code, not novel directional filter. Same image-up/non-inverted prior as original, no camera/pose truth. Development data, no G/yolov10-main.'))
    print('Stage28 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def infer():
    protocol=verify()
    for case in CASES:
        path=study27.input_path(case);assert sha(path)==protocol['input_hashes'][case]
        assert sha(INPUT/case/'forecasts.jsonl')==protocol['prior_forecasts'][case]
        rows=list(map(json.loads,path.read_text(encoding='utf-8').splitlines()))
        previous=[r['frontends']['per_id_frequency'] for r in map(json.loads,(INPUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines())]
        folder=OUT/case;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'forecasts.jsonl').exists();pp={m:Predictor(m) for m in METHODS}
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r,old in zip(rows,previous):
                result={m:p.step(r)[0] for m,p in pp.items()};assert result['original']==old
                assert len({q['valid'] for q in result.values()})==1
                for q in result.values():
                    for name in prior.robust.MODELS:assert q['coefficients'].get(name)==old['coefficients'].get(name)
                f.write(json.dumps(dict(t=r['t'],frontends=result),allow_nan=False)+'\n')
        assert len(rows)==len(previous);save(folder/'completion.json',dict(frames=len(rows),full_original_records_exact=len(rows),labels_in_inference=False,sources_unchanged=sources()==protocol['source_hashes']))
        print('Completed '+case,flush=True)


def score():
    verify();s=study27.scorer;s.OUT=OUT;s.INPUT=OUT;s.CASES=list(CASES);s.FRONTENDS=METHODS;s.MODELS=MODELS;s.verify=verify;s.evaluate=prior.evaluate;s.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score','all']);a=p.parse_args()
    if a.action=='all':infer();score()
    else:globals()[a.action]()
