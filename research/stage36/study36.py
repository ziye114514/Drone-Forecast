from pathlib import Path
import argparse,datetime,importlib.util,json,sys
from body36 import ROOT,PARAMS,METHODS,factory
from checks36 import run as checks

sys.path.insert(0,str(ROOT/'research/stage35'))
import study35 as prior
OLD35=ROOT.parent/'output/stage35_body_feature_flow_20261018';OUT=ROOT.parent/'output/stage36_full_side_body_support_20261018'
CASES={k:dict(v) for k,v in prior.CASES.items()};CASES['side6']['duration']=12.
load=prior.load;save=prior.save;sha=prior.sha


def sources():
    result={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('body36.py'),Path(__file__).with_name('checks36.py')]}
    for n,v in load(OLD35/'protocol.json')['source_hashes'].items():assert sha(ROOT/n)==v;result[n]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks());p=load(OLD35/'protocol.json')
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,cases=CASES,
        input_hashes=p['input_hashes'],label_hashes=p['label_hashes'],strict_cache_hashes={c:{name:sha(OLD35/c/name) for name in ['observations.jsonl','forecasts.jsonl']} for c in CASES},
        design='Execute frozen stage35 runner/scorer with separate per-variant module namespaces, no source edits. Same event images, current geometry, 2s age and quality/FB/scale gates; compare original6-point similarity/translation,3-point counterparts,3-point similarity with current ROI feature replenishment ONLY after accepted motion. Expand side108–120 to include extreme segment.',
        regression='Baseline full ordinary streams must match previous six historical coefficients; strict35 variants must exactly reproduce all earlier35 observations/forecasts on common prefixes. Dropout remains control and not FDE-scored.',
        limits='Post-development engineering ablation, no blind claim. Three image features cannot establish true3D plane pose or semantic identity. Replenishment may drift, independent-geometry age never reset. No G/yolov10-main.'))
    print('Stage36 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def configure():
    prior.OUT=OUT;prior.CASES=CASES;prior.METHODS=METHODS;prior.PARAMS=PARAMS;prior.BodyTracker=factory;prior.verify=verify


def run():
    protocol=verify();configure();prior.run();checks=[]
    for case in CASES:
        for name in ['observations.jsonl','forecasts.jsonl']:assert sha(OLD35/case/name)==protocol['strict_cache_hashes'][case][name]
        a=list(map(json.loads,(OLD35/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()));b=list(map(json.loads,(OUT/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()))
        fa=list(map(json.loads,(OLD35/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines()));fb=list(map(json.loads,(OUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines()));assert len(b)>=len(a)
        for i in range(len(a)):
            assert a[i]['t']==b[i]['t']==fa[i]['t']==fb[i]['t']
            for m in ['baseline','similarity_long','translation_long']:
                assert a[i]['methods'][m]==b[i]['methods'][m],(case,i,m,'strict observation differs')
                assert fa[i]['frontends'][m]==fb[i]['frontends'][m],(case,i,m,'strict forecast differs')
        checks.append(dict(case=case,prefix_frames=len(a),all3_strict_observations_and_forecasts_exact=True))
    save(OUT/'strict_prefix_regression.json',checks);print('All strict35 prefixes exact',flush=True)


def score():configure();prior.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','score','all']);a=p.parse_args()
    if a.action=='all':run();score()
    else:globals()[a.action]()
