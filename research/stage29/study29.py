from pathlib import Path
import argparse,datetime,importlib.util,json,sys
from select29 import ROOT,PARAMS,MODELS,Selector,prior
from checks29 import run as checks

sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage28')]
import study28
OUT=ROOT.parent/'output/stage29_delayed_model_weights_20261018';INPUT=study28.OUT
CASES=study28.CASES;METHOD='canonical_static';load=study28.load;save=study28.save;sha=study28.sha


def sources():
    result={}
    for p in [Path(__file__),Path(__file__).with_name('select29.py'),Path(__file__).with_name('checks29.py')]:result[str(p.relative_to(ROOT))]=sha(p)
    for name,value in load(INPUT/'protocol.json')['source_hashes'].items():assert sha(ROOT/name)==value;result[name]=value
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    p=load(INPUT/'protocol.json');save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        parameters=PARAMS,input_hashes=p['input_hashes'],input_forecasts={c:sha(INPUT/c/'forecasts.jsonl') for c in CASES},label_hashes=p['label_hashes'],models=MODELS,
        selected_axis=METHOD,design='Use frozen stage28 canonical_static forecasts. Same base10 models unchanged, same CV state/validity. Add fixed and delayed adaptive history/all blends and hard expert choice. Detector position arrives AFTER old prediction, used only then; no GT label feedback. All forecasts saved before scoring.',
        source_reuse='Existing paper_bank.py delayed weighting, generalized model list; primary prior art Freund/Schapire1997 https://doi.org/10.1006/jcss.1997.1504; River EWARegressor docs inspected, River not installed/executed.',
        limits='No new online weighting algorithm claim, no regret guarantee under noisy dependent pseudo-reference. Physics experts fall back CV when inactive, so their weight alone is not evidence of physical utility. Development data; no G/yolov10-main.'))
    print('Stage29 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def infer():
    protocol=verify()
    for case in CASES:
        rawpath=study28.study27.input_path(case);assert sha(rawpath)==protocol['input_hashes'][case]
        path=INPUT/case/'forecasts.jsonl';assert sha(path)==protocol['input_forecasts'][case]
        rows=list(map(json.loads,path.read_text(encoding='utf-8').splitlines()));obs=list(map(json.loads,rawpath.read_text(encoding='utf-8').splitlines()));assert len(rows)==len(obs)
        folder=OUT/case;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'forecasts.jsonl').exists();selector=Selector();prefix=[]
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for i,(r,o) in enumerate(zip(rows,obs)):
                q=selector.step(r['frontends'][METHOD],o['observation']);assert q['valid']==r['frontends'][METHOD]['valid']
                for m in prior.MODELS:assert q['coefficients'].get(m)==r['frontends'][METHOD]['coefficients'].get(m)
                if i<80:prefix.append(q)
                f.write(json.dumps(dict(t=r['t'],frontends={'delayed_weights':q}),allow_nan=False)+'\n')
        pref=Selector();assert all(pref.step(r['frontends'][METHOD],o['observation'])==q for r,o,q in zip(rows[:80],obs[:80],prefix))
        save(folder/'completion.json',dict(frames=len(rows),original_coefficients_exact=len(rows),prefix_exact=80,labels_in_inference=False,sources_unchanged=sources()==protocol['source_hashes']))
        print('Completed '+case,flush=True)


def score():
    verify();s=study28.study27.scorer;s.OUT=OUT;s.INPUT=OUT;s.CASES=list(CASES);s.FRONTENDS=['delayed_weights'];s.MODELS=MODELS;s.verify=verify;s.evaluate=prior.evaluate;s.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score','all']);a=p.parse_args()
    if a.action=='all':infer();score()
    else:globals()[a.action]()
