"""Frozen cached-feature factorial study; forecasts saved before sparse-box scoring."""
from pathlib import Path
from dataclasses import asdict
import argparse,datetime,hashlib,importlib.util,json,sys
from physics27 import ROOT,PARAMS,PHYSICS,MODELS,Predictor,PaperConfig,evaluate
from checks27 import run as checks

sys.path.append(str(ROOT/'research/stage22'))
spec=importlib.util.spec_from_file_location('endpoint22_physics27',ROOT/'research/stage22/study22.py')
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
load=scorer.load;save=scorer.save;sha=scorer.sha
OUT=ROOT.parent/'output/stage27_same_state_physics_20261018'
CASES={'clear6':dict(start=22.,duration=5.,folder='stage12_clear_22_27_20261017'),
    'side6':dict(start=108.,duration=12.,folder='stage12_side_108_120_20261017')}


def input_path(case):return ROOT.parent/'output'/CASES[case]['folder']/'observations.jsonl'


def sources():
    result={}
    for p in [Path(__file__),Path(__file__).with_name('physics27.py'),Path(__file__).with_name('checks27.py')]:result[str(p.relative_to(ROOT))]=sha(p)
    for name,value in load(ROOT.parent/'output/stage26_natural_layout_20261018/protocol.json')['source_hashes'].items():
        assert sha(ROOT/name)==value;result[name]=value
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        input_hashes={c:sha(input_path(c)) for c in CASES},label_hashes=load(ROOT.parent/'output/stage26_natural_layout_20261018/protocol.json')['label_hashes'],
        parameters=PARAMS,paper_config=asdict(PaperConfig()),models=MODELS,
        cached_input_provenance={c:dict(protocol_sha256=sha(input_path(c).parent/'protocol.json'),
            archived_source_hashes=load(input_path(c).parent/'protocol.json')['source_hashes'],
            current_source_mismatches=[name for name,value in load(input_path(c).parent/'protocol.json')['source_hashes'].items() if sha(ROOT/name)!=value]) for c in CASES},
        design='Reuse unmodified stage12 native10 current-position/per-ID frequency records; shared stage23 CV state/validity for ALL predictors. Baselines identical; projected models vary adaptation, changing normal, relative frequency squared. Same freshness/quality gate to isolate inputs; axis differential set zero. Missing features fallback exactly CV. All forecasts saved before label reads.',
        adaptation='Reuse ProjectedThrustModel without edits: new current filtered position corrects delayed previous prediction, no GT labels. Effective image gains/drag/residual only, not mass/thrust coefficient/wind identification.',
        evaluation='Reuse stage22 endpoint scorer; saved rollout interpolated at exact future box timestamp. Nominal horizons .1/.2/.4; anchors up to25ms older, this input about10ms. All models same validity; report actual feature-active targets separately. Sparse developmental boxes, no independent RPM/pose truth.',
        prior_art='https://arxiv.org/abs/2604.18289; existing local paper_physics.py, stage23 Huber and stage22 endpoint scorer. Independent image-domain adaptation, not author code or claimed published-table reproduction.',
        scope='Already explored development recordings; do not select gains after results. Stage12 cached inputs are hashed and reused, not regenerated: archived clear-run frequency.py differs from current wrapper; this provenance discrepancy is retained, while current executed runtime dependencies are verified against stage26. No G or access to yolov10-main.'))
    print('Stage27 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def infer():
    protocol=verify()
    for case in CASES:
        assert sha(input_path(case))==protocol['input_hashes'][case]
        rows=list(map(json.loads,input_path(case).read_text(encoding='utf-8').splitlines()));p=Predictor();folder=OUT/case
        folder.mkdir(parents=True,exist_ok=True);assert not (folder/'forecasts.jsonl').exists();prefix=[]
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for i,r in enumerate(rows):
                q,b=p.step(r)
                assert q['valid']==b['valid']
                for model in b['coefficients']:assert q['coefficients'][model]==b['coefficients'][model]
                if i<80:prefix.append(q)
                f.write(json.dumps(dict(t=r['t'],frontends={'per_id_frequency':q}),allow_nan=False)+'\n')
        pref=Predictor();assert all(pref.step(r)[0]==q for r,q in zip(rows[:80],prefix))
        save(folder/'completion.json',dict(frames=len(rows),baseline_identical_same_stream=len(rows),prefix_exact=len(prefix),labels_in_inference=False,sources_unchanged=sources()==protocol['source_hashes']))
        print('Completed '+case,flush=True)


def score():
    verify();scorer.OUT=OUT;scorer.INPUT=OUT;scorer.CASES=list(CASES);scorer.FRONTENDS=['per_id_frequency'];scorer.MODELS=MODELS;scorer.verify=verify;scorer.evaluate=evaluate
    scorer.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score','all']);args=p.parse_args()
    if args.action=='all':infer();score()
    else:globals()[args.action]()
