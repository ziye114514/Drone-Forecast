"""Frozen cross-recording check of the complete RAW physics/selection runner."""
from pathlib import Path
import argparse,datetime,importlib.util,json,sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage30'))
import study30
pipeline=study30.pipeline
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage22'))
spec=importlib.util.spec_from_file_location('sparse_score31',ROOT/'research/stage22/study22.py');scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
OUT=ROOT.parent/'output/stage31_cross_recording_physics_20261018'
CASES={'train3':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=6.,role='Cross recording for physical/selector variants, earlier stage24 sampled this interval; not globally blind.')}
load=study30.load;save=study30.save;sha=study30.sha


def sources():
    result=study30.sources();result[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__));return result


def freeze():
    assert not (OUT/'protocol.json').exists();metadata={}
    for cfg in CASES.values():
        p=Path(cfg['raw']);st=p.stat();metadata[str(p)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,raw_metadata=metadata,
        label_hashes={'train3.json':sha(ROOT.parent/'output/raw_only_project/evaluation_data/train3.json')},models=pipeline.select29.MODELS,
        design='Run unchanged RAW-only stage30 entry on additional train3 aligned RAW,25–31s. Same settings, no result-driven tuning. Save all predictions before label scoring; directly reuse stage22 exact-target scorer. Compare CV/history/physics/adaptive experts with identical coverage.',
        scope='Cross-recording check, earlier20ms visual/detection exploration known, not blind or independent new dataset. Sparse approximate boxes only, no RPM/pose truth; no G/yolov10-main.'))
    print('Stage31 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run():
    verify()
    for c,cfg in CASES.items():pipeline.run(cfg['raw'],OUT/c,cfg['start'],cfg['duration'],preview=True)


def score():
    verify()
    for c in CASES:
        path=OUT/c/'forecasts.jsonl';assert not path.exists()
        with path.open('x',encoding='utf-8') as f:
            for r in map(json.loads,(OUT/c/'observations_forecasts.jsonl').read_text(encoding='utf-8').splitlines()):
                f.write(json.dumps(dict(t=r['t'],frontends={'raw_adaptive_physics':r['forecasting']}),allow_nan=False)+'\n')
    scorer.OUT=OUT;scorer.INPUT=OUT;scorer.CASES=list(CASES);scorer.FRONTENDS=['raw_adaptive_physics'];scorer.MODELS=pipeline.select29.MODELS;scorer.verify=verify
    scorer.label_path=lambda c:ROOT.parent/'output/raw_only_project/evaluation_data/train3.json';scorer.evaluate=pipeline.select29.prior.evaluate;scorer.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','score','all']);a=p.parse_args()
    if a.action=='all':run();score()
    else:globals()[a.action]()
