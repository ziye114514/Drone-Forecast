"""Freeze and smoke-test the RAW-only persistent local detector bridge."""
from pathlib import Path
import argparse,datetime,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('body_runtime42',ROOT/'run_body_forecast.py');pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
OUT=ROOT.parent/'output/stage42_raw_body_bridge_20261018'
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def freeze():
    assert not (OUT/'protocol.json').exists();sources=pipeline.fingerprints();sources['research/stage42/study42.py']=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources,cases={'train3':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=2.),'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=1.)},
        design='New RAW-only entry, persistent local GPU batch1 over PNG stdin/stdout, SDK311 parent. Same original geometry + body center and history models as41; weights40 fixed. Compare native frames/geometric exactness, per-frame GPU vs batched41 numeric difference, full output determinism with independent short RAW prefix. No GT read.',
        scope='Integration smoke, not additional accuracy evidence; batch1 timing includes PNG IPC and geometry, wall includes model init/decode/video. Source timestamps50fps do not imply real time. No pose/frequency/force features enter these history predictors.'))
def run():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for c,cfg in p['cases'].items():pipeline.run(cfg['raw'],OUT/c,cfg['start'],cfg['duration'],preview=True)
    pipeline.run(p['cases']['train3']['raw'],OUT/'prefix_train3',25.,.8,preview=False)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);globals()[a.parse_args().action]()
