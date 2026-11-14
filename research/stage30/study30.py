from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('raw_physical30',ROOT/'core/run_adaptive_physics.py');pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
OUT=ROOT.parent/'output/stage30_raw_adaptive_physics_20261018'
CASES={'clear6':dict(start=22.,duration=2.,input_folder='stage12_clear_22_27_20261017'),
    'side6':dict(start=108.,duration=5.,input_folder='stage12_side_108_120_20261017')}


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def sources():
    result=pipeline.fingerprints();result[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    for name,value in load(ROOT.parent/'output/stage29_delayed_model_weights_20261018/protocol.json')['source_hashes'].items():assert sha(ROOT/name)==value;result[name]=value
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();raw=Path('E:/dataset/train/6/media/event.raw');st=raw.stat()
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,raw=str(raw),raw_bytes=st.st_size,raw_mtime_ns=st.st_mtime_ns,
        input_hashes={c:sha(ROOT.parent/'output'/cfg['input_folder']/'observations.jsonl') for c,cfg in CASES.items()},
        forecast_hashes={c:sha(ROOT.parent/'output/stage29_delayed_model_weights_20261018'/c/'forecasts.jsonl') for c in CASES},
        design='New RAW-only root entry actually connects original geometry/per-ID sensing, canonical-static axis/projected-thrust ablations and delayed weighting. Cold-start same10ms recipe; compare fresh RAW observations/sensing/full15 forecasts with cached stage12/29 after inference only. No parameter tuning.',
        scope='Integration/regression/preview, not additional blind accuracy evidence. No G/yolov10-main.'))
    print('Stage30 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run():
    p=verify()
    for case,cfg in CASES.items():pipeline.run(p['raw'],OUT/case,cfg['start'],cfg['duration'],preview=True)


def analyze():
    p=verify();summary=[]
    for case,cfg in CASES.items():
        rowpath=ROOT.parent/'output'/cfg['input_folder']/'observations.jsonl';fp=ROOT.parent/'output/stage29_delayed_model_weights_20261018'/case/'forecasts.jsonl'
        assert sha(rowpath)==p['input_hashes'][case] and sha(fp)==p['forecast_hashes'][case]
        old={round(r['t'],8):r for r in map(json.loads,rowpath.read_text(encoding='utf-8').splitlines())}
        forecasts={round(r['t'],8):r['frontends']['delayed_weights'] for r in map(json.loads,fp.read_text(encoding='utf-8').splitlines())}
        rows=list(map(json.loads,(OUT/case/'observations_forecasts.jsonl').read_text(encoding='utf-8').splitlines()));completion=load(OUT/case/'completion.json')
        assert len(rows)==completion['frames'] and rows
        for r in rows:
            t=round(r['t'],8);assert dict(t=r['t'],event_count=r['event_count'],observation=r['observation'],sensing=r['sensing'])==old[t],(case,t,'RAW observation differs')
            assert r['forecasting']==forecasts[t],(case,t,'forecast differs')
        result=dict(case=case,**completion,observations_sensing_exact=len(rows),all15_forecast_records_exact=len(rows))
        save(OUT/case/'regression.json',result);summary.append(result)
    save(OUT/'analysis.json',summary);print('Fresh RAW observations/sensing/all forecasts exactly match cached studies',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','analyze','all']);a=p.parse_args()
    if a.action=='all':run();analyze()
    else:globals()[a.action]()
