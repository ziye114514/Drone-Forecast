"""Frozen research-pipeline smoke plus additional recordings; no label use in runs."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('raw_research_pipeline',ROOT/'core/run_research_forecast.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
OUT=ROOT.parent/'output/stage24_raw_pipeline_20261018'
CASES={'self_clear':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=2.,rotor_features=True,role='Regression against staged observations and forecasts.'),
    'train3_extra':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=6.,rotor_features=False,role='Additional recording for these variants; sampled by earlier audits, not globally blind.'),
    'day10':dict(raw='E:/dataset/experiment/10/media/event.raw',start=20.,duration=4.,rotor_features=False,role='Previously unviewed in stages18–23, no archived boxes; qualitative only.'),
    'night6':dict(raw='E:/dataset/experiment/6/media/event.raw',start=20.,duration=4.,rotor_features=False,role='Previously unviewed in stages18–23, no archived boxes; qualitative only.'),
    'dusk17':dict(raw='E:/dataset/experiment/17/media/event.raw',start=20.,duration=4.,rotor_features=False,role='Previously unviewed in stages18–23, no archived boxes; qualitative only.')}


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sources():
    result=pipeline.fingerprints();result[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    previous=load(ROOT.parent/'output/stage23_robust_acceleration_20261018/protocol.json')['source_hashes']
    for p,v in previous.items():assert sha(ROOT/p)==v;result[p]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();metadata={}
    for c in CASES.values():
        p=Path(c['raw']);st=p.stat()
        with p.open('rb') as f:head=hashlib.sha256(f.read(65536)).hexdigest()
        metadata[str(p)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,first64k_sha256=head)
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        raw_metadata=metadata,frontend='native20',primary='trusted_acc',all_predictors=pipeline.robust.MODELS,preview=True,
        design='Select settings from development stage23, freeze before these recordings. Inference entry reads RAW only; case study later scores archived train3 boxes. No experiment boxes exist, so do not claim accuracy there.',
        scope='Image-position research forecasts, optional original RotorSensing diagnostics only. No future frames/labels feed inference. No mass/wind/physical thrust/RPM truth. No G or access to yolov10-main.'))
    print('Stage24 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run():
    verify()
    for case,c in CASES.items():
        pipeline.run(c['raw'],OUT/case,c['start'],c['duration'],'native20','trusted_acc',c['rotor_features'],True)
    print('All pipeline cases complete',flush=True)


def analyze():
    verify();summaries=[]
    prior_obs={round(r['t'],8):r['methods']['native20']['observation'] for r in map(json.loads,(ROOT.parent/'output/stage21_native_windows_20261018/clear6/observations.jsonl').read_text().splitlines())}
    prior_forecast={round(r['t'],8):r['frontends']['native20'] for r in map(json.loads,(ROOT.parent/'output/stage23_robust_acceleration_20261018/clear6/forecasts.jsonl').read_text().splitlines())}
    for case,c in CASES.items():
        rr=list(map(json.loads,(OUT/case/'observations_forecasts.jsonl').read_text().splitlines()));completion=load(OUT/case/'completion.json')
        assert len(rr)==round(c['duration']/.02)==completion['frames'];assert completion['source_unchanged']
        if case=='self_clear':
            assert all(r['observation']==prior_obs[round(r['t'],8)] and r['forecasting']==prior_forecast[round(r['t'],8)] for r in rr)
            save(OUT/case/'regression.json',dict(frames=len(rr),observations_exact=True,all_forecasts_exact=True,
                optional_frequency_diagnostics_do_not_change_predictions=True))
        summary=dict(case=case,**completion,role=c['role'])
        if case=='train3_extra':
            labels=load(ROOT.parent/'output/raw_only_project/evaluation_data/train3.json')['labels'];tt=np.array([r['t'] for r in rr]);scored=[]
            for l in labels:
                if not c['start']<l['t']<=c['start']+c['duration']:continue
                i=int(np.searchsorted(tt,l['t'],side='right')-1)
                if i<0 or l['t']-tt[i]>.025:continue
                o=rr[i]['observation'];d=dict(t=l['t'],state=l['state'],detected=o['detected'])
                if l['state']=='present' and o['detected']:
                    b=np.array(l['bbox']);d['center_error_px']=float(np.linalg.norm(np.array(o['center'])-(b[:2]+b[2:])/2))
                scored.append(d)
            save(OUT/case/'sparse_positions.json',scored)
            summary.update(present_labels=sum(r['state']=='present' for r in scored),present_detected=sum(r['state']=='present' and r['detected'] for r in scored),
                absent_labels=sum(r['state']=='absent' for r in scored),absent_detections=sum(r['state']=='absent' and r['detected'] for r in scored))
        summaries.append(summary)
    save(OUT/'analysis.json',dict(cases=summaries,scope='Only train3 archived coarse boxes permit quantitative checks. Unlabelled new experiment recordings qualitative only.'))
    print('Stage24 pipeline checks complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','analyze','all']);a=p.parse_args()
    if a.action=='all':run();analyze()
    else:globals()[a.action]()
