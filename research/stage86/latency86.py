"""Recorded trace deadline audit; no new trajectory or accuracy experiment."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT.parent/'output';OUT=BASE/'stage86_latency_deadlines_20261021'
STAGES={78:BASE/'stage78_RAW_separated_forecast_20261021',81:BASE/'stage81_RAW_exact_compute_20261021',
        83:BASE/'stage83_RAW_streamed_window_v2_20261021',85:BASE/'stage85_RAW_batched_rollout_20261021'}
CASES=dict(clear6=100,side6=250,train3=150,extra2=100)
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fifo(source,cost):
    finish=[];previous=-float('inf')
    for s,p in zip(source,cost):
        previous=max(float(s),previous)+float(p);finish.append(previous)
    return np.array(finish)-source
def run():
    assert not OUT.exists() and (STAGES[85]/'checks85.json').is_file()
    np.testing.assert_allclose(fifo(np.arange(5)*.02,np.ones(5)*.005),.005,rtol=0,atol=1e-15)
    np.testing.assert_allclose(fifo(np.arange(5)*.02,np.ones(5)*.04),[.04,.06,.08,.10,.12],rtol=0,atol=1e-15)
    inputs={f'{stage}:{case}':sha(folder/case/'observations_forecasts.jsonl') for stage,folder in STAGES.items() for case in CASES}
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),
        input_hashes=inputs,fixed_prefix_counts=CASES,
        design='Audit already recorded processing_ms. Not a new inference/model '
            'or live-camera measurement. Idealized FIFO replay assumes50Hz '
            'arrival, no dropping, one processing server, no cold start. '
            'Costs exclude decode/JSON/preview so optimistic scheduling estimate.',
        scope='Sparse offline source-time prediction accuracy remains a separate '
            'quantity. No latest-only accuracy is claimed; skipping changes '
            'model state and must be tested by a new causal RAW experiment.',
        controls=dict(faster_than_arrival_constant_age=True,slower_than_arrival_backlog_grows=True)))
    table=[]
    for stage,folder in STAGES.items():
        for case,n in CASES.items():
            rr=[]
            with (folder/case/'observations_forecasts.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    rr.append(json.loads(line))
                    if len(rr)==n:break
            assert len(rr)==n
            t=np.array([r['t'] for r in rr]);relative=t-t[0];cost=np.array([r['processing_ms']/1000 for r in rr]);age=fifo(relative,cost)
            valid=np.array([r['frontends']['delayed_feedback']['valid'] for r in rr],bool);lead={}
            for h in [.1,.2,.4]:
                lead[str(h)]=dict(valid_forecasts=int(valid.sum()),
                    isolated_processing_before_target=int(np.sum(valid&(cost<h))),
                    optimistic_FIFO_before_target=int(np.sum(valid&(age<h))))
            table.append(dict(stage=stage,case=case,parent=n,source_duration_s=float(n*.02),
                median_processing_ms=float(np.median(cost)*1000),sum_processing_s=float(cost.sum()),
                optimistic_no_decode_processing_fps=float(n/cost.sum()),source_time_realtime_factor=float(cost.sum()/(n*.02)),
                optimistic_FIFO_last_age_s=float(age[-1]),lead_deadlines=lead,
                source_relative_s=relative.tolist(),processing_s=cost.tolist(),optimistic_FIFO_age_s=age.tolist()))
    save(OUT/'audit86.json',dict(table=table,predictions_changed=False,labels_read=False,
        live_camera_measurement=False,physical_advantage_proven=False,default_promoted=False,
        conclusion='Lower computation helps but source50fps video is not realtime. '
            'Dense online arrival needs bounded queues/asynchronous detector and '
            'timestamp-aware updates, not just more forecast models.'))
    print('86 recorded deadline audit complete',flush=True)
if __name__=='__main__':run()
