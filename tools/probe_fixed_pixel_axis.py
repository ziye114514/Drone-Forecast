"""One fixed spatial subset, verifying exact per-selected-pixel SDK outputs."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage141')]
from live_decode import native_batches
from live_pixel_axis import FixedPixelAxis
from native_axis141 import NativeAxis
OUT=WORK/'output/fixed_pixel_periodicity_20261026';OUT.mkdir(exist_ok=False)
full=NativeAxis(1280,720);subset=FixedPixelAxis(1280,720)
run=WORK/'output/live_scale_rollout_RAW_20261026/side_v2/tracking.jsonl'
body={round(r['t'],6):r for r in [json.loads(s) for s in run.read_text('utf-8').splitlines()]}
times={'full':[],'subset':[]};qualified={'full':0,'subset':0};axes={'full':0,'subset':0}
rows=[];records_compared=0;raw_count=0;matched_body=0
for t,e in native_batches('E:/dataset/train/6/media/event.raw',109.9,1.3,build_index=False):
    raw_count+=len(e);start=time.perf_counter();q1=full.feed(e,t);times['full'].append((time.perf_counter()-start)*1000)
    start=time.perf_counter();q2=subset.feed(e,t);times['subset'].append((time.perf_counter()-start)*1000)
    selected=q1[subset.mask[q1['y'],q1['x']]]
    assert len(selected)==len(q2),(t,len(selected),len(q2))
    for name in ['x','y','t','frequency']:assert np.array_equal(selected[name],q2[name]),(t,name)
    records_compared+=len(q2)
    r=body.get(round(t,6));info={}
    if r is not None:
        matched_body+=1
        for name,engine,q in [('full',full,q1),('subset',subset,q2)]:
            d,_=engine.measure(q,r['body'],t,(r['coefficient'] or {}).get('velocity',[0.,0.]))
            qualified[name]+=bool(d['qualified']);axes[name]+=bool(d['axis_valid']);info[name]=d
    rows.append(dict(t=t,native_events=len(e),full_frequency_records=len(q1),subset_frequency_records=len(q2),diagnostic=info))
report=dict(scope='One actual RAW1.3s incl0.1s warmup, no NN rerun. Saved current NN boxes only for matched-tick observer qualification.',
    fixed_seed=141000,selected_sensor_pixels=int(subset.mask.sum()),mask_sha256=hashlib.sha256(subset.mask.tobytes()).hexdigest(),
    native_events=raw_count,selected_events=subset.selected_events,native_slices=len(rows),
    exact_selected_frequency_records=records_compared,all_native_fields_exact_at_selected_pixels=True,
    matched_body_ticks=matched_body,qualified=qualified,unsigned_axis_qualified=axes,
    feed_ms={k:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in times.items()},
    timing_is_single_sequential_in_process_probe=True,whole_frame_observability_identical=False,
    detector_input_or_default_forecast_changed=False,real_RPM_or_force_truth=False,default_promoted=False,rows=rows)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
