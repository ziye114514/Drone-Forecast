"""One native-event probe with frozen current NN observations, no labels fed."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage141')]
from live_decode import native_batches
from live_pixel_axis import SDKFixedPixelAxis
from live_periodic_recovery import PeriodicRecovery,PARAMS
from native_axis141 import measure_current

OUT=WORK/'output/periodic_center_recovery_20261026';OUT.mkdir(exist_ok=False)
(OUT/'source_snapshot').mkdir()
for name in ['live_periodic_recovery.py','live_pixel_axis.py']:
    (OUT/'source_snapshot'/name).write_bytes((ROOT/name).read_bytes())
(OUT/'design.json').write_text(json.dumps(PARAMS,indent=2),encoding='utf-8')
results={}
for case,start,path in [('side',110.,'E:/dataset/train/6/media/event.raw'),('train2',40.,'E:/dataset/train/2/aligned/event.raw')]:
    folder='side_v2' if case=='side' else case
    tracking=WORK/'output/live_scale_rollout_RAW_20261026'/folder/'tracking.jsonl'
    saved={round(r['t'],6):r for r in [json.loads(s) for s in tracking.read_text('utf-8').splitlines()]}
    native=SDKFixedPixelAxis(1280,720);recovery=PeriodicRecovery(1280,720,measure_current)
    rows=[];velocity=[0.,0.];events=0;slices=0
    for t,e in native_batches(path,start-.1,4.1,build_index=False):
        events+=len(e);slices+=1;q=native.feed(e,t);r=saved.get(round(t,6))
        if r is None:continue
        body=r['body'];candidate,periodic,info=recovery.observe(q,body,t,velocity)
        if r['coefficient'] is not None:velocity=r['coefficient']['velocity']
        rows.append(dict(t=t,NN_detected=bool(body['detected']),recovered_body=candidate,
                         periodic=periodic,recovery=info))
    # No fresh support cannot extend the track even with a recent proposal.
    test_t=t+.04
    no_evidence=recovery.observe(q[:0],dict(detected=False),test_t,velocity)[0]
    assert no_evidence is None
    result=dict(native_slices=slices,native_events=events,matched_NN_ticks=len(rows),
        missing_NN=sum(not r['NN_detected'] for r in rows),
        recovered_current_centers=sum(r['recovered_body'] is not None for r in rows),
        reasons=dict(Counter(r['recovery']['reason'] for r in rows)),
        empty_current_packet_cannot_recover=True,rows=rows)
    (OUT/(case+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    results[case]={k:v for k,v in result.items() if k!='rows'}
report=dict(scope='Native RAW event observer with saved current NN inputs; not a fresh detector/pipeline benchmark.',
    parameters=PARAMS,results=results,manual_labels_used=False,default_promoted=False,
    forecast_or_thrust_changed=False)
(OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
