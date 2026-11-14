"""Check semantic equality of full/compacted current periodic packets."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage141')]
from live_decode import native_batches
from live_pixel_axis import SDKFixedPixelAxis
from live_periodic_packet import compact_current
from native_axis141 import measure_current
OUT=WORK/'output/periodic_packet_20261026';OUT.mkdir(exist_ok=False)
(OUT/'live_periodic_packet.py').write_bytes((ROOT/'core/live_periodic_packet.py').read_bytes())
saved=[json.loads(l) for l in (WORK/'output/live_scale_rollout_RAW_20261026/side_v2/tracking.jsonl').read_text('utf-8').splitlines()]
body={round(r['t'],6):r for r in saved};axis=SDKFixedPixelAxis(1280,720)
rows=[];input_bytes=output_bytes=observations=0;latency=[]
for t,e in native_batches('E:/dataset/train/6/media/event.raw',109.9,1.3,build_index=False):
    q=axis.feed(e,t);tick=time.perf_counter();small=compact_current(q,t,1280);latency.append((time.perf_counter()-tick)*1000)
    input_bytes+=q.nbytes;output_bytes+=small.nbytes
    assert np.array_equal(small,compact_current(small,t,1280))
    row=body.get(round(t,6))
    if row is not None:
        velocity=(row['coefficient'] or {}).get('velocity',[0.,0.])
        a,ap=measure_current(q,row['body'],t,velocity,1280,720)
        b,bp=measure_current(small,row['body'],t,velocity,1280,720)
        assert a==b and np.array_equal(ap,bp),(t,a,b)
        observations+=1
    rows.append(dict(t=t,frequency_records=len(q),transport_records=len(small)))
# Invalid newer frequency must not suppress an earlier valid pixel record.
test=np.zeros(5,dtype=q.dtype);test['x']=[5,5,5,6,7];test['y']=1
test['t']=[999970,999980,999990,999995,900000];test['frequency']=[100,200,2000,300,100]
small=compact_current(test,1.,1280)
assert small['frequency'].tolist()==[200,2000,300]
assert compact_current(None,1.,1280) is None
report=dict(scope='Exact current141 summary/line/points equality at saved NN cutoffs; not stateful frequency-input compression.',
    native_slices=len(rows),compared_current_observations=observations,all_observations_exact=True,
    idempotent=True,invalid_frequency_exclusion_semantics_preserved=True,
    input_bytes=input_bytes,transport_bytes=output_bytes,reduction_fraction=1-output_bytes/input_bytes,
    compaction_ms=dict(median=float(np.median(latency)),p95=float(np.percentile(latency,95))),
    rows=rows,forecast_changed=False,default_promoted=False)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
