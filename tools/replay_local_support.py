"""Offline image-algorithm iteration from exact packed native support frames.

This never replaces production RAW decoding. Polarity and event timing inside
a batch are absent, so the cache must not be used for RPM/periodicity analysis.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_local_flow import FeatureFlow
parser=argparse.ArgumentParser();parser.add_argument('--cache-json',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--spatial-diagnostics',action='store_true');args=parser.parse_args()
meta=args.cache_json.resolve();out=args.output.resolve();assert not out.exists()
data=json.loads(meta.read_text('utf-8'));cache=np.load(meta.with_suffix('.npz'),allow_pickle=False)
assert cache['shape'].tolist()==[720,1280] and len(data['records'])==len(cache['packbits'])
cv2.setNumThreads(2);cv2.setRNGSeed(261026);observer=FeatureFlow(crop=True,padding=112,spatial_diagnostics=args.spatial_diagnostics)
rows=[];start=time.perf_counter()
for record,packed in zip(data['records'],cache['packbits']):
    support=np.unpackbits(packed).reshape(720,1280).astype(np.uint8)*255
    assert hashlib.sha256(support.tobytes()).hexdigest()==record['support_sha256']
    rows.append(dict(t=record['t'],observation=observer.step(record['t'],support,record['body'])))
report=dict(rows=rows,elapsed_s=time.perf_counter()-start,all_support_hashes_exact=True,RAW_or_GPU_started=False,
    labels_read=False,scope='Offline current-frame spatial-algorithm cache; not production performance.')
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
