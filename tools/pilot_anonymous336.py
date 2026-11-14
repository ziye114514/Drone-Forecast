"""Persistent author geometry/phase, original frozen4 observations."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from anonymous_ring_phase import AnonymousRingPhase

if __name__=='__main__':
    out=ROOT.parent/'output/anonymous336_20261109';out.mkdir(exist_ok=False)
    files=['anonymous_ring_phase.py','tools/pilot_anonymous336.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,no_full_template=True,no_parameter_search=True,no_reference_read=True),indent=2),'utf8')
    src=ROOT.parent/'output/frozen324_20261109';rows=list(map(json.loads,(src/'control.jsonl').read_text('utf8').splitlines()));manifest=json.loads((src/'cache/manifest.json').read_text('utf8'))['frames']
    model=AnonymousRingPhase(ROOT.parent/'output/snapshot335_20261109/build/Release/helix_reference.dll');result=[];cost=[]
    try:
        for i,(row,packet) in enumerate(zip(rows,manifest)):
            e=np.load(src/'cache'/packet['file']);tick=time.perf_counter();r=model.step(row,e);cost.append(time.perf_counter()-tick)
            r.update(original_current_detection=bool(row['body'].get('detected')),original_valid=row['valid']);result.append(r)
            with (out/'periods.jsonl').open('a',encoding='utf8') as fh:fh.write(json.dumps(r,allow_nan=False)+'\n')
            if i%20==0:print(i,row['t'],'native',r['original_current_detection'],'phase',r['meta']['qualified_current_rings'],'support',r['meta']['current_support_events'],flush=True)
    finally:model.close()
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    gained=[dict(t=r['t'],rings=r['meta']['qualified_current_rings']) for r in result if not r['original_current_detection'] and r['meta']['qualified_current_rings']]
    summary=dict(frames=len(result),current_phase_frames=sum(r['meta']['qualified_current_rings']>0 for r in result),
        phase_current_during_detection_gap=gained,phase_CPU_ms_mean=float(np.mean(cost)*1000),prediction_gain_proven=False)
    (out/'results.json').write_text(json.dumps(summary,indent=2),'utf8');print(summary,flush=True)
