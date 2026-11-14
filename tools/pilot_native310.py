"""Automatic initialization from native data, fixed31Hz-source image windows."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from native_phase_discovery import NativeContinuousPhase
from pure_visibility170 import window
from replay_joint236 import source

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=('13','15','16'),required=True);a=p.parse_args();n=a.recording
    OUT=ROOT.parent/f'output/native310_20261109/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    files=['native_phase_discovery.py','continuous_rotor_phase.py','continuous_phase_filter.py','angular_phase_candidates.py','tools/pilot_native310.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        initializer_uses_only_native_events_and_current_automatic_conics=True,
        author_periods_or_true_RPM_inputs=False,no_recording_specific_rules=True,no_parameter_search=True,
        validation_thresholds_changed=False,no_trajectory_labels_read=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
    cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
    model=NativeContinuousPhase(ROOT.parent/'output/continuous305_20261109/build/Release/continuous_phase.dll');periods=[];cost=[]
    for i,row in enumerate(rows):
        events=window(cache,row['t']);tick=time.perf_counter();period=model.step(row,events);cost.append(1000*(time.perf_counter()-tick));periods.append(period)
        with (OUT/'periods.jsonl').open('a',encoding='utf8') as fh:fh.write(json.dumps(period,allow_nan=False)+'\n')
        if i%30==0:print(n,i,period['meta']['qualified_current_rings'],flush=True)
    result=dict(frames=len(periods),ring_histogram={str(k):sum(p['meta']['qualified_current_rings']==k for p in periods) for k in range(5)},
        current_force_frames=sum(p['u'] is not None for p in periods),phase_ms_mean=float(np.mean(cost)),phase_ms_p95=float(np.percentile(cost,95)))
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(summary=result,protocol=protocol),indent=2),'utf8');print(result,flush=True)
