from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from bootstrapped_phase import BootstrappedContinuousPhase
from pure_visibility170 import window
from replay_joint236 import source

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=('13','15','16'),required=True);a=p.parse_args();n=a.recording
    OUT=ROOT.parent/f'output/bootstrap311_20261109/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    files=['bootstrapped_phase.py','native_phase_discovery.py','joint_ring_phase.py','continuous_rotor_phase.py','continuous_phase_filter.py','tools/pilot_bootstrap311.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        cold_start_from_native_events_only=True,saved_author_periods_required=False,
        current_measurements_require_later_independent_late_halves=True,
        no_reference_reads=True,no_parameter_search=True,no_recording_specific_rules=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    model=BootstrappedContinuousPhase(ROOT.parent/'output/continuous305_20261109/build/Release/continuous_phase.dll',ROOT.parent/'output/helix_cpu232_20261108/build/Release/helix_reference.dll')
    rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
    periods=[];cost=[]
    for i,row in enumerate(rows):
        events=window(cache,row['t']);tick=time.perf_counter();period=model.step(row,events);cost.append(1000*(time.perf_counter()-tick));periods.append(period)
        with (OUT/'periods.jsonl').open('a',encoding='utf8') as fh:fh.write(json.dumps(period,allow_nan=False)+'\n')
        if i%30==0:print(n,i,period['meta']['qualified_current_rings'],flush=True)
    result=dict(frames=len(periods),ring_histogram={str(k):sum(r['meta']['qualified_current_rings']==k for r in periods) for k in range(5)},
        current_force_frames=sum(r['u'] is not None for r in periods),phase_ms_mean=float(np.mean(cost)),phase_ms_p95=float(np.percentile(cost,95)),
        prior_refinements=sum(r['meta'].get('author_bootstrap_refinements',0) for r in periods))
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(summary=result,protocol=protocol),indent=2),'utf8');print(result,flush=True)
