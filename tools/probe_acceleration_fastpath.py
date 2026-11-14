"""Compare the exact initial-WLS shortcut against the first actual solver."""
from pathlib import Path
import importlib.util,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import live_flow_acceleration as current
from live_scale_rollout import ScaleRolloutMotion
from live_local_flow import velocity_coefficient
from summarize_live_pilot import read_rows
OUT=WORK/'output/flow_acceleration_20261026/fit_fastpath';OUT.mkdir(exist_ok=False)
frozen=OUT.parent/'actual_side/source_snapshot/live_flow_acceleration.py'
spec=importlib.util.spec_from_file_location('frozen_flow_acceleration',frozen);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sequences={}
for case in ['side','train2']:
    rows=read_rows(OUT.parent/('actual_'+case)/'tracking.jsonl')
    sequences[case]=[(r['t'],r['velocity_only_coefficient'],r['body'],r['diagnostic']['local_flow']['observation']) for r in rows]
run=WORK/'output/heldout_motion_RAW_20261026';model=ScaleRolloutMotion(25.)
obs={round(r['t'],6):r['observation'] for r in json.loads((WORK/'output/spatial_flow_20261026/test3.json').read_text('utf-8'))};sequences['test3']=[]
for row in read_rows(run/'tracking.jsonl'):
    t=row['t'];c,_,_=model.step(t,row['body']);flow=obs[round(t,6)]
    c,_=velocity_coefficient(c,model.kf.P[2:4,2:4] if c else None,flow,t)
    sequences['test3'].append((t,c,row['body'],flow))
calls=dict(old=0,new=0);old_solver=old.least_squares;new_solver=current.least_squares
def count_old(*a,**kw):calls['old']+=1;return old_solver(*a,**kw)
def count_new(*a,**kw):calls['new']+=1;return new_solver(*a,**kw)
old.least_squares=count_old;current.least_squares=count_new
checks={}
for case,sequence in sequences.items():
    a,b=old.FlowAcceleration(),current.FlowAcceleration()
    for sample in sequence:assert a.step(*sample)==b.step(*sample),case
    checks[case]=len(sequence)
old.least_squares=old_solver;current.least_squares=new_solver
timings={n:[] for n in ['frozen_generic','quadratic_fastpath']}
for repetition in range(3):
    order=[('frozen_generic',old.FlowAcceleration),('quadratic_fastpath',current.FlowAcceleration)]
    if repetition%2:order.reverse()
    for name,constructor in order:
        for sequence in sequences.values():
            observer=constructor()
            for sample in sequence:
                start=time.perf_counter();observer.step(*sample);timings[name].append((time.perf_counter()-start)*1000)
summary={n:dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95))) for n,v in timings.items()}
result=dict(exact_coefficients_and_audits_by_case=checks,nonlinear_solver_calls=calls,
    all_fixed_sequence_outputs_exact=True,timing=summary,RAW_or_GPU_started=False,labels_read=False,
    scope='Observer CPU cost only on fixed actual/cached states. No new point accuracy or pipeline timing claim.')
(OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(OUT/'source_before.py').write_bytes(frozen.read_bytes());(OUT/'source_after.py').write_bytes((ROOT/'live_flow_acceleration.py').read_bytes())
print(json.dumps(result),flush=True)
