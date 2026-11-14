"""Exact numerical and alternating CPU comparison with the frozen v4 solver."""
from pathlib import Path
import importlib.util,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion
from live_local_flow import velocity_coefficient
from summarize_live_pilot import read_rows
OUT=WORK/'output/fast_ci_20261026';OUT.mkdir(exist_ok=False)
frozen=WORK/'output/local_displacement_tight_20261026/source_snapshot/live_local_flow.py'
spec=importlib.util.spec_from_file_location('frozen_v4_ci',frozen);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
old=module.velocity_coefficient
cases=dict(side=WORK/'output/local_displacement_tight_20261026/actual_side',train2=WORK/'output/local_displacement_tight_20261026/actual_train2',test3=WORK/'output/heldout_motion_RAW_20261026')
inputs=[];replay={};branches=dict(flow_endpoint=0,position_endpoint=0,optimizer_fallback=0)
for case,run in cases.items():
    model=ScaleRolloutMotion(25.);observations={round(r['t'],6):r['observation'] for r in json.loads((WORK/'output/spatial_flow_20261026'/(case+'.json')).read_text('utf-8'))};exact=0
    for row in read_rows(run/'tracking.jsonl'):
        t=row['t'];c,valid,_=model.step(t,row['body']);expected=row['coefficient'] if case=='test3' else row['position_only_coefficient'];assert c==expected
        obs=observations[round(t,6)];P=model.kf.P[2:4,2:4].copy() if c is not None else None
        before,a=old(c,P,obs,t);after,b=velocity_coefficient(c,P,obs,t)
        assert before==after and a==b,dict(case=case,t=t)
        exact+=1
        if c is not None and obs['valid']:
            e=np.linalg.eigvalsh(np.linalg.inv(obs['velocity_covariance'])-np.linalg.inv(P))
            branch='flow_endpoint' if e[0]>=0 else ('position_endpoint' if e[-1]<=0 else 'optimizer_fallback');branches[branch]+=1
            inputs.append((c,P,obs,t))
    replay[case]=dict(exact_coefficient_and_audit_ticks=exact)
# Exercise both endpoints, equal matrices, and non-dominant complementary axes.
c=dict(position=[40.,60.],velocity=[100.,-20.],acceleration=[0.,0.])
for P,R in [(np.eye(2)*100,np.eye(2)*25),(np.eye(2)*25,np.eye(2)*100),(np.eye(2)*25,np.eye(2)*25),(np.diag([1.,100.]),np.diag([100.,1.]))]:
    obs=dict(valid=True,velocity=[110.,-30.],velocity_covariance=R.tolist(),midpoint_t=1.)
    assert old(c,P,obs,1.)==velocity_coefficient(c,P,obs,1.)
for f in [old,velocity_coefficient]:
    for sample in inputs[:10]:f(*sample)
timings={n:[] for n in ['frozen_optimizer','dominance_shortcut']}
for repetition in range(5):
    order=[('frozen_optimizer',old),('dominance_shortcut',velocity_coefficient)]
    if repetition%2:order.reverse()
    for n,f in order:
        for sample in inputs:
            start=time.perf_counter();f(*sample);timings[n].append((time.perf_counter()-start)*1000)
summary={n:dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95))) for n,v in timings.items()}
report=dict(replay=replay,branches=branches,known_endpoint_and_complementary_axis_cases_exact=True,
    actual_coefficients_and_audits_exact=True,timing=summary,alternating_fixed_five_pairs=True,
    source_frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),source_new_sha256=hashlib.sha256((ROOT/'core/live_local_flow.py').read_bytes()).hexdigest(),
    RAW_or_GPU_started=False,labels_read=False,scope='CPU solver cost only on fixed current states; not whole-pipeline latency or a new accuracy gain.')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'source_new.py').write_bytes((ROOT/'core/live_local_flow.py').read_bytes());(OUT/'source_frozen.py').write_bytes(frozen.read_bytes())
print(json.dumps(report),flush=True)
