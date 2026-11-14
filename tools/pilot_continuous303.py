"""Continuous state vs exact same geometry/discovery with per-window reset."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from continuous_phase_filter import initial,feed,predict
from continuous_rotor_phase import ContinuousRotorPhase
from pure_visibility170 import window
from replay_joint236 import source
from rotor_torque_innovation import RotorTorqueInnovation,adapt
from physical_publication import evaluate


def mechanisms(dll):
    tt=np.arange(10000000,12000001,100,dtype=np.int64);dt=(tt-tt[0])/1e6;omega=2*np.pi*260;alpha=7.
    z=.3+omega*dt+.5*alpha*dt*dt;w=np.ones(len(tt))
    whole=feed(initial(tt[0],.3,omega,alpha),tt,z,w,dll=dll)
    chunked=initial(tt[0],.3,omega,alpha)
    for ids in np.array_split(np.arange(len(tt)),61):chunked=feed(chunked,tt[ids],z[ids],w[ids],dll=dll)
    assert np.max(abs(whole-chunked))<1e-10
    assert abs(whole[2]-(omega+alpha*dt[-1]))<1e-6
    return dict(native_packetization_invariant=True,instantaneous_rate_with_nonzero_acceleration_correct=True,
        synthetic_end_rate_rad_s=float(whole[2]),no_current_validation_in_state_fit=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=('13','15','16'),required=True);p.add_argument('--batch',default='continuous305_20261109');p.add_argument('--policy',default='cohelix_dynamics',choices=('mixture','cohelix_dynamics'));args=p.parse_args();n=args.recording
    OUT=ROOT.parent/f'output/{args.batch}/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    files=['native/continuous_phase303.cpp','continuous_phase_filter.py','continuous_rotor_phase.py','tools/pilot_continuous303.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files};dll=ROOT.parent/f'output/{args.batch}/build/Release/continuous_phase.dll'
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        past_automatic_discovery='joint242 current accepted observations, never future',
        source_geometry_before_current_validation=True,no_recording_specific_rules=True,no_parameter_search=True,
        validation_thresholds_not_lowered=True,continuous_state_in_same_reference_plane=True,
        dynamics_policy=args.policy,initial_acceleration_is_zero_prior_not_short_fit_measurement=True,
        published_dynamics_source='https://arxiv.org/html/2610.31384v1#S2',
        no_current_labels_or_future_events_used=True,not_using_true_RPM_or_IMU=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    checks=mechanisms(dll);rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
    seeds=[json.loads(s) for s in (ROOT.parent/f'output/joint242_20261108/recording{n}_10_16.01/periods.jsonl').read_text('utf8').splitlines()]
    cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
    models={a:ContinuousRotorPhase(dll,retain_phase=a=='continuous',filter_policy=args.policy) for a in ('continuous','reset')}
    torque={a:RotorTorqueInnovation() for a in models};periods={a:[] for a in models};cost={a:[] for a in models};out={a:[] for a in models}
    previous=None
    for i,row in enumerate(rows):
        t=row['t'];events=window(cache,t)
        mid=(round(t*1e6)-33333+33333//2)/1e6
        candidates=[j for j in range(i) if rows[j]['t']<=mid+1e-9];j=candidates[-1] if candidates else None
        past=None if j is None else previous if j==i-1 else window(cache,rows[j]['t'])
        for arm,m in models.items():
            tick=time.perf_counter();u,q,meta=m.step(row,events,None if j is None else rows[j],None if j is None else seeds[j],past)
            period=dict(t=t,u=None if u is None else u.tolist(),quality=q,meta=meta);periods[arm].append(period)
            info=torque[arm].observe(row,period);new=adapt(row,info);new['physical']=evaluate(new,[.1,.2,.4]) if new['coefficient'] else {};out[arm].append(new)
            cost[arm].append(1000*(time.perf_counter()-tick))
        previous=events
        if i%30==0:
            print(n,i,{a:periods[a][-1]['meta']['qualified_current_rings'] for a in models},flush=True)
            (OUT/'progress.json').write_text(json.dumps(dict(frame=i,t=t,confirmed={a:sum(r['meta']['qualified_current_rings']>0 for r in periods[a]) for a in models})),'utf8')
    results={}
    for arm,rs in periods.items():
        (OUT/f'{arm}_periods.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rs),'utf8')
        (OUT/f'{arm}_torque.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in out[arm]),'utf8')
        results[arm]=dict(frames=len(rs),ring_histogram={str(k):sum(r['meta']['qualified_current_rings']==k for r in rs) for k in range(5)},
            phase_ms_mean=float(np.mean(cost[arm])),phase_ms_p95=float(np.percentile(cost[arm],95)),
            prior_births=sum(len(r['meta'].get('born_slots',[])) for r in rs),
            informative_torque_changes=sum(abs(r['rotor_torque_innovation'].get('drive_change',0))>1e-12 for r in out[arm]))
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,checks=checks,protocol=protocol),indent=2),'utf8')
    print(results,flush=True)
