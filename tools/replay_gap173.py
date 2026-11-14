"""One gap-consistency correction; frozen172 observations, no parameter search."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from relative_rotor_observer import RelativeRotorForceObserver
from coherent_gap_observer import CoherentGapObserver,ShortGapObserver
from physical_publication import evaluate
from score_rotor164 import score
OUT=ROOT.parent/'output/gap173_20261107/verified'

def mechanism():
    result={}
    for cls in (RelativeRotorForceObserver,CoherentGapObserver,ShortGapObserver):
        m=cls();m._initialize(np.array([100.,200.]));m.t=0.;m.seen=0.;m.clone_t=0.;m.observations=6
        m.x[2:6]=[30.,120.,100.,-1700.];m._clone_position()
        c=dict(position=m.x[:2].tolist(),velocity=m.x[2:4].tolist())
        f=m._force(False,m.x[4:6]+m.x[m.gain_index]*m.u)
        if cls is ShortGapObserver:f['acceleration_correlation_s']=cls.gap_correlation_s
        expected=np.array(evaluate(dict(coefficient=c,force=f),[.1])['0.1'])
        # RAW cutoffs are integer microseconds, as required by transition().
        for t in (.033333,.066666,.1):
            got,_,_,_=m.step(t,dict(detected=False),None,input_meta={})
            assert np.linalg.eigvalsh(m.P).min()>-1e-5
        error=float(np.linalg.norm(np.array(got['position'])-expected))
        if cls in (CoherentGapObserver,ShortGapObserver):assert error<1e-7,error
        result[cls.__name__]=dict(position_disagreement_with_prior_published_expectation_px=error,
            multiple_missing_ticks_not_multiple_measurements=m.observations==6,
            seen_not_advanced=m.seen==0.,covariance_min_eigenvalue=float(np.linalg.eigvalsh(m.P).min()))
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out-name',default='verified_shortgap');args=parser.parse_args()
    if Path(args.out_name).name!=args.out_name:raise ValueError('One new leaf name required')
    OUT=OUT.parent/args.out_name
    OUT.mkdir(parents=True,exist_ok=False);summary=dict(mechanism=mechanism(),cases={},NN_used=False,new_front_inference=False,
        source_time_horizon_s=.4,parameters_searched=False)
    files=['coherent_gap_observer.py','relative_rotor_observer.py','rotor_force_observer.py','physical_publication.py','singer_position_state.py','tools/replay_gap173.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for n in files:
        p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    for name,source in [('side','final172/side_fixed.jsonl'),('train2','final_regional172/train2_fixed.jsonl')]:
        path=ROOT.parent/'output/geometry170_20261106'/source;rs=[json.loads(s) for s in path.read_text('utf-8').splitlines()]
        summary['cases'][name]={}
        for arm,cls in [('control',RelativeRotorForceObserver),('coherent_gap',CoherentGapObserver),('short_gap',ShortGapObserver)]:
            m=cls();output=[];delta=0.;missing=[];capped=0
            for old in rs:
                r=copy.deepcopy(old);u=r['force']['input_u'];meta=r['period_selection']
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
                c,v,f,s=m.step(r['t'],r['body'],None if u is None else np.array(u),.7 if u is not None else 0.,
                    allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'])
                if arm=='control' and c:
                    for k in ('position','velocity','acceleration'):delta=max(delta,float(np.max(abs(np.array(c[k])-old['coefficient'][k]))))
                if m.P is not None:
                    assert np.linalg.eigvalsh(m.P).min()>-1e-5
                    capped+=np.linalg.norm(m.x[4:6]+m.x[m.gain_index]*m.u)>3000.
                if s.get('gap_transition_uses_published_net_force_prior'):missing.append(r['t'])
                r.update(coefficient=c,valid=v,force=f,state=s);r['physical']=evaluate(r,[.1,.2,.4]) if c else {};output.append(r)
            if arm=='control':assert delta<1e-8,delta
            dest=OUT/f'{name}_{arm}.jsonl';dest.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output),'utf-8')
            summary['cases'][name][arm]=dict(control_parity_max=delta,missing_position_cutoffs=missing,
                latent_acceleration_above_output_cap_frames=int(capped),rows=len(output),valid=sum(r['valid'] for r in output))
    # All causal outputs are frozen before opening any development labels.
    for name in summary['cases']:
        labels=ROOT.parent/('output/dense_labels_side_110_1195/labels.json' if name=='side' else 'output/dense_labels_train2_40_46/labels.json')
        queries=np.arange(110.4,114.0001,.2) if name=='side' else np.arange(43.2,44.4001,.2)
        for arm,r in summary['cases'][name].items():
            r['score']=score(OUT/f'{name}_{arm}.jsonl',labels,queries)
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    summary.update(source_sha256=hashes,source_changed=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    print(json.dumps(dict(mechanism=summary['mechanism'],scores={n:{a:dict(mean=r['score']['error']['mean'],n=r['score']['available'],valid=r['valid']) for a,r in arms.items()} for n,arms in summary['cases'].items()})))
