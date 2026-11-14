"""Fixed untrained derivative prior on frozen causal rotor-only observations."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from layout_reference_observer import LayoutReferenceObserver
from integrated_force_rate import ForceRateLayoutObserver as SmoothLayoutForceObserver
from physical_publication import evaluate
from run_rotor_only import SOURCE_FILES,serial
BASE=ROOT.parent/'output/rate184_20261107'


def mechanism():
    m=SmoothLayoutForceObserver();m._initialize(np.array([100.,200.]));m.t=0.;m.seen=0.;m.clone_t=0.;m.observations=6
    m.u=np.array([.3,-.4]);m.components={'visible':m.u.copy()};m.x[2:6]=[40.,100.,200.,-700.];m.x[11:13]=[400.,-900.];m._clone_position()
    F,Q=m._transition(.4);c=dict(position=m.x[:2],velocity=m.x[2:4],acceleration=m.x[4:6]+m.x[6]*m.u)
    f=m._force(True,c['acceleration']);point=evaluate(dict(coefficient=c,force=f),[.4])['0.4']
    assert np.allclose((F@m.x)[:2],point,atol=1e-8)
    F1,Q1=m._transition(.15);F2,Q2=m._transition(.25)
    assert np.allclose(F2@F1,F,atol=1e-8)
    assert np.allclose(F2@Q1@F2.T+Q2,Q,atol=1e-5)
    n=copy.deepcopy(m);n._set_input(np.zeros(2),transfer_delta=-m.u);n.components={}
    Fn,_=n._transition(.4)
    # Compare physical net acceleration rather than its bookkeeping coordinate.
    xn=Fn@n.x;xo=F@m.x
    assert np.allclose(xn[:4],xo[:4],atol=1e-8)
    assert np.allclose(xn[4:6]+xn[6]*n.u,xo[4:6]+xo[6]*m.u,atol=1e-8)
    assert np.allclose(xn[11:13],xo[11:13],atol=1e-8)
    assert np.linalg.eigvalsh(Q).min()>-1e-6
    return dict(rollout_transition_match=True,mean_and_covariance_semigroup=True,
        hidden_partition_preserves_motion_and_derivative=True,process_covariance_PSD=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--holdout',action='store_true');args=parser.parse_args()
    BASE.mkdir(exist_ok=True);OUT=BASE/('holdout' if args.holdout else 'development');OUT.mkdir(exist_ok=False)
    files=list(dict.fromkeys(SOURCE_FILES+['tools/replay_rate184.py']));hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for name in files:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(hypothesis='Integrated force-rate prior, same.35 integrated correlation/1000acceleration sigma, no tuning',
        NN_used=False,labels_during_inference=False,parameters_searched=False,holdout=args.holdout,front_unchanged=True,
        frozen_observation_replay=True,RAW_FPS_measured=False,source_sha256=hashes),indent=2),'utf-8')
    checks=mechanism();results={}
    def guard(event,items):
        if event=='open' and items and isinstance(items[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(items[0]))
            if p.name=='labels.json' or 'evaluation_data' in p.parts:raise RuntimeError('No labels during causal replay')
    sys.addaudithook(guard)
    sources={case:ROOT.parent/f'output/layout176_20261107/integrated/{case}_fixed.jsonl' for case in ('side','train2')}
    if args.holdout:sources={f'recording{n}':ROOT.parent/f'output/motion181_20261107/holdout/recording{n}_control176.jsonl' for n in [15,19]}
    for case,path in sources.items():
        data=[json.loads(s) for s in path.read_text('utf-8').splitlines()];results[case]={}
        for arm,cls in [('control176',LayoutReferenceObserver),('rate',SmoothLayoutForceObserver)]:
            m=cls();out=[];delta=0.;minimum=0.
            for old in data:
                r=copy.deepcopy(old);meta=r['period_selection'];u=r['force']['input_u']
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
                c,v,f,s=m.step(r['t'],r['body'],None if u is None else np.asarray(u),.7 if u is not None else 0.,
                    allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'])
                if arm=='control176' and c:
                    for k in ('position','velocity','acceleration'):delta=max(delta,float(np.max(abs(np.array(c[k])-r['coefficient'][k]))))
                if m.P is not None:minimum=min(minimum,float(np.linalg.eigvalsh(m.P).min()))
                r.update(coefficient=c,valid=v,force=f,state=s);r['physical']=evaluate(r,[.1,.2,.4]) if c else {};out.append(r)
            assert minimum>-1e-4,(case,arm,minimum)
            if arm=='control176':assert delta<1e-8,delta
            (OUT/f'{case}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in out),'utf-8')
            results[case][arm]=dict(frames=len(out),valid=sum(r['valid'] for r in out),control_replay_max_delta=delta,
                covariance_min_eigenvalue=minimum)
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    (OUT/'completion.json').write_text(json.dumps(dict(source_changed=False,checks=checks,results=results),indent=2),'utf-8');print(json.dumps(results))
