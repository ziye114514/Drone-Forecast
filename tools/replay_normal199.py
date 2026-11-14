"""One physical-response hypothesis; no parameter search or target fitting."""
from pathlib import Path
import sys,json,numpy as np,hashlib,argparse,time,copy
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from normal_dynamic_response import DynamicNormalResponseObserver
from physical_publication import evaluate
from run_rotor_only import serial
BASE=ROOT.parent/'output/normal199_20261107'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['14','15','16'],required=True);args=p.parse_args()
    out=BASE/f'recording{args.recording}';out.mkdir(parents=True,exist_ok=False)
    source=ROOT.parent/f'output/{"normal187_20261107/recording15_with_input/normal_legacy_input" if args.recording=="15" else "timing194_20261107/recording"+args.recording+"/plane191"}.jsonl'
    rr=[json.loads(s) for s in source.read_text('utf-8').splitlines()];m=DynamicNormalResponseObserver();result=[];minimum=0.;elapsed=[]
    for old in rr:
        r=copy.deepcopy(old);meta=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
        if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
        start=time.perf_counter();c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'],structure=r['structure']);elapsed.append(1000*(time.perf_counter()-start))
        if m.P is not None:minimum=min(minimum,float(np.linalg.eigvalsh(m.P).min()))
        r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'));r['physical']=evaluate(r,[.1,.2,.4]) if c else {};result.append(r)
    assert minimum>-1e-4,minimum
    (out/'dynamic.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in result),'utf-8')
    paths=['normal_dynamic_response.py','normal_damping_observer.py','tools/replay_normal199.py'];hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in paths}
    for n in paths:
        dst=out/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    d=dict(final_response=None if m.x is None else float(m.x[11]),covariance_min_eigenvalue=minimum,mean_observer_ms=float(np.mean(elapsed)),source_sha256=hashes,parameters_searched=False,labels_used_in_estimation=False)
    (out/'completion.json').write_text(json.dumps(d,indent=2),'utf-8');print(json.dumps(d))
