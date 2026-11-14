"""Same frozen current phase observations; only the gain information policy."""
from pathlib import Path
import sys,json,copy,hashlib,numpy as np,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from phase_gain_observer import PhaseGainObserver
from physical_publication import evaluate
from run_rotor_only import serial
BASE=ROOT.parent/'output/gain186_20261107'


def mechanism():
    cases=[PhaseGainObserver(),PhaseGainObserver(default_gain=200/7.)]
    constant=PhaseGainObserver();dt=.033333;p=np.array([100.,200.]);v=np.zeros(2);previous=np.array([0.,-.8]);maxerror=0.
    for i in range(120):
        t=i*dt;u=np.array([.4*np.sin(3*t),-.8+.15*np.cos(4*t)])
        if i:p+=v*dt+.5*dt*dt*(800*previous+np.array([0.,700.]));v+=dt*(800*previous+np.array([0.,700.]))
        body=dict(detected=True,center=p.tolist(),measurement_source='synthetic_position',center_provisional=False)
        cs=[]
        for model,scale in zip(cases,[1.,7.]):
            um=u*scale;meta=dict(source='current_unordered_visible_ring_phase_sum',observed_components={'ring':um.tolist()},input_sigma_proxy=0.)
            c,_,f,_=model.step(t,body,um,.7,input_meta=meta,allow_gain_learning=True);cs.append(c)
        for key in ['position','velocity','acceleration']:maxerror=max(maxerror,float(np.max(abs(np.asarray(cs[0][key])-cs[1][key]))))
        meta=dict(source='current_unordered_visible_ring_phase_sum',observed_components={'ring':[0.,-.8]},input_sigma_proxy=0.)
        constant.step(t,dict(body,center=[100+10*t,200+20*t]),np.array([0.,-.8]),.7,input_meta=meta,allow_gain_learning=True)
        previous=u
    assert maxerror<1e-5,maxerror
    assert abs(constant.x[6]-200)<1e-5,constant.x[6]
    # Gauge broadening cannot change any current net-acceleration covariance.
    u=np.array([.2,-.5]);g=np.zeros(11);g[4:6]=-u;g[6]=1.;H=np.zeros((2,11));H[:,4:6]=np.eye(2);H[:,6]=u
    assert np.linalg.norm(H@np.outer(g,g)@H.T)<1e-12
    return dict(input_unit_rescaling_max_motion_delta=maxerror,constant_input_gain_change=float(constant.x[6]-200),
        current_net_force_covariance_gauge_invariant=True,synthetic_true_gain=800.,synthetic_estimated_gain=float(cases[0].x[6]),
        synthetic_gain_learned=cases[0].learned)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='recording15');args=parser.parse_args();OUT=BASE/args.case;OUT.mkdir(exist_ok=False)
    files=['phase_gain_observer.py','rotor_force_observer.py','relative_rotor_observer.py','layout_reference_observer.py','tools/replay_gain186.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for name in files:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    checks=mechanism();source=ROOT.parent/f'output/phase185_20261107/{args.case}'
    rows=[json.loads(s) for s in (source/'control176.jsonl').read_text('utf-8').splitlines()]
    periods=[json.loads(s) for s in (source/'periods.jsonl').read_text('utf-8').splitlines()]
    model=PhaseGainObserver();output=[]
    for old,phase in zip(rows,periods):
        assert old['t']==phase['t'];r=copy.deepcopy(old);meta=phase['meta'];u=None if phase['u'] is None else np.asarray(phase['u'])
        if r['state'].get('two_point_initialization_selected_by_partial_entry'):model.two_point_init=True
        c,v,f,s=model.step(r['t'],r['body'],u,phase['quality'],allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'])
        r.update(coefficient=c,valid=v,force=f,state=s,phase_bank_input=meta);r['physical']=evaluate(r,[.1,.2,.4]) if c else {};output.append(r)
    (OUT/'adaptive_gain.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in output),'utf-8')
    result=dict(checks=checks,frames=len(output),gain_updates=sum(r['force'].get('gain_learned',False) for r in output),
        final_gain=float(model.x[6]) if model.x is not None else None,source_sha256=hashes,parameters_searched=False,NN_used=False,
        labels_during_inference=False,RAW_FPS_measured=False)
    (OUT/'completion.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
