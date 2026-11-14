"""Test current plane values and directional unknown response, no label fit."""
from pathlib import Path
import sys,json,copy,hashlib,numpy as np,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from normal_response_observer import NormalResponseObserver
from physical_publication import evaluate
from run_rotor_only import serial
BASE=ROOT.parent/'output/normal187_20261107'


def mechanism():
    models=[NormalResponseObserver(),NormalResponseObserver()];p=np.array([100.,200.]);v=np.zeros(2);oldn=np.array([0.,-1.]);oldp=p.copy();dt=.033333
    for i in range(150):
        t=i*dt;n=np.array([.3*np.sin(3*t),-np.sqrt(1-(.3*np.sin(3*t))**2)])
        if i:p+=dt*v+.5*dt*dt*(1000*oldn+[0,1000]);v+=dt*(1000*oldn+[0,1000])
        delta=p-oldp;body=dict(detected=True,center=p.tolist(),measurement_source='filtered_native_translation')
        motion=dict(current_observation=True,dt_s=None if not i else dt,estimate=dict(dx=float(delta[0]),dy=float(delta[1]),score=1.,peak_at_search_boundary=False))
        for j,model in enumerate(models):
            plane=dict(axis_xy=(n if not j else np.array([0.,-1.])).tolist(),mode='axis_only',axis_support=dict(valid=True,confidence=.7))
            model.step(t,body,None,0.,input_meta=dict(observed_components={},input_sigma_proxy=0.),motion=motion,structure=plane)
        oldn=n;oldp=p.copy()
    assert abs(models[1].x[11])<1e-9
    assert np.linalg.eigvalsh(models[0].P).min()>-1e-5
    # Visibility bookkeeping moves both d and its axial decomposition, so
    # known+unknown net motion and external e stay unchanged for axial input.
    m=models[0];estimated_response=float(m.x[11]);before=m.x[4:6]+m.x[6]*m.u;e_before=m.x[4:6]-m.x[11]*m.normal
    added=.1*m.normal;m._set_input(added,transfer_delta=added,transfer_cov=np.eye(2)*.001)
    assert np.allclose(m.x[4:6]+m.x[6]*m.u,before)
    assert np.allclose(m.x[4:6]-m.x[11]*m.normal,e_before)
    return dict(constant_normal_does_not_identify_response=True,synthetic_response_true=1000.,
        synthetic_response_estimate=estimated_response,visibility_transfer_preserves_net_and_external=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='recording15');parser.add_argument('--out-name');args=parser.parse_args()
    BASE.mkdir(exist_ok=True);OUT=BASE/(args.out_name or args.case);OUT.mkdir(exist_ok=False)
    files=['normal_response_observer.py','physical_publication.py','rotor_force_observer.py','relative_rotor_observer.py','layout_reference_observer.py','tools/replay_normal187.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for n in files:
        p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    checks=mechanism();source=ROOT.parent/f'output/phase185_20261107/{args.case}'
    input_path=source/'control176.jsonl'
    if not input_path.exists():
        input_path=ROOT.parent/f'output/layout176_20261107/integrated/{args.case}_fixed.jsonl'
    rows=[json.loads(s) for s in input_path.read_text('utf-8').splitlines()]
    periods=[json.loads(s) for s in (source/'periods.jsonl').read_text('utf-8').splitlines()] if (source/'periods.jsonl').exists() else [dict(u=None,quality=0.,meta=dict(observed_components={},input_sigma_proxy=0.)) for r in rows]
    results={}
    for arm in ['normal_legacy_input','normal_no_period','normal_phase','constant_normal_control']:
        model=NormalResponseObserver();output=[];anchor=None;minimum=0.
        for old,phase in zip(rows,periods):
            r=copy.deepcopy(old);s=copy.deepcopy(r['structure'])
            if arm=='constant_normal_control' and s.get('axis_xy') is not None:
                if anchor is None:anchor=list(s['axis_xy'])
                s['axis_xy']=anchor
            if arm=='normal_legacy_input':
                meta=old['period_selection'];u=None if old['force']['input_u'] is None else np.asarray(old['force']['input_u']);q=.7 if u is not None else 0.
            elif arm=='normal_phase':meta=phase['meta'];u=None if phase['u'] is None else np.asarray(phase['u']);q=phase['quality']
            else:meta=dict(observed_components={},input_sigma_proxy=0.,learning_input_current=False);u=None;q=0.
            if r['state'].get('two_point_initialization_selected_by_partial_entry'):model.two_point_init=True
            c,v,f,state=model.step(r['t'],r['body'],u,q,allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'],structure=s)
            if model.P is not None:minimum=min(minimum,float(np.linalg.eigvalsh(model.P).min()))
            r.update(coefficient=c,valid=v,force=f,state=state,normal_turn_forecast=state.get('normal_turn_forecast'))
            r['physical']=evaluate(r,[.1,.2,.4]) if c else {};output.append(r)
        assert minimum>-1e-4,minimum
        (OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in output),'utf-8')
        results[arm]=dict(frames=len(output),normal_updates=sum(r['state'].get('normal_response_state_updated',False) for r in output),
            future_turns=sum((r.get('normal_turn_forecast') or {}).get('applied',False) for r in output),
            final_response=float(model.x[11]) if model.x is not None else None,covariance_min_eigenvalue=minimum)
    (OUT/'completion.json').write_text(json.dumps(dict(checks=checks,results=results,source_sha256=hashes,parameters_searched=False,NN_used=False,RAW_FPS_measured=False),indent=2),'utf-8');print(json.dumps(results))
