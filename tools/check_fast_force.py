"""Controlled mechanism checks; explicitly not an event-camera accuracy result."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from fast_force_response import FastForceResponse


def simulate(changing):
    model=FastForceResponse();p=np.zeros(2);v=np.zeros(2);dt=1/30
    true_gain=600.;d=np.array([0.,600.]);rows=[]
    for i in range(241):
        t=i*dt;tilt=.45*np.sin(2.5*(t-1)) if changing and t>=1 else 0.
        u=np.array([np.sin(tilt),-np.cos(tilt)]);a=true_gain*u+d
        c=dict(position=p.tolist(),velocity=v.tolist(),acceleration=[0.,0.])
        pred,diag=model.step(t,c,p.copy(),u,.8)
        rows.append(dict(t=t,acceleration=diag['acceleration_px_s2'],actual=a.tolist(),
            gain=diag['effective_gain_px_s2'],learned=diag['gain_learned'],frozen=diag['frozen_reason']))
        # Midpoint input gives a consistent continuous controlled trajectory.
        tm=t+dt/2;phi=.45*np.sin(2.5*(tm-1)) if changing and tm>=1 else 0.
        am=true_gain*np.array([np.sin(phi),-np.cos(phi)])+d
        p+=dt*v+.5*dt*dt*am;v+=dt*am
    return rows


def main():
    steady=simulate(False);moving=simulate(True)
    assert all(not r['learned'] for r in steady),'Unexcited hover must not identify gain'
    assert abs(steady[-1]['gain']-200)<1e-9
    assert abs(moving[-1]['gain']-600)<100,(moving[-1])
    onset=next(r for r in moving if r['t']>1.01)
    assert abs(onset['acceleration'][0])>0,'Current tilt must respond before delayed target matures'
    model=FastForceResponse();c=dict(position=[0,0],velocity=[0,0],acceleration=[3,4])
    pred,diag=model.step(0,c,[0,0],None,0)
    assert diag['acceleration_px_s2']==[3.,4.] and not diag['active']
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    result=dict(controlled_only=True,hover_gain_frozen=steady[-1]['gain'],
        changing_gain_final=moving[-1]['gain'],current_response_after_onset_s=onset['t']-1,
        final_acceleration_error=float(np.linalg.norm(np.array(moving[-1]['acceleration'])-moving[-1]['actual'])),
        actual_event_accuracy_measured=False)
    (out/'controlled.json').write_text(json.dumps(result,indent=2),'utf-8')
    (out/'controlled_trace.json').write_text(json.dumps(moving),'utf-8');print(json.dumps(result))


if __name__=='__main__':main()
