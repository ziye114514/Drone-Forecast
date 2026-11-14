"""Focused causal/gauge checks; no training, dataset labels or GPU."""
from pathlib import Path
import sys,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nn_rotor_gate import CausalRotorResidualGate,evaluate_gated
from nn_rotor_residual import ObservedForceGauge
from adaptive_force_memory import kernel
from physical_publication import evaluate


def row(t,speed):
    return dict(t=t,valid=True,force_observed_position=[speed*t,0.],
        coefficient=dict(position=[speed*t,0.],velocity=[0.,0.]),
        force=dict(acceleration_px_s2=[0.,0.],acceleration_correlation_s=.35,active=False),
        image_motion=dict(observation={}),physical={})


def stream(speed,stop=31):
    gate=CausalRotorResidualGate();out=[]
    for t in np.arange(stop)/30:
        base=row(float(t),speed);candidate=copy.deepcopy(base)
        candidate['force']['acceleration_px_s2']=[10/kernel(-1/.35,.4),0.]
        out.append(gate.step(base,candidate,True))
    return out


if __name__=='__main__':
    good=stream(25.);bad=stream(-25.)
    assert all(r['rotor_residual_fusion']['weight']==0 for r in good if r['t']<.4-1e-9)
    assert any(r['rotor_residual_fusion']['applied'] for r in good if r['t']>=.4)
    assert all(r['rotor_residual_fusion']['weight']==0 for r in bad)
    assert [r['physical'] for r in stream(25.,15)]==[r['physical'] for r in good[:15]]
    r=good[-1];assert evaluate_gated(r,[0.])['0.0']==evaluate(r,[0.])['0.0']
    gate=CausalRotorResidualGate();b=row(0.,25.);c=copy.deepcopy(b)
    assert gate.step(b,c,False)['physical']==evaluate(b,[.1,.2,.4])
    gauge=ObservedForceGauge()
    def step(parts):
        u=np.sum(list(parts.values()),axis=0) if parts else None
        return gauge.step(u,dict(observed_components=parts))[0]
    assert np.allclose(step({'a':[0.,-1.],'b':[0.,-1.]}),[0.,0.])
    assert np.allclose(step({'a':[0.,-1.2],'b':[0.,-1.]}),[0.,-.2])
    assert np.allclose(step({'a':[0.,-1.2]}),[0.,-.2])
    assert np.allclose(step({'a':[0.,-1.2],'b':[0.,-1.]}),[0.,-.2])
    print('PASS: delayed learning, negative evidence, prefix, origin, no evidence, visibility gauge')
