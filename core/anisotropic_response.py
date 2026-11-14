"""Conditional image-domain disk drag, with observed tangential response.

Faessler et al. RAL2018 model body-axis drag as R D R^T. A headless disk
permits only a two-direction IMAGE approximation here: tangent response can
be learned after eliminating all normal thrust, whereas normal drag remains
the existing prior. No metric coefficient, hidden RPM or thrust is measured.

For a=A*n+b-K*v, held A,b imply da/dt=A*dn/dt-K*a-dK/dt*v.
The dK term is required when the disk turns. The current p/v/a is preserved.
"""
import copy
import numpy as np
from scipy.linalg import expm
from identifiable_tangent_memory import IdentifiableTangentMemory
from axis_amplitude_bridge import current_axis_qualified


class AnisotropicResponse:
    def __init__(self):self.memory=IdentifiableTangentMemory();self.observations=None

    def step(self,row):
        obs=row.get('state',{}).get('observations')
        if obs is not None and self.observations is not None and obs<self.observations:self.memory=IdentifiableTangentMemory()
        self.observations=obs;info=self.memory.observe(row);r=copy.deepcopy(row)
        current=bool(row['coefficient'] and row['motion'].get('current_observation') and current_axis_qualified(row['structure']))
        n=np.asarray(row['structure'].get('axis_xy') or [],float)
        if not current or not info.get('qualified') or n.shape!=(2,) or np.linalg.norm(n)<1e-6:
            r['anisotropic_response']=dict(applied=False,past_fit=info);return r
        n=n/np.linalg.norm(n)
        if n[1]>0:n=-n
        def apply(packet):
            f=packet['force'];normal=packet.get('normal_turn_forecast') or {}
            beta=1/float(f['acceleration_correlation_s'])
            if abs(beta-1/.35)>1e-7:return
            drive=np.asarray(f.get('normal_turn_vector_px_s2') or [0.,0.],float)
            f['anisotropic_disk_response']=dict(normal_xy=n.tolist(),tangent_response_s_inv=info['effective_response_s_inv'],
                normal_response_s_inv=beta,normal_response_is_prior=True,
                normal_drive_px_s2=float(n@drive),current_state_unchanged=True,
                disk_axis_is_headless=True,measured_metric_drag=False,
                unknown_force_owner_unchanged=True)
            for part in f.get('response_model_mixture',[]):apply(part)
        apply(r);r['anisotropic_response']=dict(applied=True,past_fit=info,heading_used=False)
        return r


def rollout(c,f,normal,horizons,steps=64):
    """RK4 of the full rotating-drag derivative, including existing j state."""
    d=f.get('anisotropic_disk_response')
    amplitude_rate=f.get('visible_force_rate') or {}
    if d is None:
        beta=1/float(f['acceleration_correlation_s']);n=np.asarray(amplitude_rate['normal_xy'],float)
        drive=f.get('normal_turn_vector_px_s2') if normal.get('applied') else None
        if drive is None:drive=np.asarray(f['input_u'])*float(f['effective_gain_px_s2']) if f.get('active') and f.get('input_u') is not None else [0.,0.]
        d=dict(normal_xy=n,tangent_response_s_inv=beta,normal_response_s_inv=beta,normal_drive_px_s2=float(n@drive))
    n0=np.asarray(d['normal_xy'],float)
    tangent=float(d['tangent_response_s_inv']);axial=float(d['normal_response_s_inv'])
    A=float(d['normal_drive_px_s2']);trend=normal.get('trend') or {}
    omega=float(trend.get('shrunk_angular_rate_rad_s',0.)) if normal.get('applied') else 0.
    alpha=float(trend.get('angular_acceleration_rad_s2',0.)) if normal.get('applied') else 0.
    tau=trend.get('angular_acceleration_correlation_s',trend.get('torque_innovation_correlation_s'))
    j=np.asarray(f.get('external_unknown_rate_px_s3') or [0.,0.],float)
    jt=float(f.get('external_rate_correlation_s',.35))
    initial=np.r_[c['position'],c['velocity'],f['acceleration_px_s2'],j]
    if not alpha:
        # Exact constant-rate solution in the rotating disk frame. Its
        # world-fixed external force rotates oppositely in these coordinates.
        # This includes dK/dt automatically, rather than dropping that term.
        J=np.array([[0.,-1.],[1.,0.]]);W=omega*J
        K=tangent*np.eye(2)+(axial-tangent)*np.outer(n0,n0)
        M=np.zeros((10,10));M[:2,:2]=-K-W;M[:2,2:4]=np.eye(2);M[:2,8]=n0
        M[2:4,2:4]=-W;M[2:4,4:6]=np.eye(2)
        M[4:6,4:6]=-np.eye(2)/jt-W
        M[6:8,:2]=np.eye(2);M[6:8,6:8]=-W
        M[8,9]=1.;M[9,9]=-1/float(amplitude_rate.get('correlation_s',.35))
        v=initial[2:4];a=initial[4:6]
        z=np.r_[v,a-A*n0+K@v,j,np.zeros(2),A,float(amplitude_rate.get('amplitude_rate_px_s3',0.))];result={}
        for horizon in horizons:
            h=float(horizon)
            if not np.isfinite(h) or not 0<=h<=1:raise ValueError('Horizon must be within0..1s')
            value=expm(M*h)@z;co=np.cos(omega*h);si=np.sin(omega*h)
            R=np.array([[co,-si],[si,co]])
            result[str(h)]=(initial[:2]+R@value[6:8]).tolist()
        return result
    def derivative(t,x):
        if tau:
            angle=omega*t+alpha*tau*(t-tau*(-np.expm1(-t/tau)));rate=omega+alpha*tau*(-np.expm1(-t/tau))
        else:angle=omega*t+.5*alpha*t*t;rate=omega+alpha*t
        co=np.cos(angle);si=np.sin(angle);n=np.array([co*n0[0]-si*n0[1],si*n0[0]+co*n0[1]])
        nd=rate*np.array([-n[1],n[0]])
        K=tangent*np.eye(2)+(axial-tangent)*np.outer(n,n)
        Kd=(axial-tangent)*(np.outer(nd,n)+np.outer(n,nd))
        ar=float(amplitude_rate.get('amplitude_rate_px_s3',0.));at=float(amplitude_rate.get('correlation_s',.35))
        As=A+ar*at*(-np.expm1(-t/at));Ad=ar*np.exp(-t/at)
        return np.r_[x[2:4],x[4:6],As*nd+Ad*n-K@x[4:6]-Kd@x[2:4]+x[6:8],-x[6:8]/jt]
    result={}
    for horizon in horizons:
        h=float(horizon)
        if not np.isfinite(h) or not 0<=h<=1:raise ValueError('Horizon must be within0..1s')
        x=initial.copy();dt=h/steps
        for i in range(steps):
            t=i*dt;k1=derivative(t,x);k2=derivative(t+dt/2,x+dt*k1/2);k3=derivative(t+dt/2,x+dt*k2/2);k4=derivative(t+dt,x+dt*k3)
            x+=dt*(k1+2*k2+2*k3+k4)/6
        result[str(h)]=x[:2].tolist()
    return result
