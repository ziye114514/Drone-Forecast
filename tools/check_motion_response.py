"""Physical response controls: acceleration endpoint, maturity and invariance."""
from pathlib import Path
import os,sys,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
import numpy as np
from curved_flow_motion import CurvedFlowAcceleration
from adaptive_force_memory import AdaptiveForceMemory,kernel
from thrust_trend_forecast import ThrustTrendForecast

def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);result={}
    model=CurvedFlowAcceleration();last=None;dt=1/30;a0=np.array([40.,10.]);j=np.array([2200.,-100.]);v0=np.array([60.,10.])
    for i in range(13):
        t=i*dt;p=np.array([400.,300.])+v0*t+.5*a0*t*t+j*t**3/6
        body=dict(detected=True,center=p.tolist(),bbox=[p[0]-20,p[1]-15,p[0]+20,p[1]+15])
        c=dict(position=p.tolist(),velocity=(v0+a0*t+.5*j*t*t).tolist(),acceleration=[0.,0.])
        mid=t-dt/2;obs=dict(valid=i>0,midpoint_t=mid,velocity=(v0+a0*mid+.5*j*mid*mid).tolist(),velocity_covariance=np.eye(2).tolist())
        value,diagnostic=model.step(t,c,body,obs);last=(value,diagnostic)
    assert last[1]['reason']=='qualified_current_curve_acceleration',last
    error=np.linalg.norm(np.asarray(last[0]['acceleration'])-(a0+j*t));assert error<20.,error
    result['known_jerk_current_acceleration_error_px_s2']=float(error)
    result['jerk_not_asserted_as_future_command']=last[1]['jerk_not_extrapolated_into_future']
    # A source timestamp after the cutoff is not accepted.
    try:model.step(t+dt,c,body,dict(obs,midpoint_t=t+2*dt))
    except ValueError:result['future_velocity_rejected']=True
    else:raise AssertionError('future velocity accepted')
    a=np.array([1000.,0.]);f=dict(active=True,input_u=[1.,0.],effective_gain_px_s2=1000.,acceleration_px_s2=a.tolist(),acceleration_correlation_s=.35)
    memory=AdaptiveForceMemory();h=.4;baseline={'0.4':(kernel(-1/.35,h)*a).tolist()};c=dict(position=[0.,0.],velocity=[0.,0.],acceleration=a.tolist())
    memory.step(0.,c,[0.,0.],baseline,f)
    for t in [.05,.1,.2,.3,.39]:
        memory.step(t,c,(.5*t*t*a).tolist(),{},f);assert memory.weight==0.
    memory.step(.4,c,(.5*.4*.4*a).tolist(),{},f);assert memory.weight>0.,memory.weight
    result['own_future_position_waits_for_maturity']=True
    memory=AdaptiveForceMemory();memory.weight=1.
    p,d=memory.step(1.,c,None,baseline,dict(f,active=False,input_u=None))
    assert p==baseline and not d['applied'];result['no_current_physical_input_preserves_previous_rollout']=True
    # Constant-memory endpoint: rotate the known term only, independently
    # compare against numerical quadrature. Unknown force stays fixed.
    known=np.array([100.,-200.]);unknown=np.array([60.,20.]);omega=.7;tau=.35;h=.4
    complex_known=complex(*known);base=kernel(-1/tau+1j*omega,h)*complex_known
    base=np.array([base.real,base.imag])+kernel(-1/tau,h)*unknown
    force=dict(f,effective_gain_px_s2=1.,input_u=known.tolist(),acceleration_px_s2=(known+unknown).tolist())
    normal=dict(applied=True,trend=dict(shrunk_angular_rate_rad_s=omega))
    memory=AdaptiveForceMemory();memory.weight=1.
    p,d=memory.step(1.,c,None,{'0.4':base.tolist()},force,normal)
    s=np.linspace(0,h,20001);z=np.exp(1j*omega*s)*complex_known;acc=np.column_stack((z.real,z.imag))+unknown
    expected=np.trapezoid((h-s)[:,None]*acc,s,axis=0)
    err=np.linalg.norm(np.asarray(p['0.4'])-expected);assert err<1e-6,err
    result['known_direction_and_fixed_unknown_force_integral_error_px']=float(err)
    # A stable f^2 creates zero amplitude correction; hidden/changed source
    # does not silently teach a persistent amplitude trend.
    trend=ThrustTrendForecast();selection=dict(source='pooled_flicker_representative_speed_hypothesis',relative_f2=1.,coherent=True)
    pooled=dict(qualified=True,mode_fraction=.8)
    for i in range(10):r=trend.observe(i/30,selection,pooled)
    assert r['qualified'] and r['log_thrust_rate_per_s']==0.
    p,d=trend.correct(baseline,f,None,r);assert p==baseline
    r=trend.observe(10/30,dict(selection,source='visible_frequency_squared_mean_hidden_equal_mean_hypothesis'),pooled)
    assert not r['qualified'];result['constant_amplitude_zero_and_partial_composition_excluded']=True
    from fast_force_response import FastForceResponse
    gain=FastForceResponse(gain_signal_to_noise=2.);gain.evidence_gain=True;gain.d=np.array([0.,200.]);gain.last_t=0.;gain.last_valid=0.
    for i in range(8):
        u=np.array([.15*i,-1.+.03*i]);gain.rows.append((i*.01,u,4000.*u+np.array([30.,1200.])))
    p,d=gain.step(.1,c,None,np.array([.2,-.9]),.8)
    assert d['gain_learned'] and abs(gain.gain-4000.)<1e-5,d
    first=gain.gain;p,d=gain.step(.133333,c,None,np.array([.2,-.9]),.8)
    assert not d['gain_learned'] and gain.gain==first
    result['same_matured_gain_rows_not_counted_as_new_evidence']=True
    gain.step(.8,c,None,None,0.);assert gain.evidence_gain
    result['explicit_gain_policy_survives_timestamp_gap']=True
    hover=FastForceResponse();hover.evidence_gain=True;hover.last_t=0.
    for i in range(8):hover.rows.append((i*.01,np.array([0.,-1.]),np.array([0.,0.])))
    hover.step(.1,c,None,np.array([0.,-1.]),.8);assert hover.gain==200.
    result['unexcited_hover_cannot_identify_gain']=True
    from endpoint_force_balance import EndpointForceBalance
    structure=dict(mode='axis_only',axis_xy=[0.,-1.],axis_support=dict(valid=True,confidence=.8))
    curve=dict(applied=True,reason='qualified_current_curve_acceleration',shrunk_acceleration=[700.,-300.],standard_error_proxy=[50.,80.])
    observation=dict(valid=True,midpoint_t=-.01,velocity=[0.,0.],velocity_covariance=np.eye(2).tolist())
    force=EndpointForceBalance();force.observe_flow(observation,True);force.observe_axis_motion(structure,dict(acceleration=curve))
    _,d=force.step(0.,dict(position=[0.,0.],velocity=[0.,0.],acceleration=[0.,0.]),[0.,0.],np.array([0.,-1.]),.8)
    assert d['endpoint_force_balance']['updated'] and np.allclose(d['acceleration_px_s2'],[700.,-300.]),d
    assert np.allclose(np.array(d['disturbance_px_s2'])+d['effective_gain_px_s2']*np.array(d['input_u']),[700.,-300.])
    result['new_period_retains_qualified_current_total_acceleration']=True
    force=EndpointForceBalance();force.observe_flow(observation,True)
    force.observe_axis_motion(structure,dict(acceleration=dict(curve,standard_error_proxy=[4000.,4000.])))
    _,d=force.step(0.,dict(position=[0.,0.],velocity=[0.,0.],acceleration=[0.,0.]),[0.,0.],np.array([0.,-1.]),.8)
    assert not d['endpoint_force_balance']['updated'];result['uncertain_endpoint_cannot_override_better_prior']=True
    memory=AdaptiveForceMemory(extend_calibrated=True);memory.weight=1.
    missing=dict(f,active=False,input_u=None,gain_ever_learned=True)
    p,d=memory.step(1.,c,None,baseline,missing,current_total_qualified=True,current_axis_qualified=True)
    assert d['applied'] and d['calibrated_gap_eligible'] and not d['current_frequency_invented']
    memory=AdaptiveForceMemory(extend_calibrated=True);memory.weight=1.
    p,d=memory.step(1.,c,None,baseline,missing,current_total_qualified=False,current_axis_qualified=True)
    assert not d['applied'] and p==baseline
    result['calibrated_gap_requires_current_motion_and_axis_without_inventing_frequency']=True
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
