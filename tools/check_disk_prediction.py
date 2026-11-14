"""Mechanism checks: headless invariance, fixed-body wobble, and distinct temporal votes."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from normal_turn_forecast import NormalTurnForecast
from temporal_period_support import TemporalPeriodSupport
DTYPE=np.dtype([('x','u2'),('y','u2'),('t','i8'),('frequency','f4')])
def q(points,t,frequency=200.):
    a=np.zeros(len(points),DTYPE)
    a['x']=[p[0] for p in points];a['y']=[p[1] for p in points]
    a['t']=round(t*1e6)-100;a['frequency']=frequency
    return a
def structure(t,flip=False):
    angle=-np.pi/2+.3*t
    n=np.array([np.cos(angle),np.sin(angle)])*(-1 if flip else 1)
    return dict(axis_xy=n.tolist(),axis_support=dict(valid=True,confidence=.8,multiscale_spread_deg=1.,line_pca_disagreement_deg=.5),
        mode='axis_only',observed_model_ids=[3,1,0,2])
def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    a=NormalTurnForecast();b=NormalTurnForecast();stationary=NormalTurnForecast()
    c=dict(position=[0.,0.],velocity=[200.,0.],acceleration=[0.,0.])
    moving=dict(observation=dict(valid=True,velocity=[200.,0.]),acceleration={})
    resting=dict(observation=dict(valid=True,velocity=[0.,0.]),acceleration={})
    for i in range(11):
        t=i/30;sa=structure(t);sb=structure(t,flip=i%2==0);sb['observed_model_ids']=[2,0,3,1]
        left=a.observe(t,sa,c,moving);right=b.observe(t,sb,c,moving)
        stop=stationary.observe(t,sa,c,resting)
    assert left['qualified'] and right['qualified']
    partial=NormalTurnForecast()
    for i in range(8):
        s=structure(i/30);s['axis_support']['line_pca_disagreement_deg']=None
        partial.observe(i/30,s,c,moving)
    assert np.isclose(left['shrunk_angular_rate_rad_s'],right['shrunk_angular_rate_rad_s'],atol=1e-12)
    assert not stop['qualified']
    predictions={str(h):[10.*h,2.] for h in [.1,.2,.4]}
    force=dict(active=True,input_u=[0.,-1.],effective_gain_px_s2=200.,acceleration_correlation_s=.35)
    revised,info=a.correct(predictions,force,left);assert info['applied']
    # Independent numerical quadrature of just the rotating known component.
    h=.4;ts=np.linspace(0,h,10001);w=left['shrunk_angular_rate_rad_s']
    known=np.array([0.,-200.]);rot=np.column_stack([200*np.sin(w*ts),-200*np.cos(w*ts)])
    num=np.trapezoid((h-ts)[:,None]*np.exp(-ts[:,None]/.35)*(rot-known),ts,axis=0)
    assert np.allclose(np.asarray(revised['0.4'])-predictions['0.4'],num,atol=1e-8)
    zero=dict(left,shrunk_angular_rate_rad_s=0.)
    z,_=a.correct(predictions,force,zero);assert z==predictions
    bad,_=a.correct(predictions,dict(force,active=False),left);assert bad==predictions
    still,_=a.correct(predictions,force,stop);assert still==predictions
    wobble=NormalTurnForecast();wobble_ticks=0
    for i in range(91):
        t=i/30;s=structure(t);angle=-np.pi/2+.08*np.sin(2*np.pi*t)
        s['axis_xy']=[float(np.cos(angle)),float(np.sin(angle))]
        d=wobble.observe(t,s,c,resting);p,_=wobble.correct(predictions,force,d)
        assert p==predictions;wobble_ticks+=1
    # Current and earlier native pixels are disjoint, from a fixed test ROI.
    bbox=[100.,100.,200.,200.];base=dict(qualified=False,frequency_hz=None,pixels=8,reason='weak_or_common_mode')
    support=TemporalPeriodSupport()
    old=[(110+i,120) for i in range(8)];now=[(130+i,120) for i in range(8)]
    p,_=support.step(0.,q(old,0.,200.),bbox,[0.,0.],base);assert not p['qualified']
    p,d=support.step(.033333,q(now,.033333,202.),bbox,[0.,0.],base)
    assert p['qualified'] and p['frequency_hz']==202. and d['distinct_mode_pixels']==16
    repeats=TemporalPeriodSupport()
    for i in range(4):
        p,_=repeats.step(i/30,q(old,i/30),bbox,[0.,0.],base);assert not p['qualified']
    none,_=support.step(.2,q(now,.2),bbox,[0.,0.],base);assert not none['qualified']
    bg=TemporalPeriodSupport();ring=[(80+i,120) for i in range(24)]
    bg.step(0.,q(old+ring,0.),bbox,[0.,0.],base)
    p,d=bg.step(.033333,q(now+ring,.033333),bbox,[0.,0.],base);assert not p['qualified']
    strong=dict(base,qualified=True,frequency_hz=210.)
    p,_=support.step(.233333,q(now,.233333),bbox,[0.,0.],strong);assert p==strong
    result=dict(synthetic_only=True,normal_sign_and_rotor_permutation_invariant=True,
        stationary_image_motion_protects_wobble_despite_noisy_state_velocity=True,
        unknown_force_not_rotated=True,analytic_integral_matches_quadrature=True,
        zero_angular_rate_and_no_force_preserve_original=True,
        current_frequency_not_old_median=True,distinct_native_pixel_gate_kept16=True,
        repeated_hot_pixels_not_counted_again=True,expired_history_and_common_flicker_rejected=True,
        strong_current_period_preserved=True,periodic_stationary_wobble_ticks_preserved=wobble_ticks)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
