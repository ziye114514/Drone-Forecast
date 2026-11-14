"""Local rotating-phase prior from one observed nonflat ring, without four hubs.

Reuse current harmonic frequency grid, the spatial exp(2j*theta) measurement
in SharedPhaseFFT, and its .5 concentration/3:1 direction-power convention.
Select frequency+direction on the earlier half, validate the frozen choice on
the later half. No per-pixel stationarity requirement, new hidden rotor, RPM
truth or independent-noise probability claim. Geometry is shared across halves.
"""
import numpy as np
from predictive_harmonics import BASE_FREQUENCIES


def ring_phase_prior(events,ellipse,cutoff_s):
    begin=round(cutoff_s*1e6)-33333;mid=begin+33333//2;end=round(cutoff_s*1e6)
    e=events[(events['t']>=begin)&(events['t']<end)]
    empty=dict(qualified=False,phase_hz=None,direction=None,mechanical_RPM_measured=False,
        geometry_shared_across_train_validation=True,calibrated_false_alarm_bound=False,
        source='single_current_ring_spatial_phase_not_complete_five_circle',hidden_rotors_added=0)
    if ellipse.get('fov_clipped') or ellipse.get('ratio',ellipse['minor']/ellipse['major'])<=.16:
        return dict(empty,reason='clipped_or_flat_ring')
    angle=np.deg2rad(ellipse['angle']);co,si=np.cos(angle),np.sin(angle)
    d=np.c_[e['x'],e['y']]-[ellipse['cx'],ellipse['cy']]
    x=(d[:,0]*co+d[:,1]*si)/(ellipse['major']/2)
    y=(-d[:,0]*si+d[:,1]*co)/(ellipse['minor']/2)
    r=np.hypot(x,y);select=(r>=.2)&(r<=1.5)
    e=e[select];r=r[select];theta=np.arctan2(y[select],x[select])
    w=np.exp(-.5*((r-1)/.15)**2);train=e['t']<mid;validation=~train
    if min(train.sum(),validation.sum())<32:return dict(empty,reason='sparse_ring_halves')
    frequency=np.r_[BASE_FREQUENCIES,-BASE_FREQUENCIES]
    phase=2*theta
    def projection(mask,candidates=frequency):
        times=(e['t'][mask]-begin)/1e6
        z=np.exp(1j*(phase[mask,None]-2*np.pi*times[:,None]*candidates))
        return np.sum(w[mask,None]*z,axis=0)/max(float(w[mask].sum()),1e-12)
    discovery=projection(train);k=int(np.argmax(abs(discovery)))
    f_grid=float(frequency[k]);f=f_grid
    # Same bounded log-power parabolic refinement as SharedPhaseFFT. A15Hz
    # discovery bin alone can accumulate >1rad of heldout phase error.
    left=np.flatnonzero(frequency==f_grid-15.);right=np.flatnonzero(frequency==f_grid+15.)
    if len(left) and len(right):
        logp=np.log(np.maximum(abs(discovery[[left[0],k,right[0]]])**2,1e-12))
        den=logp[0]-2*logp[1]+logp[2]
        if den < -1e-12:f+=15.*float(np.clip(.5*(logp[0]-logp[2])/den,-.5,.5))
    frozen=np.array([f,-f]);train_projection=projection(train,frozen);heldout=projection(validation,frozen)
    power=abs(heldout[0])**2;opposite=abs(heldout[1])**2
    contrast=float((power-opposite)/max(power+opposite,1e-12))
    phase_agreement=float(np.real(heldout[0]*np.conj(train_projection[0]))/max(abs(train_projection[0]),1e-12))
    current=theta[validation & (w>=.1)]
    sectors=len(np.unique(np.floor((current+np.pi)*12/(2*np.pi)).astype(int)%12))
    qualified=bool(abs(train_projection[0])>=.5 and abs(heldout[0])>=.5 and contrast>=.5 and phase_agreement>=.5 and sectors>=9)
    return dict(empty,qualified=qualified,phase_hz=abs(f) if qualified else None,
        direction=1 if f>0 else -1,candidate_signed_phase_hz=f,
        discovery_grid_frequency_hz=f_grid,refinement_uses_discovery_only=True,
        discovery_concentration=float(abs(train_projection[0])),validation_concentration=float(abs(heldout[0])),
        validation_direction_power_contrast=contrast,phase_agreement=phase_agreement,
        validation_sectors=sectors,train_events=int(train.sum()),validation_events=int(validation.sum()),
        reason='qualified_current_ring_phase_prior' if qualified else 'ring_phase_not_validated',
        frequency_direction_frozen_before_validation=True)
