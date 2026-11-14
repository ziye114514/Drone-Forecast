"""Causal tilt-response calibration from unequal current rotor contributions.

theta = offset + rate*t + unknown_torque*t^2/2 + gain*integral((t-s)*drive(s))ds.
Each observed source set has its own offset, rate and unknown constant torque.
Eliminating these nuisance terms prevents hidden balanced torque or a constant
frequency difference from manufacturing identification. Only matured plane
observations enter the fit; trajectory labels never enter it. Weak perspective,
locally isotropic in-plane inertia and a constant unknown torque per block are
model assumptions, not known drone parameters.
"""
import copy
import numpy as np
from axis_amplitude_bridge import current_axis_qualified


def integrated_drive(t,u):
    first=np.zeros(len(t));second=np.zeros(len(t))
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        second[i]=second[i-1]+first[i-1]*dt+(2*u[i-1]+u[i])*dt*dt/6
        first[i]=first[i-1]+(u[i-1]+u[i])*dt/2
    return first,second


def design(block):
    a=np.asarray(block,float);t=a[:,0]-a[0,0];span=t[-1]
    first,second=integrated_drive(t,a[:,2])
    B=np.c_[np.ones(len(t)),t/span,(t/span)**2];weight=1/a[:,3]
    A=B*weight[:,None];x=second*weight;y=a[:,1]*weight
    inverse=np.linalg.pinv(A);q=x-A@(inverse@x);z=y-A@(inverse@y)
    info=float(q@q)
    # Roundoff projection of constant drive is not physical excitation.
    if np.linalg.norm(q)<=1e-10*max(1.,np.linalg.norm(x)):info=0.
    return dict(information=info,rhs=float(q@z) if info else 0.,A=A,x=x,y=y,
        inverse=inverse,span=span,current_drive=float(a[-1,2]),first_integral=float(first[-1]))


def fit(blocks):
    fits=[design(b) for b in blocks if len(b)>=5 and b[-1][0]-b[0][0]>=.1-1e-8]
    information=sum(b['information'] for b in fits)
    if information<=1e-20:
        return dict(qualified=False,reason='no_drive_excitation_separable_from_unknown_torque',
            data_only_gain_information=information,usable_blocks=len(fits))
    gamma=sum(b['rhs'] for b in fits)/information;sigma=1/np.sqrt(information)
    return dict(qualified=bool(gamma>2*sigma),reason='positive_gain_identified' if gamma>2*sigma else 'gain_not_resolved_above_plane_noise',
        effective_azimuth_response_gain=float(gamma),gain_sigma_proxy=float(sigma),
        gain_zscore=float(gamma/sigma),data_only_gain_information=float(information),usable_blocks=len(fits))


class RotorTorqueResponse:
    def __init__(self):self.blocks=[];self.last_t=None;self.active_ids=None

    def observe(self,t,structure,moment):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing cutoffs required')
        previous=self.last_t;self.last_t=t
        n=np.asarray(structure.get('axis_xy') if structure.get('axis_xy') is not None else [],float)
        info=dict(qualified=False,reason='no_current_independent_spatial_contrast',
            hidden_rotor_torque_filled=False,unknown_torque_eliminated_before_gain_learning=True,
            physical_motor_names_required=False,metric_inertia_or_thrust_identified=False)
        if not moment.get('qualified') or not current_axis_qualified(structure) or n.shape!=(2,) or np.linalg.norm(n)<1e-8:
            self.active_ids=None;return info
        ids=tuple(moment['source_ids']);angle=float(np.arctan2(n[1],n[0]));amplitude=float(np.linalg.norm(n))
        tangent=np.array([-n[1],n[0]])/amplitude
        # Pixel lever arms project the in-plane torque. Normal amplitude
        # converts its tangential projected acceleration to azimuth acceleration.
        drive=float(tangent@moment['normal_acceleration_contrast_xy']/amplitude)
        fresh=(ids!=self.active_ids or previous is None or t-previous>.040001)
        if fresh:self.blocks.append([])
        elif self.blocks[-1]:
            old=self.blocks[-1][-1][1];angle=old+.5*float(np.angle(np.exp(2j*(angle-old))))
            if abs(angle-old)>np.deg2rad(30):self.blocks.append([])
        self.active_ids=ids
        # Same conservative five-degree floor as the disk trend, no division
        # by the number of native events to invent subpixel pose accuracy.
        sigma=np.deg2rad(max(5.,float(structure.get('metric_axis_sigma_proxy_deg') or 5.)))
        self.blocks[-1].append((t,angle,drive,sigma))
        self.blocks=[b for b in self.blocks if b and t-b[-1][0]<=3.]
        calibration=fit(self.blocks);info.update(calibration,current_block_frames=len(self.blocks[-1]),
            current_block_span_s=t-self.blocks[-1][0][0],current_drive=drive,
            qualified=False,calibration_qualified=calibration['qualified'])
        if not calibration['qualified']:return info
        block=self.blocks[-1]
        if len(block)<5 or t-block[0][0]<.1-1e-8:return dict(info,reason='current_unknown_torque_not_yet_resolved')
        d=design(block);gamma=calibration['effective_azimuth_response_gain'];inverse=d['inverse']
        beta=inverse@(d['y']-gamma*d['x']);L=np.array([0.,0.,2/d['span']**2])
        sensitivity=d['current_drive']-L@(inverse@d['x'])
        alpha=float(L@beta+gamma*d['current_drive'])
        variance=float(L@inverse@inverse.T@L+sensitivity**2*calibration['gain_sigma_proxy']**2)
        se=np.sqrt(max(0.,variance))
        good=abs(alpha)>2*se
        return dict(info,qualified=bool(good),reason='current_torque_acceleration_resolved' if good else 'current_angular_acceleration_below_noise',
            angular_acceleration_rad_s2=alpha,angular_acceleration_sigma_proxy=float(se),
            unknown_angular_acceleration_bias=float(L@beta),
            uncertainty_calibrated=False,current_force_contrast_is_not_complete_torque=True)


def adapt(row,info):
    """Only future plane evolution changes; p/v/a and unknown external force stay."""
    result=copy.deepcopy(row);result['rotor_torque_response']=info
    if not info.get('qualified'):return result
    def apply(r):
        normal=r.get('normal_turn_forecast') or {}
        if normal.get('applied'):
            trend=normal.setdefault('trend',{});omega=float(trend.get('shrunk_angular_rate_rad_s',0.))
            alpha=info['angular_acceleration_rad_s2']
            # Retain the existing 30-degree /0.4s normal sweep bound.
            bound=np.deg2rad(30);h=.4
            alpha=float(np.clip(alpha,2*(-bound-omega*h)/h**2,2*(bound-omega*h)/h**2))
            trend.update(angular_acceleration_rad_s2=alpha,angular_acceleration_source='causal_current_rotor_spatial_contrast')
        for part in r.get('force',{}).get('response_model_mixture',[]):apply(part)
    apply(result)
    return result
