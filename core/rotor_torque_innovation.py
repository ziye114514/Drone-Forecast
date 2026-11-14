"""Matched angular observers isolate per-source thrust-change information.

The positive initial gain is a PRIOR: for an approximate uniform disk,
I/m=L^2/4. With image force gain K and rotor radius R, gamma=4*K*R/L^2;
unknown image scale cancels. Neither this inertia nor K is measured truth.
An otherwise identical zero-input angular observer provides the ablation, so
merely smoothing the plane cannot masquerade as a rotor-frequency benefit.
"""
import copy
from functools import lru_cache
import numpy as np
from rotor_torque_partition import components
from axis_amplitude_bridge import current_axis_qualified
from native_torque_inputs import components as native_components,selected as native_selected

TAU=.35


def coefficients(dt):
    decay=np.exp(-dt/TAU);kv=TAU*(1-decay);kp=TAU*(dt-kv)
    return decay,kv,kp


@lru_cache(maxsize=128)
def dynamics(dt,size):
    decay,kv,kp=coefficients(dt);F=np.eye(size);F[0,1]=dt;F[0,2]=kp;F[1,2]=kv;F[2,2]=decay
    # Singer angular-acceleration process, using the existing 30deg/.4s bound
    # as a broad process prior. The same process is present in BOTH arms.
    sigma=2*np.deg2rad(30)/.4**2
    nodes,weights=np.polynomial.legendre.leggauss(8);Q=np.zeros((size,size))
    for s,w in zip((nodes+1)*dt/2,weights*dt/2):
        d,v,p=coefficients(s);g=np.zeros(size);g[:3]=[p,v,d]
        Q+=w*(2*sigma*sigma/TAU)*np.outer(g,g)
    F.setflags(write=False);Q.setflags(write=False)
    return F,Q


def update(x,P,theta,sigma):
    residual=.5*float(np.angle(np.exp(2j*(theta-x[0]))))
    if abs(residual)>np.deg2rad(30):return None,None
    S=P[0,0]+sigma*sigma;gain=P[:,0]/S;x=x+gain*residual
    A=np.eye(len(x));A[:,0]-=gain
    P=A@P@A.T+sigma*sigma*np.outer(gain,gain)
    return x,(P+P.T)*.5


class RotorTorqueInnovation:
    def __init__(self,*,gain_scale=1.,fixed_gain=False,frequency_changes_only=False,redistribution_only=False):
        self.x=None;self.P=None;self.control=None;self.C=None;self.last_t=None
        self.previous={};self.epoch=None;self.prior=None
        self.gain_scale=float(gain_scale);self.fixed_gain=bool(fixed_gain);self.frequency_changes_only=bool(frequency_changes_only);self.previous_frequency={}
        self.redistribution_only=bool(redistribution_only)

    @staticmethod
    def initial_gain(row,period):
        hubs=np.asarray(row['structure'].get('model_centers') or [],float)
        current=[r for r in (period.get('meta') or {}).get('records',[]) if r.get('current_force_admitted')]
        radii=[r['result']['arms']['joint_pose']['inferred_geometry']['major']/2 for r in current]
        if not radii:radii=[g['major']/2 for _,_,g in native_selected(row,period)]
        if hubs.shape!=(4,2) or not radii:return None
        s=np.linalg.svd(hubs-hubs.mean(0),compute_uv=False);arm=s[0]/np.sqrt(2)
        if arm<2:return None
        K=float(row['force'].get('effective_gain_px_s2',200.));radius=float(np.median(radii))
        value=4*K*radius/arm**2
        return dict(gain=value,force_gain_px_s2=K,rotor_radius_px=radius,arm_projection_scale_px=float(arm),
            inertia_model='uniform disk I/m=L^2/4, broad uncertain prior',
            physical_inertia_or_thrust_measured=False,scale_cancelled_under_weak_perspective=True)

    def observe(self,row,period):
        t=row['t'];st=row['structure']
        joint=bool('records' in (period.get('meta') or {}))
        current,reference=(components if joint else native_components)(row,period)
        if self.frequency_changes_only:
            from rotor_frequency_drive import sources
            frequency=sources(row,period)
        else:frequency={}
        n=np.asarray(st.get('axis_xy') if st.get('axis_xy') is not None else [],float)
        info=dict(applied=False,reason='await_current_plane_and_mechanical_prior',
            hidden_frequencies_measured=0,heading_used=False,current_source_count=len(current),
            matched_zero_rotor_input_control=True,force_gain_and_inertia_are_priors=True,
            only_current_source_changes_are_inputs=True,source_membership_impulse_added=False)
        info['period_source']='joint_late_confirmed_rings' if joint else 'native_slot_pixel_period_proxies'
        axis_good=current_axis_qualified(st) and n.shape==(2,) and np.linalg.norm(n)>1e-8
        reset=self.last_t is None or t-self.last_t>.12 or st.get('geometry_epoch')!=self.epoch
        if not axis_good:
            self.previous={};self.previous_frequency={};return info
        theta=float(np.arctan2(n[1],n[0]));sigma=np.deg2rad(max(5.,float(st.get('metric_axis_sigma_proxy_deg') or 5.)))
        if self.x is None or reset:
            prior=self.initial_gain(row,period)
            if prior is None:self.previous=current;return info
            prior=dict(prior,gain=prior['gain']*self.gain_scale,mechanical_gain_scale_prior=self.gain_scale)
            self.prior=prior;g=prior['gain'];self.x=np.array([theta,0.,0.,g]);self.control=self.x[:3].copy()
            self.P=np.diag([sigma*sigma,(np.deg2rad(30)/.4)**2,(2*np.deg2rad(30)/.4**2)**2,0. if self.fixed_gain else (2*g)**2]);self.C=self.P[:3,:3].copy()
            self.previous=current;self.previous_frequency=frequency;self.last_t=t;self.epoch=st.get('geometry_epoch')
            return dict(info,reason='mechanical_prior_initialized',prior=prior)
        dt=t-self.last_t
        if dt<=0:raise ValueError('Increasing cutoffs required')
        F,Q=dynamics(dt,4);self.x=F@self.x;self.P=F@self.P@F.T+Q
        A,W=dynamics(dt,3);self.control=A@self.control;self.C=A@self.C@A.T+W
        shared=set(current)&set(self.previous) if dt<=.040001 else set()
        raw=sum((current[k]-self.previous[k] for k in shared),np.zeros(2))
        tangent=np.array([-n[1],n[0]])/np.linalg.norm(n)
        drive=float(tangent@raw/np.linalg.norm(n))
        if self.frequency_changes_only:
            from rotor_frequency_drive import change,redistribution
            measure=redistribution if self.redistribution_only else change
            drive,drive_var,shared=measure(frequency,self.previous_frequency,tangent,np.linalg.norm(n)) if dt<=.040001 else (0.,0.,[])
        transform=np.eye(4);transform[2,3]=drive
        self.x=transform@self.x;self.P=transform@self.P@transform.T
        # Four-percent individual force-proxy error is the pre-existing prior,
        # not calibrated RPM accuracy. Geometry/model errors may be larger.
        if not self.frequency_changes_only:drive_var=sum(.04**2*((tangent@current[k])**2+(tangent@self.previous[k])**2) for k in shared)/float(n@n)
        input_variance=(self.x[3]**2+self.P[3,3])*drive_var
        self.P[2,2]+=input_variance;self.C[2,2]+=input_variance
        residual=.5*float(np.angle(np.exp(2j*(theta-self.x[0]))));innovation_var=float(self.P[0,0]+sigma*sigma)
        log_likelihood=-.5*(residual*residual/innovation_var+np.log(2*np.pi*innovation_var))
        null_residual=.5*float(np.angle(np.exp(2j*(theta-self.control[0]))));null_variance=float(self.C[0,0]+sigma*sigma)
        null_likelihood=-.5*(null_residual*null_residual/null_variance+np.log(2*np.pi*null_variance))
        likelihood_ratio=log_likelihood-null_likelihood
        self.x,self.P=update(self.x,self.P,theta,sigma);self.control,self.C=update(self.control,self.C,theta,sigma)
        self.previous=current;self.previous_frequency=frequency;self.last_t=t;self.epoch=st.get('geometry_epoch')
        if self.x is None or self.control is None:
            self.x=None;return dict(info,reason='plane_jump_reinitializes_both_arms')
        gamma=float(self.x[3]);gamma_sigma=float(np.sqrt(self.P[3,3]))
        if gamma<=0:return dict(info,reason='positive_mechanical_gain_inconsistent_with_observed_planes',gain=gamma)
        if max(abs(self.x[1]-self.control[1]),abs(self.x[2]-self.control[2]))<1e-12:
            return dict(info,reason='zero_rotor_increment_relative_to_matched_control',prior=self.prior,
                current_source_changes=sorted(shared),drive_change=drive,past_plane_log_likelihood=log_likelihood,
                past_plane_log_likelihood_ratio_vs_matched_noise_control=likelihood_ratio)
        return dict(info,applied=True,reason='causal_differential_torque_innovation',prior=self.prior,
            current_source_changes=sorted(shared),drive_change=drive,
            delta_rate_rad_s=float(self.x[1]-self.control[1]),
            delta_acceleration_rad_s2=float(self.x[2]-self.control[2]),
            angular_acceleration_correlation_s=TAU,gain=gamma,gain_sigma_proxy=gamma_sigma,
            gain_data_identification_claimed=False,body_reference=reference,
            past_plane_log_likelihood=log_likelihood,frequency_changes_only=self.frequency_changes_only,
            past_plane_log_likelihood_ratio_vs_matched_noise_control=likelihood_ratio,
            input_uncertainty_also_in_zero_input_control=True,
            visible_collective_scaling_removed=self.redistribution_only,
            input_geometry_and_proxy_error_calibration_missing=True,
            gained_angular_rate_is_not_measured_IMU=True)


def adapt(row,info):
    result=copy.deepcopy(row);result['rotor_torque_innovation']=info
    if not info.get('applied'):return result
    applied=0
    def apply(r):
        nonlocal applied
        normal=r.get('normal_turn_forecast') or {}
        if normal.get('applied'):
            trend=normal.setdefault('trend',{});rate=float(trend.get('shrunk_angular_rate_rad_s',0.))
            if trend.get('angular_acceleration_rad_s2'):raise ValueError('Existing angular curvature is not supported by this isolated trial')
            omega=rate+info['delta_rate_rad_s'];alpha=info['delta_acceleration_rad_s2']
            # Conservative normal-sweep bound, inherited from the mainline.
            bound=np.deg2rad(30);sweep=abs(omega)*.4+.5*abs(alpha)*.4**2;scale=min(1.,bound/max(sweep,1e-12))
            trend.update(shrunk_angular_rate_rad_s=omega*scale,angular_acceleration_rad_s2=alpha*scale,
                torque_innovation_correlation_s=TAU,angular_acceleration_source='differential_current_rotor_change_vs_matched_no_rotor_control')
            applied+=1
        for part in r.get('force',{}).get('response_model_mixture',[]):apply(part)
    apply(result);result['rotor_torque_innovation']['normal_components_modified']=applied
    return result


NODES,WEIGHTS=np.polynomial.legendre.leggauss(16)


def rotation_kernel(lam,h,omega,alpha,tau=TAU):
    s=(NODES+1)*h/2;angle=omega*s+alpha*tau*(s-tau*(-np.expm1(-s/tau)))
    return complex(np.sum(WEIGHTS*(h-s)*np.exp(lam*s+1j*angle))*h/2)
