"""Fit current velocity and two-force response jointly from native increments.

Past plane directions enter exact linear-drag integration BEFORE differentiating
noisy motion. Unknown constant external bias is diffuse, not forced to zero.
An unknown collective normal response is identified only by plane excitation.
Disjoint image pairs avoid reusing a common endpoint as independent evidence.
No current/future reference labels, body NN, metric mass or hidden RPM.
"""
import copy
from collections import deque
import numpy as np
from scipy.special import log_ndtr
from axis_amplitude_bridge import current_axis_qualified

BETA=1/.35


def axis(row):
    n=np.asarray(row['structure'].get('axis_xy') or [],float)
    if not current_axis_qualified(row['structure']) or n.shape!=(2,) or np.linalg.norm(n)<1e-6:return None
    if n[1]>0:n=-n
    return n


def frequency_parts(row,n):
    return {str(k):float(n@np.asarray(v)/float(n@n)) for k,v in row.get('period_selection',{}).get('observed_components',{}).items()}


def fit(pairs,current_n,gain,flat=False):
    # theta = [initial vx,initial vy,unknown A,external bx,external by].
    V=np.zeros((2,5));V[:,:2]=np.eye(2);P=np.zeros((2,5));vf=np.zeros(2);pf=np.zeros(2)
    equations=[];values=[];covariances=[];qt=0.;selected=set(range(len(pairs)-1,-1,-2));q0=pairs[0]['q0']
    for i,pair in enumerate(pairs):
        dt=pair['end']-pair['start'];decay=np.exp(-BETA*dt);kv=-np.expm1(-BETA*dt)/BETA;kp=(dt-kv)/BETA
        n=current_n if flat else (pair['n0']+pair['n1'])/2
        G=np.zeros((2,5));G[:,2]=n;G[:,3:]=np.eye(2)
        q=.5*(pair['q0']+pair['q1'])-q0;known=gain*q*n
        displacement=kv*V+kp*G;fixed=kv*vf+kp*known
        if i in selected:
            R=np.asarray(pair['R']);L=np.linalg.cholesky(R)
            equations.append(np.linalg.solve(L,displacement));values.append(np.linalg.solve(L,pair['delta']-fixed));covariances.append(R)
        P+=displacement;pf+=fixed;V=decay*V+kv*G;vf=decay*vf+kv*known;qt=pair['q1']-q0
    X=np.vstack(equations);y=np.concatenate(values)
    # The external bias and initial velocity have NO finite zero-mean prior.
    # Only A retains the existing1000px/s² weak response prior.
    precision=X.T@X;precision[2,2]+=1e-6
    if np.linalg.matrix_rank(precision)<5:return dict(qualified=False,reason='rank_deficient_motion_window')
    C=np.linalg.inv(precision);theta=C@(X.T@y);sigma=np.sqrt(max(C[2,2],1e-12));raw_A=float(theta[2]);z=raw_A/sigma
    mills=np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z));positive=raw_A+sigma*mills
    # A>=0 conditions the SAME joint posterior, preserving its bias covariance.
    theta=theta+C[:,2]/C[2,2]*(positive-raw_A)
    residual=y-X@theta;noise=max(1.,float(residual@residual)/max(len(y)-5,1))
    v=V@theta+vf;amplitude=float(theta[2]+gain*qt);b=theta[3:];a=amplitude*current_n+b-BETA*v
    direction_information=max(0.,1/max(C[2,2],1e-12)-1e-6)
    return dict(qualified=True,velocity=v.tolist(),acceleration=a.tolist(),normal_amplitude_px_s2=amplitude,
        external_constant_bias_px_s2=b.tolist(),raw_signed_amplitude=raw_A,amplitude_sigma_proxy=sigma*np.sqrt(noise),
        normal_amplitude_data_identified=bool(direction_information>=1e-6),direction_information=direction_information,
        selected_nonoverlapping_pairs=len(selected),residual_noise_scale=noise,weak_prior_only_for_A=True,
        external_bias_prior_zero_mean=False,metric_thrust_or_hidden_RPM_measured=False)


class RotorForceWindow:
    def __init__(self,flat=False,parameter_window=False):self.previous=None;self.pairs=deque();self.q=0.;self.flat=flat;self.parameter_window=parameter_window
    def step(self,row):
        r=copy.deepcopy(row);t=r['t'];n=axis(r);old=self.previous;estimate=r['motion'].get('estimate') or {};dt=r['motion'].get('dt_s')
        current=bool(n is not None and r['coefficient'] and r['valid'] and r['motion'].get('current_observation') and dt and 0<dt<=.12 and estimate.get('score',0)>0 and not estimate.get('peak_at_search_boundary') and estimate.get('current_displacement_qualified',True))
        info=dict(qualified=False,reason='await_contiguous_current_native_motion',past_only=True,flat_plane_control=self.flat)
        if current:
            parts=frequency_parts(r,n)
            if old is not None and abs(t-old['t']-dt)<1e-6:
                angle=np.arccos(np.clip(n@old['n']/np.linalg.norm(n)/np.linalg.norm(old['n']),-1.,1.))
                if angle<=np.deg2rad(30):
                    shared=set(parts)&set(old['parts']);dq=sum(parts[k]-old['parts'][k] for k in shared);q0=self.q;self.q+=dq
                    R=np.asarray(estimate.get('displacement_covariance_px2',np.eye(2)*4.),float);R=(R+R.T)/2;ev,U=np.linalg.eigh(R);R=(U*np.maximum(ev,4.))@U.T
                    self.pairs.append(dict(start=old['t'],end=t,n0=old['n'],n1=n,q0=q0,q1=self.q,
                        delta=np.array([estimate['dx'],estimate['dy']]),R=R))
                else:self.pairs.clear();self.q=0.
            else:self.pairs.clear();self.q=0.
            self.previous=dict(t=t,n=n,parts=parts)
        else:self.previous=None;self.pairs.clear();self.q=0.
        span=3. if self.parameter_window else .35
        while self.pairs and t-self.pairs[0]['start']>span+1e-8:self.pairs.popleft()
        if current and len(self.pairs)>=6 and t-self.pairs[0]['start']>=.16-1e-8:
            info=fit(list(self.pairs),n,float(r['force']['effective_gain_px_s2']),self.flat)
            info.update(past_only=True,flat_plane_control=self.flat,span_s=t-self.pairs[0]['start'],
                force_membership_changes_not_input_impulses=True,same_native_evidence_not_an_independent_sensor=True)
            if self.parameter_window and info['qualified']:
                # Estimate slow force parameters with the SAME3s history used
                # by the original gain learner; retain current native velocity.
                # Large model residuals reduce usable parameter information.
                effective=info['direction_information']/info['residual_noise_scale']
                info.update(parameter_window=True,effective_direction_information=effective,
                    normal_amplitude_data_identified=bool(effective>=1e-6))
                velocity=np.asarray(r['coefficient']['velocity'])
                acceleration=info['normal_amplitude_px_s2']*n+np.asarray(info['external_constant_bias_px_s2'])-BETA*velocity
                info.update(velocity=velocity.tolist(),acceleration=acceleration.tolist())
                if not info['normal_amplitude_data_identified'] or np.linalg.norm(acceleration)>3000:
                    info.update(qualified=False,reason='parameters_unresolved_or_current_force_exceeds_existing3000prior')
            if info['qualified']:
                c=r['coefficient'];f=r['force'];c.update(velocity=info['velocity'],acceleration=info['acceleration']);f['acceleration_px_s2']=info['acceleration']
                known=np.zeros(2) if f.get('input_u') is None else float(f['effective_gain_px_s2'])*np.asarray(f['input_u']);total=info['normal_amplitude_px_s2']*n
                f['disturbance_px_s2']=(np.asarray(info['acceleration'])-known).tolist()
                f['unobserved_normal_response_px_s2']=(total-known).tolist();f['external_unknown_not_rotated']=(np.asarray(info['acceleration'])-total).tolist()
                f.pop('response_model_mixture',None);f.pop('anisotropic_disk_response',None);f.pop('external_unknown_rate_px_s3',None)
                f['future_normal_response_model']='drag_filtered_normal_derivative'
                if info['normal_amplitude_data_identified']:f['normal_turn_vector_px_s2']=total.tolist()
                # Otherwise retain old future normal-drive prior, not an
                # arbitrary positive amplitude from an unobservable gauge.
                from physical_publication import evaluate
                r['physical']=evaluate(r,[.1,.2,.4])
        r['rotor_force_window']=info;return r
