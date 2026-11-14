"""Learn net-force persistence from matured OWN position observations.

One physical rollout is retained. A scalar blends the original Singer force
memory with persistent current net acceleration. It is an effective forecast
response, not a separately identified gravity, wind, drag or true RPM. The
headless known-input direction correction is integrated with the same blend.
"""
from collections import deque
import numpy as np

def kernel(lam,h):
    return .5*h*h if abs(lam)<1e-10 else (np.expm1(lam*h)-lam*h)/(lam*lam)

class AdaptiveForceMemory:
    def __init__(self,extend_calibrated=False,allow_shorter=False):
        self.extend_calibrated=extend_calibrated
        self.allow_shorter=allow_shorter;self.short_weight=0.
        self.pending=deque();self.rows=deque();self.previous=None;self.last_origin=None;self.weight=0.;self.last_t=None
    def step(self,t,coefficient,observed_position,predictions,force,normal=None,valid=True,
             current_total_qualified=False,current_axis_qualified=False,native_net_motion_supported=False):
        updated=False;reference=None
        if self.last_t is not None and t<=self.last_t:raise ValueError('Strictly increasing cutoffs required')
        if self.previous is not None and t-self.previous[0]>.5:self.__init__(self.extend_calibrated,self.allow_shorter)
        self.last_t=t
        if observed_position is not None:
            z=np.asarray(observed_position,float)
            if self.previous is not None:
                prev,old=self.previous
                while self.pending and self.pending[0]['target']<=t+1e-8:
                    r=self.pending.popleft();target=r['target']
                    if not prev-1e-8<=target<=t+1e-8 or t-prev>.08+1e-8:continue
                    q=0. if t==prev else float(np.clip((target-prev)/(t-prev),0.,1.))
                    current=(1-q)*old+q*z
                    record=(t,r['delta'],current-r['base'],r['horizon'])
                    if self.allow_shorter:record=(*record,r['short_delta'])
                    self.rows.append(record)
                    updated=True
            self.previous=(t,z.copy());reference=z
        while self.rows and t-self.rows[0][0]>3.:self.rows.popleft()
        if updated and self.allow_shorter:
            H=np.eye(2)*36.;q=np.zeros(2);old=np.array([self.short_weight,self.weight])
            for stamp,long,error,h,short in self.rows:
                X=np.column_stack((short,long));residual=error-X@old
                robust=min(1.,6./max(float(np.linalg.norm(residual)),1e-9))
                w=np.exp(-(t-stamp)/.35)*robust/3.
                H+=w*X.T@X;q+=w*X.T@error
            self.short_weight,self.weight=simplex_fit(H,q)
        elif updated:
            numerator=0.;denominator=36.
            for stamp,delta,error,h in self.rows:
                w=np.exp(-(t-stamp)/.35)/3.
                # A wrong origin/visibility jump must not receive unlimited
                # squared-loss influence. Huber uses the existing 6px scale.
                residual=error-self.weight*delta
                robust=min(1.,6./max(float(np.linalg.norm(residual)),1e-9))
                numerator+=w*robust*float(delta@error);denominator+=w*robust*float(delta@delta)
            self.weight=float(np.clip(numerator/denominator,0.,1.))
        result={k:list(v) for k,v in predictions.items()};changes={}
        qualified_input=bool(force.get('active') and force.get('input_u') is not None)
        calibrated_gap=bool(self.extend_calibrated and force.get('gain_ever_learned') and
            current_total_qualified and current_axis_qualified)
        eligible=qualified_input or calibrated_gap or bool(self.allow_shorter and (current_total_qualified or native_net_motion_supported))
        if coefficient is not None and predictions and eligible:
            tau=float(force.get('acceleration_correlation_s',.35));a=np.asarray(force['acceleration_px_s2'])
            omega=(normal or {}).get('trend',{}).get('shrunk_angular_rate_rad_s',0.) if (normal or {}).get('applied') else 0.
            known=np.zeros(2)
            if force.get('active') and force.get('input_u') is not None:known=float(force['effective_gain_px_s2'])*np.asarray(force['input_u'])
            zknown=complex(*known)
            origin=bool(valid and reference is not None and (self.last_origin is None or t-self.last_origin>=.199))
            for key,p in predictions.items():
                h=float(key);real=-1/tau
                delta=(kernel(0.,h)-kernel(real,h))*a
                rotation=(kernel(1j*omega,h)-kernel(0.,h))-(kernel(real+1j*omega,h)-kernel(real,h))
                d=rotation*zknown;delta+=np.array([d.real,d.imag])
                short=np.zeros(2)
                if self.allow_shorter:
                    fast=-1/.1
                    short=(kernel(fast,h)-kernel(real,h))*a
                    change=((kernel(fast+1j*omega,h)-kernel(fast,h))-
                        (kernel(real+1j*omega,h)-kernel(real,h)))*zknown
                    short+=np.array([change.real,change.imag])
                correction=self.weight*delta+self.short_weight*short
                result[key]=(np.asarray(p)+correction).tolist();changes[key]=correction.tolist()
                if origin and max(float(delta@delta),float(short@short))>=36.:
                    self.pending.append(dict(target=t+h,base=np.asarray(p),delta=delta.copy(),short_delta=short.copy(),horizon=h))
            if origin:
                self.last_origin=t
                # Horizons from successive origins overlap; maturity order is
                # by target time, not origin insertion order.
                self.pending=deque(sorted(self.pending,key=lambda r:r['target']))
        info=dict(applied=bool((self.weight>0. or self.short_weight>0.) and changes),weight=self.weight,updated=updated,
            short_weight=self.short_weight,shorter_memory_enabled=self.allow_shorter,
            current_physical_input_available=qualified_input,
            calibrated_model_current_total_acceleration=current_total_qualified,
            calibrated_gap_eligible=calibrated_gap,
            current_frequency_invented=False,
            matured_rows=len(self.rows),pending_targets=len(self.pending),corrections_px=changes,
            current_annotations_used=False,future_observations_used_before_maturity=False,
            prior='Singer at weight0; constant net acceleration at weight1',
            gravity_is_not_decaying=True,known_direction_uses_same_memory=True,
            no_second_trajectory_filter=True,metric_force_identified=False)
        if native_net_motion_supported:info['native_net_motion_supported']=True
        return result,info

    def rebase(self,shift):
        """A known reference change is not prediction error or a force step."""
        shift=np.asarray(shift,float)
        if shift.shape!=(2,) or not np.isfinite(shift).all():raise ValueError('Finite2D reference change required')
        if self.previous is not None:self.previous=(self.previous[0],self.previous[1]+shift)
        for r in self.pending:r['base']=r['base']+shift


def simplex_fit(H,q):
    """Exact convex 2D ridge fit over w>=0, sum(w)<=1; no grid search."""
    candidates=[np.zeros(2)]
    unconstrained=np.linalg.solve(H,q)
    if np.all(unconstrained>=0.) and unconstrained.sum()<=1.:candidates.append(unconstrained)
    for i in (0,1):
        w=np.zeros(2);w[i]=np.clip(q[i]/H[i,i],0.,1.);candidates.append(w)
    v=np.array([1.,-1.]);origin=np.array([0.,1.])
    t=float(np.clip((v@q-v@H@origin)/(v@H@v),0.,1.))
    candidates.append(origin+t*v)
    return min(candidates,key=lambda w:float(w@H@w-2*q@w)).tolist()
