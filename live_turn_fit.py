"""Optional direct coordinated-turn fit of past body centers.

Standard coordinated-turn kinematics, not an event/RPM innovation. Variable
projection solves position/velocity linearly for each angular rate. Existing
current state, eligibility and default rollout remain untouched.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from live_motion import LiveMotion,predictions,cap

DESIGN=dict(history_s=.35,noise_floor_px=6.,minimum_points=6,maximum_rate_rad_s=float(np.pi/(2*.4)),
    minimum_rate_z=1.96,AICc_selection=True,parameters_swept=False,default_promoted=False,
    position_velocity_acceleration_and_validity_unchanged=True,
    reference='https://stonesoup.readthedocs.io/en/v1.6/stonesoup.models.transition.html')

def turn_design(tau,omega):
    theta=omega*tau;s=tau*np.sinc(theta/np.pi)
    c=.5*omega*tau*tau*np.sinc(theta/(2*np.pi))**2
    matrix=np.zeros((2*len(tau),4));matrix[::2,0]=1;matrix[1::2,1]=1
    matrix[::2,2]=s;matrix[::2,3]=-c;matrix[1::2,2]=c;matrix[1::2,3]=s
    return matrix

def fit_turn(history):
    if len(history)<6:return None,dict(reason='few_points')
    tau=np.asarray([t-history[-1][0] for t,p in history]);points=np.asarray([p for t,p in history]).ravel()
    def residual(value):
        matrix=turn_design(tau,float(value[0]));coeff=np.linalg.lstsq(matrix,points,rcond=None)[0]
        return matrix@coeff-points
    bound=DESIGN['maximum_rate_rad_s']
    fit=least_squares(residual,[0.],bounds=(-bound,bound),max_nfev=30)
    rate=float(fit.x[0]);n=len(points);null=residual([0.]);r=fit.fun
    def aicc(k):return 2*k+2*k*(k+1)/(n-k-1)
    gain=float((null@null-r@r)/36.-(aicc(5)-aicc(4)))
    information=float((fit.jac.T@fit.jac).item());se=6./np.sqrt(max(information,1e-12))
    diagnostic=dict(reason='turn_not_supported',rate=rate,rate_standard_error=se,AICc_gain=gain,
        fit_RSS_px2=float(r@r),linear_RSS_px2=float(null@null),nfev=fit.nfev)
    if not fit.success or fit.active_mask[0]!=0 or gain<=0 or abs(rate)<=1.96*se:return None,diagnostic
    diagnostic['reason']='past_coordinated_turn_supported';return rate,diagnostic

class TurnFitMotion(LiveMotion):
    def step(self,t,body):
        c,valid,d=super().step(t,body);rate=None;info=dict(reason='original_acceleration_not_eligible')
        if d['acceleration_eligible']:rate,info=fit_turn(list(self.history))
        if c is not None:c['fitted_turn_rate']=rate
        d['coordinated_turn_fit']=info
        return c,valid,d

def turn_predictions(c,age=0.):
    result,cv=predictions(c,age)
    rate=c.get('fitted_turn_rate') if c is not None else None
    if rate is None:return result,cv
    v=np.asarray(c['velocity']);a=np.asarray(c['acceleration']);v2=float(v@v)
    if v2<30.**2:return result,cv
    candidate=copy.deepcopy(c)
    candidate['acceleration']=cap(v*float(a@v)/v2+rate*np.array([-v[1],v[0]])).tolist()
    alternate,_=predictions(candidate,age)
    for h in [.1,.2,.4]:
        if abs(rate*(age+h))<=np.pi/2:result[str(h)]=alternate[str(h)]
    return result,cv
