"""Research-only forward line-projection periodicity, without inverse conics.

For a projected rotating blade, even spatial moments change with phase.
Fit their harmonic clock in the early half; freeze geometry/clock/harmonic coefficients
before the late half. This is a diagnostic, NOT a motor-RPM/force admission.
Frequency-dependent illumination with a stationary spatial distribution is
an explicit negative. Late R² is descriptive, not a calibrated noise p-value.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from predictive_harmonics import BASE_FREQUENCIES


def clock_design(t,p,frequency=None):
    u=t/.0166665
    B=np.c_[np.ones(len(t)),u,u*u,p]
    if frequency is None:return B
    angle=2*np.pi*frequency*t
    return np.c_[B,np.cos(angle),np.sin(angle),np.cos(2*angle),np.sin(2*angle)]


def fit_clock(t,p,y,train):
    # All choices, including frequency refinement, use discovery events only.
    tt=t[train];pp=p[train];yy=y[train]
    B=clock_design(tt,pp);base=np.linalg.lstsq(B,yy,rcond=None)[0]
    base_residual=yy-B@base;scale=np.maximum(np.mean(base_residual**2,axis=0),1e-6)
    def fit(f):
        X=clock_design(tt,pp,f);coef=np.linalg.lstsq(X,yy,rcond=None)[0];res=yy-X@coef
        return float(np.sum(res*res/scale)),coef
    errors=[fit(f)[0] for f in BASE_FREQUENCIES];index=int(np.argmin(errors));seed=float(BASE_FREQUENCIES[index])
    low=max(float(BASE_FREQUENCIES[0]),seed-15);high=min(float(BASE_FREQUENCIES[-1]),seed+15)
    result=minimize_scalar(lambda f:fit(f)[0],bounds=(low,high),method='bounded',options=dict(xatol=.01));f=float(result.x);loss,coef=fit(f)
    prediction=clock_design(t,p,f)@coef;null=clock_design(t,p)@base
    def r2(keep):
        # A slow nuisance trend is NOT required to extrapolate from early to
        # late. Eliminate it separately in each scored half, from BOTH the
        # observation and the FROZEN harmonic prediction. No late phase,
        # frequency, harmonic amplitude or geometry is fitted here.
        B=clock_design(t[keep],p[keep])
        observed=y[keep]-B@np.linalg.lstsq(B,y[keep],rcond=None)[0]
        clock=prediction[keep]-B@np.linalg.lstsq(B,prediction[keep],rcond=None)[0]
        den=float(np.sum(observed**2/scale));num=float(np.sum((observed-clock)**2/scale))
        return 1-num/max(den,1e-12)
    return dict(frequency_proxy_hz=f,early_seed_hz=seed,train_R2=r2(train),late_R2=r2(~train),
        discovery_events=int(train.sum()),late_events=int((~train).sum()),
        blade_count_or_shaft_RPM_known=False,force_input_enabled=False,
        geometry_and_frequency_frozen_before_late=True),prediction,null


def measure_xy(xy,time,polarity,early_end=.0166665):
    xy=np.asarray(xy,float);time=np.asarray(time,float);p=2*np.asarray(polarity,float)-1
    train=time<early_end
    if min(int(train.sum()),int((~train).sum()))<128:return dict(qualified=False,reason='few_spatial_clock_events'),None
    center=xy[train].mean(0);z=xy[train]-center;C=z.T@z/len(z);val,U=np.linalg.eigh(C)
    ratio=float(np.sqrt(max(val[0],0)/max(val[1],1e-12)))
    if ratio>=.38:return dict(qualified=False,reason='early_support_not_side_line',early_axis_ratio=ratio),None
    direction=U[:,-1]
    if direction[np.argmax(abs(direction))]<0:direction=-direction
    u=(xy-center)@direction;r=max(float(np.quantile(abs(u[train]),.985)),1.)
    u=u/r;inside=abs(u)<=1.2
    xy=xy[inside];time=time[inside];p=p[inside];u=u[inside];train=time<early_end
    if min(int(train.sum()),int((~train).sum()))<128:return dict(qualified=False,reason='few_frozen_line_events'),None
    y=np.c_[u*u,u**4]
    info,prediction,null=fit_clock(time,p,y,train)
    info.update(qualified=True,early_center_xy=center.tolist(),early_axis_xy=direction.tolist(),early_radius_proxy_px=r,
        early_axis_ratio=ratio,late_R2_is_not_a_detection_or_GT_accuracy=True)
    trace=dict(t=time.tolist(),spatial_second_moment=y[:,0].tolist(),frozen_second_moment_prediction=prediction[:,0].tolist(),early_end=early_end)
    return info,trace


def measure_cells(xy,time,polarity,early_end=.0166665):
    """A diagnostic partition, NOT two measured motors or two thrust inputs.

    Opposite rotor phases can cancel a pooled second moment. Keep both fixed
    early-geometry halves and report EVERY result, never select by late R².
    """
    info,trace=measure_xy(xy,time,polarity,early_end)
    if not info.get('qualified'):return info,trace
    xy=np.asarray(xy,float);time=np.asarray(time,float);polarity=np.asarray(polarity)
    center=np.asarray(info['early_center_xy']);axis=np.asarray(info['early_axis_xy'])
    coordinate=(xy-center)@axis;cells={}
    for name,keep in (('negative_half',coordinate<0),('positive_half',coordinate>=0)):
        # A fragment of a qualified line need not itself be thin: shortening
        # the long axis changes its PCA aspect. Reuse the GLOBAL observed
        # axis; do not demand another full line/ellipse from each fragment.
        u=coordinate[keep];tt=time[keep];pp=2*polarity[keep].astype(float)-1;early=tt<early_end
        if min(int(early.sum()),int((~early).sum()))<128:
            cells[name]=dict(qualified=False,reason='few_current_events_in_frozen_spatial_cell');continue
        c=float(np.mean(u[early]));radius=max(float(np.quantile(abs(u[early]-c),.985)),1.)
        z=(u-c)/radius;valid=abs(z)<=1.2;z=z[valid];tt=tt[valid];pp=pp[valid];early=tt<early_end
        if min(int(early.sum()),int((~early).sum()))<128:
            cells[name]=dict(qualified=False,reason='few_events_after_discovery_cell_support');continue
        cell,_,_=fit_clock(tt,pp,np.c_[z*z,z**4],early)
        cells[name]=dict(cell,qualified=True,axis_inherited_from_current_early_global_line=True,
            local_shape_is_not_a_new_motor_or_pose_measurement=True)
    return dict(info,spatial_cells=cells,cells_are_not_rotor_count_or_RPM_assignments=True),trace
