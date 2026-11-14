"""Unified image-center / appearance-bias state using pinned FilterPy math."""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'external/filterpy_linear_upstream'
snapshot=json.loads((VENDOR/'UPSTREAM_SNAPSHOT.json').read_text('utf-8'))
for row in snapshot['files']:
    data=(VENDOR/row['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha1']
# Execute the actual author function ASTs, without rewriting the arithmetic or
# importing unrelated stats/display dependencies from an installed package.
namespace=dict(np=np,dot=np.dot,zeros=np.zeros,eye=np.eye,isscalar=np.isscalar,shape=np.shape,linalg=np.linalg)
for file,names in [('helpers.py',['reshape_z']),('kalman_filter.py',['predict','update'])]:
    tree=ast.parse((VENDOR/file).read_text('utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(VENDOR/file),'exec'),namespace)
predict=namespace['predict'];update=namespace['update']

PARAMS=dict(sigma_acceleration=6400.,sigma_body_px=6.,sigma_geometry_floor_px=6.,
    bias_diffusion_px_sqrt_s=30.,correlation_body_geometry=.5,side_parallel_sigma_factor=5.,
    minimum_view_weight=.05,minimum_confidence=.45,reset_body_gap_s=.5,smooth_tau_s=.1,
    initial_covariance=10000.,bias_initial_variance=36.,
    state='[cx,cy,vx,vy,bx,by]; body center observes c+b, current joint geometric center observes c.',
    gauge='Keep first qualified model-to-body image offset across later geometry expirations and reinitializations for this object. No center/velocity reset on view changes.',
    covariance='Heuristic geometry variance from fitted residual, center-noise proxy, quality and view weight; not calibrated coverage. Body/conic cross-correlation .5 is assumed, not measured.',
    side='Current validated body-line orientation enlarges along-line BODY-box measurement noise5x when current conic geometry missing. No hidden motor or separate independent line-center measurement.',
    unknowns='Effective image state only; no wind/mass/metric pose/RPM identification. Long edge-on intervals can remain ambiguous.')
NAMES=['center_body_kf107','center_smooth_kf107','center_bias_fixed107','center_bias_observable107']


def coefficient(x):return dict(position=x[:2].tolist(),velocity=x[2:4].tolist(),acceleration=[0.,0.])


class SimpleKF:
    def __init__(self):self.x=None;self.P=None;self.t=None;self.seen=None
    def step(self,t,z):
        if self.seen is not None and t-self.seen>PARAMS['reset_body_gap_s']:self.__init__()
        if self.x is None:
            if z is None:return None
            self.x=np.r_[z,[0.,0.]];self.P=np.eye(4)*PARAMS['initial_covariance'];self.t=t;self.seen=t;return coefficient(self.x)
        dt=t-self.t;assert dt>0;self.t=t;F=np.eye(4);F[:2,2:4]=np.eye(2)*dt
        B=np.r_[np.eye(2)*(.5*dt*dt),np.eye(2)*dt]
        self.x,self.P=predict(self.x,self.P,F=F,Q=PARAMS['sigma_acceleration']**2*(B@B.T))
        if z is not None:
            self.seen=t;self.x,self.P=update(self.x,self.P,z,R=np.eye(2)*36.,H=np.c_[np.eye(2),np.zeros((2,2))])
        return coefficient(self.x)


class BiasFilter:
    def __init__(self,adaptive):self.adaptive=adaptive;self.reset()
    def reset(self):self.x=None;self.P=None;self.t=None;self.seen=None;self.anchor=None;self.geometry_seen=None
    def step(self,t,body,geometry,axis):
        if self.seen is not None and t-self.seen>PARAMS['reset_body_gap_s']:self.reset()
        z=np.asarray(body['center'],float) if body.get('detected') else None
        if z is not None:self.seen=t
        new=self.x is None
        if new:
            if z is None:return None,dict(active=False,reason='no_body_initialization')
            self.x=np.r_[z,[0.,0.,0.,0.]];self.P=np.diag([10000.]*4+[0.,0.]);self.t=t
        else:
            dt=t-self.t;assert dt>0;self.t=t;F=np.eye(6);F[:2,2:4]=np.eye(2)*dt
            B=np.r_[np.eye(2)*(.5*dt*dt),np.eye(2)*dt,np.zeros((2,2))]
            Q=PARAMS['sigma_acceleration']**2*(B@B.T)
            if self.anchor is not None:Q[4:,4:]+=np.eye(2)*PARAMS['bias_diffusion_px_sqrt_s']**2*dt
            self.x,self.P=predict(self.x,self.P,F=F,Q=Q)
        current=bool(z is not None and geometry.get('current') and geometry.get('model_geometric_center') is not None)
        if new and not current:
            return coefficient(self.x),dict(active=True,has_geometry_reference=False,current_geometry=False,
                appearance_bias_px=[0.,0.],center_covariance_px2=self.P[:2,:2].tolist(),bias_covariance_px2=self.P[4:,4:].tolist(),
                side_anisotropic_noise=False,measurement_dimensions=0,geometry_sigma_px=None,geometry_age_s=None,
                innovation=None,center_reset_on_view_change=False,uncertainty_calibrated=False)
        if current and self.anchor is None:
            # First lock aligns measurement coordinates once. Later104 model
            # resets do not change this gauge or move the dynamic state.
            self.anchor=z-np.asarray(geometry['model_geometric_center']);self.P[4:,4:]=np.eye(2)*PARAMS['bias_initial_variance']
        Hc=np.c_[np.eye(2),np.zeros((2,4))];Hb=Hc.copy()
        if self.anchor is not None:Hb[:,4:]=np.eye(2)
        Rb=np.eye(2)*PARAMS['sigma_body_px']**2;side=False;geom_sigma=None;dim=0;innovation=None
        if self.adaptive and not current and axis.get('valid') and axis.get('confidence',0)>=PARAMS['minimum_confidence']:
            a=axis['angle_rad'];u=np.array([np.cos(a),np.sin(a)])
            Rb+=36.*(PARAMS['side_parallel_sigma_factor']**2-1.)*np.outer(u,u);side=True
        if z is not None:
            if current:
                m=geometry['measurement'];weight=max(PARAMS['minimum_view_weight'],m.get('view_weight',1.))
                quality=max(PARAMS['minimum_confidence'],m.get('quality',1.))
                variance=36.
                if self.adaptive:
                    variance=36./(weight*quality)**2+m.get('center_noise_proxy_px',0.)**2+m.get('fit',{}).get('rms_px',0.)**2
                geom_sigma=float(np.sqrt(variance));Rg=np.eye(2)*variance
                cross=np.eye(2)*(PARAMS['correlation_body_geometry']*6.*geom_sigma)
                R=np.block([[Rb,cross],[cross,Rg]]);H=np.r_[Hb,Hc]
                observation=np.r_[z,np.asarray(geometry['model_geometric_center'])+self.anchor];self.geometry_seen=t;dim=4
            else:observation=z;H=Hb;R=Rb;dim=2
            innovation=(observation-H@self.x).tolist()
            self.x,self.P=update(self.x,self.P,observation,R=R,H=H)
            self.P=(self.P+self.P.T)/2
        assert np.isfinite(self.x).all() and np.isfinite(self.P).all()
        assert np.linalg.eigvalsh(self.P).min()>-1e-7
        diag=dict(active=True,has_geometry_reference=self.anchor is not None,current_geometry=current,
            appearance_bias_px=self.x[4:].tolist(),center_covariance_px2=self.P[:2,:2].tolist(),
            bias_covariance_px2=self.P[4:,4:].tolist(),side_anisotropic_noise=side,measurement_dimensions=dim,
            geometry_sigma_px=geom_sigma,geometry_age_s=None if self.geometry_seen is None else t-self.geometry_seen,
            innovation=innovation,center_reset_on_view_change=False,uncertainty_calibrated=False)
        return coefficient(self.x),diag


class CenterBank:
    def __init__(self):self.body=SimpleKF();self.smoothed=SimpleKF();self.fixed=BiasFilter(False);self.observable=BiasFilter(True);self.mean=None;self.t=None;self.seen=None
    def step(self,t,body,geometry,axis):
        if self.seen is not None and t-self.seen>.5:self.__init__()
        z=np.asarray(body['center'],float) if body.get('detected') else None
        if z is not None:self.seen=t
        dt=.02 if self.t is None else t-self.t;self.t=t
        if z is not None:
            value=np.asarray(geometry.get('correction_px',[0.,0.]),float)
            if self.mean is None:self.mean=value.copy()
            else:self.mean+=(-np.expm1(-dt/PARAMS['smooth_tau_s']))*(value-self.mean)
        k=self.body.step(t,z);s=self.smoothed.step(t,None if z is None else z+(self.mean if self.mean is not None else 0.))
        f,fd=self.fixed.step(t,body,geometry,axis);a,ad=self.observable.step(t,body,geometry,axis)
        values=[k,s,f,a];coeff={n:c for n,c in zip(NAMES,values) if c is not None}
        return dict(coefficients=coeff,diagnostics=dict(fixed=fd,observable=ad,smoothing_tau_s=.1,
            smoothed_correction=None if self.mean is None else self.mean.tolist()))
