"""Measurement reliability and joint circle-normal baselines, not novel claims.

Circle candidates use the frozen inverse cone implementation. Leave-block-out
spread and local sensitivity are diagnostics, not calibrated probability bounds.
"""
from pathlib import Path
import itertools,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.paper_physics import unit,circle_normal_candidates,rotate_normal

def axis_error(a,b):return float(np.rad2deg(np.arccos(np.clip(abs(unit(a)@unit(b)),0,1))))

def fit_ams(points):
    if len(points)<6:return None
    try:
        (cx,cy),(d1,d2),angle=cv2.fitEllipseAMS(np.float32(points).reshape(-1,1,2))
        if d1<d2:d1,d2,angle=d2,d1,angle+90
        p=np.array([cx,cy,d1/2,d2/2,np.deg2rad(angle)%np.pi])
        return p if np.isfinite(p).all() and min(d1,d2)>.2 else None
    except cv2.error:return None

def residual(points,p):
    x,y,a,b,angle=p;c,s=np.cos(angle),np.sin(angle);d=points-[x,y]
    u=d[:,0]*c+d[:,1]*s;v=-d[:,0]*s+d[:,1]*c
    return ((u/a)**2+(v/b)**2-1)/np.maximum(2*np.sqrt((u/a**2)**2+(v/b**2)**2),1e-8)

def candidates(p,K=None):
    if K is not None:result=circle_normal_candidates(p,K)
    else:
        ratio=np.clip(p[3]/p[2],0,1);xy=np.array([-np.sin(p[4]),np.cos(p[4])])*np.sqrt(1-ratio**2)
        result=[np.r_[xy,ratio],np.r_[-xy,ratio]]
    return [unit(n if n[2]>=0 else -n) for n in result]

def audit_ellipse(points,p,K=None):
    points=np.asarray(points,float).reshape(-1,2)
    bad=dict(accepted=False,reason='insufficient_or_line',normal_spread_deg=90.,center_spread_px=500.,quality_multiplier=0.)
    if p is None or len(points)<16 or p[2]<5 or p[3]<1.5 or p[3]/p[2]<.07:return bad
    if len(points)>160:points=points[np.linspace(0,len(points)-1,160).astype(int)]
    try:reference=candidates(p,K)
    except (ValueError,np.linalg.LinAlgError):return dict(bad,reason='invalid_conic')
    r=residual(points,p);sigma=max(.5,1.4826*float(np.median(abs(r-np.median(r)))))
    # Local Sampson-distance sensitivity; inflate to avoid treating adjacent
    # contour pixels as perfectly independent. No confidence-level claim.
    pars=np.r_[p[:2],np.log(p[2:4]),p[4]]
    def decode(q):return np.r_[q[:2],np.exp(q[2:4]),q[4]]
    J=[]
    for i,step in enumerate([.05,.05,.001,.001,.001]):
        a,b=pars.copy(),pars.copy();a[i]+=step;b[i]-=step
        J.append((residual(points,decode(a))-residual(points,decode(b)))/(2*step))
    J=np.column_stack(J);u,s,vh=np.linalg.svd(J,full_matrices=False)
    covariance=(vh.T*np.minimum(1/np.maximum(s,1e-12)**2,1e12))@vh*(sigma**2*4)
    center_spread=2*np.sqrt(max(np.linalg.eigvalsh(covariance[:2,:2])[-1],0))
    normal_jacobian=[]
    for i,step in enumerate([.05,.05,.001,.001,.001]):
        a,b=pars.copy(),pars.copy();a[i]+=step;b[i]-=step
        choices_a=candidates(decode(a),K);choices_b=candidates(decode(b),K)
        aa=max(choices_a,key=lambda n:abs(n@reference[0]));bb=max(choices_b,key=lambda n:abs(n@reference[0]))
        if aa@reference[0]<0:aa=-aa
        if bb@reference[0]<0:bb=-bb
        normal_jacobian.append((aa-bb)/(2*step))
    NJ=np.column_stack(normal_jacobian);ncov=NJ@covariance@NJ.T
    normal_spread=2*np.rad2deg(np.sqrt(max(np.linalg.eigvalsh(ncov)[-1],0)))
    # Contiguous leave-block-out fits expose sensitivity to missing boundary.
    spreads=[];centers=[];full=fit_ams(points)
    fits=[full]
    for block in np.array_split(np.arange(len(points)),8):
        keep=np.ones(len(points),bool);keep[block]=False;fits.append(fit_ams(points[keep]))
    for q in fits:
        if q is None or q[2]>max(640,p[2]*5):continue
        try:other=candidates(q,K)
        except (ValueError,np.linalg.LinAlgError):continue
        spreads.append(max(min(axis_error(n,m) for m in other) for n in reference))
        centers.append(float(np.linalg.norm(q[:2]-p[:2])))
    if len(spreads)<6:return dict(bad,reason='unstable_resamples')
    normal_spread=max(float(normal_spread),float(np.percentile(spreads,90)))
    center_spread=max(float(center_spread),float(np.percentile(centers,90)))
    accepted=normal_spread<=8 and center_spread<=max(2.,p[2]*.2)
    multiplier=float(np.exp(-.5*(normal_spread/8)**2)) if accepted else 0.
    return dict(accepted=bool(accepted),reason='accepted' if accepted else 'sensitivity_gate',
        normal_spread_deg=min(90.,normal_spread),center_spread_px=min(500.,center_spread),quality_multiplier=multiplier,
        normal_sigma_deg=max(1.,min(30.,normal_spread/2)),residual_sigma_px=sigma,
        residual_median_px=float(np.median(abs(r))),jackknife_samples=len(spreads),
        interpretation='Sensitivity diagnostics, not calibrated probability coverage.')

def principal_axis(normals,weights):
    matrix=sum(w*np.outer(n,n) for n,w in zip(normals,weights))
    n=np.linalg.eigh(matrix)[1][:,-1]
    return n if n[2]>=0 else -n

def joint_normal(ellipses,K,diagnostics=None,robust=False,previous=None):
    usable=[]
    for i,p in enumerate(ellipses):
        if p is None:continue
        d=diagnostics[i] if diagnostics is not None else dict(accepted=True,normal_sigma_deg=2.)
        if robust and not d['accepted']:continue
        try:choices=candidates(p,K)
        except (ValueError,np.linalg.LinAlgError):continue
        usable.append((i,choices,1/max(1.,d.get('normal_sigma_deg',2.))**2))
    if not usable:return dict(accepted=False,reason='no_usable_normals',normal=None)
    proposals=[]
    for labels in itertools.product((0,1),repeat=len(usable)):
        normals=np.array([q[1][j] for q,j in zip(usable,labels)]);weights=np.array([q[2] for q in usable])
        subsets=[np.arange(len(usable))]
        if robust and len(usable)==4:subsets.extend(np.array(q) for q in itertools.combinations(range(4),3))
        for subset in subsets:
            n=principal_axis(normals[subset],weights[subset]);errors=np.array([axis_error(n,m) for m in normals])
            keep=errors<=5 if robust else np.ones(len(normals),bool)
            if robust:
                if keep.sum()<3 and not (previous is not None and keep.sum()>=2 and axis_error(n,previous)<8):continue
                n=principal_axis(normals[keep],weights[keep]);errors=np.array([axis_error(n,m) for m in normals])
            objective=float(np.average(np.minimum(errors,10)**2,weights=weights))
            if previous is not None:objective+=.15*min(axis_error(n,previous),30.)**2
            proposals.append(dict(normal=n,objective=objective,inliers=int(keep.sum()),rms_deg=float(np.sqrt(np.average(errors[keep]**2,weights=weights[keep]))),
                                  used_rotor_indexes=[usable[j][0] for j in np.flatnonzero(keep)]))
    if not proposals:return dict(accepted=False,reason='insufficient_consensus',normal=None)
    best=min(proposals,key=lambda r:(-r['inliers'] if robust else 0,r['objective']))
    alternatives=[r for r in proposals if axis_error(r['normal'],best['normal'])>10 and r['inliers']==best['inliers']]
    second=min((r['objective'] for r in alternatives),default=None)
    margin=1. if second is None else float(np.clip((second-best['objective'])/max(second,1.),0,1))
    accepted=not robust or (best['rms_deg']<=5 and (margin>=.25 or previous is not None and axis_error(best['normal'],previous)<8))
    return dict(best,normal=best['normal'].tolist(),accepted=bool(accepted),reason='accepted' if accepted else 'ambiguous_candidates',
        competing_objective=second,margin=margin,calibrated=K is not None,
        sign_interpretation='Plane normal sign canonicalized to camera z>=0; thrust sign still needs orientation prior.')

class ConsensusAxis:
    def __init__(self,cfg):
        self.cfg=cfg;self.n=None;self.omega=np.zeros(3);self.P=np.eye(3)*.2;self.t=None;self.last_measurement=None;self.last_measurement_t=None
        self.quality=0.;self.diagnostics=dict(accepted=False,reason='uninitialized')

    def step(self,t,rotors,differential=0.,enhanced=False):
        dt=0 if self.t is None else t-self.t
        if self.n is not None:
            self.n=rotate_normal(self.n,self.omega,dt);self.omega*=np.exp(-self.cfg.angular_rate_decay*dt);self.P+=np.eye(3)*self.cfg.normal_process_sigma**2*max(dt,0)
        self.t=t;ellipses=[];diagnostics=[];qualities=[]
        for r in rotors:
            if r.get('axis_only',False) or r.get('geometry_quality',0)<=.15:continue
            ellipses.append(np.array([r['cx'],r['cy'],r['major']/2,r['minor']/2,np.deg2rad(r['angle'])]))
            diagnostics.append(r.get('measurement_audit',dict(accepted=True,normal_sigma_deg=3.)))
            qualities.append(r.get('geometry_quality',.5))
        # Known-K branch is a joint candidate solver; no calibration is invented.
        p=joint_normal(ellipses,self.cfg.camera_matrix,diagnostics,True,self.n) if self.cfg.camera_matrix is not None else None
        if p is None:
            selected=[];ids=[];weights=[]
            for i,(e,d) in enumerate(zip(ellipses,diagnostics)):
                if not d['accepted']:continue
                choices=candidates(e,None)
                n=max(choices,key=lambda n:abs(n@self.n) if self.n is not None else n[:2]@np.asarray(self.cfg.image_up))
                if self.n is not None and n@self.n<0:n=-n
                selected.append(n);ids.append(i);weights.append(1/max(1.,d.get('normal_sigma_deg',3.))**2)
            p=dict(accepted=False,reason='no_weak_consensus',normal=None)
            if selected:
                n=principal_axis(selected,weights);errors=np.array([axis_error(n,q) for q in selected]);keep=errors<=5
                if keep.sum()>=3 or self.n is not None and keep.sum()>=2:
                    n=principal_axis(np.asarray(selected)[keep],np.asarray(weights)[keep])
                    p=dict(accepted=True,reason='weak_prior_consensus',normal=n.tolist(),rms_deg=float(np.sqrt(np.mean(errors[keep]**2))),
                        used_rotor_indexes=[ids[j] for j in np.flatnonzero(keep)],perspective_ambiguity_resolved=False)
        p['calibration_supplied']=self.cfg.camera_matrix is not None
        if self.cfg.camera_matrix is None:
            p['interpretation']='Weak-perspective consistency with temporal/image-up prior; true perspective normal ambiguity remains unvalidated.'
            if self.n is None and p.get('normal') is not None:
                n=np.array(p['normal'])
                if n[:2]@np.asarray(self.cfg.image_up)<0:n[:2]*=-1
                p['normal']=n.tolist()
        self.diagnostics=p;self.quality=0.
        if not p['accepted']:return self.n
        measurement=unit(p['normal'])
        if self.n is not None and measurement@self.n<0:measurement=-measurement
        sigma=max(1.,p['rms_deg']+float(np.mean([d.get('normal_sigma_deg',3) for d in diagnostics])))
        self.quality=float(np.mean(qualities))*min(1.,len(p['used_rotor_indexes'])/3)
        if self.n is None:self.n=measurement.copy()
        else:
            R=np.eye(3)*np.deg2rad(sigma)**2;gain=np.linalg.solve(self.P+R,self.P).T
            self.n=unit(self.n+gain@(measurement-self.n));self.P=(np.eye(3)-gain)@self.P
        if self.last_measurement is not None and t-self.last_measurement_t<.12:
            elapsed=max(t-self.last_measurement_t,.001)
            cross=self.last_measurement[0]*measurement[1]-self.last_measurement[1]*measurement[0]
            signed=np.arctan2(cross,self.last_measurement[:2]@measurement[:2]);self.omega[2]=.8*self.omega[2]+.2*np.clip(signed/elapsed,-3,3)
        self.last_measurement=measurement;self.last_measurement_t=t
        return self.n.copy()
