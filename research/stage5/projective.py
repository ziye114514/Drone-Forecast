"""Four observed coplanar circles -> image homography, not metric pose.

Unknown normalized rectangle aspect and common rotor radius are learned only
from qualified four-circle views. Intrinsics are not invented. Ellipse fitting
centers are explicitly distinguished from projected physical circle centers.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment

Q=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
PARAMS=dict(init_frames=5,max_rms_px=2.5,max_one_rms_px=4.,min_flatness=.10,
            min_separation_fraction=.35,max_diameter_ratio=3.,
            max_center_correction_fraction=.25,coast_s=.30,
            aspect_bounds=[.45,2.2],radius_bounds=[.12,.98],max_nfev=45)

def project(H,p):
    p=np.c_[np.asarray(p),np.ones(len(p))]@H.T
    return p[:,:2]/p[:,2,None]

def flatness(p):
    s=np.linalg.svd(p-p.mean(0),compute_uv=False)
    return float(s[-1]/max(s[0],1e-12))

def ellipse_points(r,n=24):
    t=np.arange(n)*2*np.pi/n;a=np.deg2rad(r['angle'])
    R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return np.c_[np.cos(t)*r['major']/2,np.sin(t)*r['minor']/2]@R.T+[r['cx'],r['cy']]

def conic_ellipse(C):
    A=C[:2,:2];c=-np.linalg.solve(A,C[:2,2])
    k=-(c@A@c+2*C[:2,2]@c+C[2,2]);v,V=np.linalg.eigh(A/k)
    if np.min(v)<=0:raise ValueError('Not ellipse')
    axes=1/np.sqrt(v);angle=np.degrees(np.arctan2(V[1,0],V[0,0]))%180
    return dict(cx=float(c[0]),cy=float(c[1]),major=float(axes[0]*2),minor=float(axes[1]*2),angle=float(angle),mode='ellipse',ratio=float(axes[1]/axes[0]))

def circle_conic(q,radius):
    x,y=q
    return np.array([[1.,0,-x],[0,1.,-y],[-x,-y,x*x+y*y-radius*radius]])

def ordered_four(rotors,previous=None):
    if len(rotors)!=4 or any(r.get('mode')!='ellipse' for r in rotors):return None,'need_four_ellipses'
    p=np.array([[r['cx'],r['cy']] for r in rotors]);d=np.array([r['major'] for r in rotors])
    if np.min([r['minor'] for r in rotors])<2:return None,'thin_ellipse'
    if d.max()/max(d.min(),1)>PARAMS['max_diameter_ratio']:return None,'unequal_blobs'
    if flatness(p)<PARAMS['min_flatness']:return None,'near_collinear'
    if len(cv2.convexHull(p.astype(np.float32)))!=4:return None,'not_convex_four'
    distances=np.linalg.norm(p[:,None]-p[None],axis=2)+np.eye(4)*1e9
    if distances.min()<np.median(d)*PARAMS['min_separation_fraction']:return None,'merged_points'
    if previous is None:order=np.argsort(np.arctan2(*(p-p.mean(0))[:,::-1].T))
    else:
        ii,order=linear_sum_assignment(np.linalg.norm(np.asarray(previous)[:,None]-p[None],axis=2))
        if np.max(np.linalg.norm(p[order]-previous,axis=1))>max(30,np.linalg.norm(np.ptp(previous,axis=0))*.25):return None,'association_jump'
    return [rotors[int(i)] for i in order],'four'

def fit_joint(rotors, fixed_shape=None, max_nfev=None):
    p=np.array([[r['cx'],r['cy']] for r in rotors]);origin=p.mean(0)
    scale=max(20.,np.linalg.norm(np.ptp(p,axis=0))*.5)
    observed=np.array([np.c_[ (ellipse_points(r)-origin)/scale,np.ones(24)] for r in rotors])
    aspect=1. if fixed_shape is None else fixed_shape[0]
    q=Q*np.array([aspect,1.]);H=cv2.getPerspectiveTransform(q.astype(np.float32),((p-origin)/scale).astype(np.float32));H/=H[2,2]
    radius=np.clip(np.median([r['major'] for r in rotors])/2/scale/max(np.linalg.svd(H[:2,:2],compute_uv=False)),.15,.9) if fixed_shape is None else fixed_shape[1]
    free=fixed_shape is None
    x0=np.r_[H.ravel()[:8],np.log(radius),np.log(aspect)] if free else H.ravel()[:8]
    low=np.r_[[-20]*6,[-.65]*2,np.log(PARAMS['radius_bounds'][0]),np.log(PARAMS['aspect_bounds'][0])] if free else np.r_[[-20]*6,[-.65]*2]
    high=np.r_[[20]*6,[.65]*2,np.log(PARAMS['radius_bounds'][1]),np.log(PARAMS['aspect_bounds'][1])] if free else np.r_[[20]*6,[.65]*2]
    if np.any(x0<=low) or np.any(x0>=high):return dict(accepted=False,reason='invalid_initial_projection')
    def unpack(x):
        H=np.r_[x[:8],1.].reshape(3,3)
        return H,(float(np.exp(x[8])) if free else radius),(float(np.exp(x[9])) if free else aspect)
    def residual(x):
        H,r,a=unpack(x);q=Q*[a,1.]
        try:inv=np.linalg.inv(H)
        except np.linalg.LinAlgError:return np.ones(100)*1e5
        result=[]
        for i in range(4):
            C=inv.T@circle_conic(q[i],r)@inv
            Cp=observed[i]@C
            result.extend(np.sum(Cp*observed[i],axis=1)/np.maximum(2*np.linalg.norm(Cp[:,:2],axis=1),1e-9)*scale)
        # Keep the entire circle ahead of its projective horizon. Four scalar
        # penalties also make residual length independent of success/failure.
        denominator=np.c_[q,np.ones(4)]@H[2]
        minimum=denominator-r*np.linalg.norm(H[2,:2])
        result.extend(np.maximum(0,.15-minimum)*100)
        return np.array(result)
    try:
        opt=least_squares(residual,x0,bounds=(low,high),max_nfev=max_nfev or PARAMS['max_nfev'],
                          loss='soft_l1',f_scale=1.,diff_step=1e-5,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        hn,radius,aspect=unpack(opt.x)
        N=np.array([[scale,0,origin[0]],[0,scale,origin[1]],[0,0,1.]])
        H=N@hn;q=Q*[aspect,1.];motors=project(H,q);errors=residual(opt.x)[:96].reshape(4,24)
        rms=np.sqrt(np.mean(errors**2,axis=1));correction=np.linalg.norm(motors-p,axis=1)
        bounds_ok=not free or (min(aspect/.45,2.2/aspect,radius/.12,.98/radius)>1.01)
        valid=bool(np.sqrt(np.mean(errors**2))<PARAMS['max_rms_px'] and rms.max()<PARAMS['max_one_rms_px']
            and np.all(correction<np.array([r['major'] for r in rotors])*PARAMS['max_center_correction_fraction']) and bounds_ok)
        return dict(accepted=valid,reason='qualified' if valid else 'inconsistent_four_circles',
            H=H.tolist(),motors=motors.tolist(),center=project(H,np.array([[0.,0.]]))[0].tolist(),
            aspect=aspect,radius=radius,rms_px=float(np.sqrt(np.mean(errors**2))),per_circle_rms_px=rms.tolist(),
            center_correction_px=correction.tolist(),nfev=int(opt.nfev),success=bool(opt.success),
            shape_fixed=not free,metric_pose=False)
    except (ValueError,np.linalg.LinAlgError):return dict(accepted=False,reason='optimizer_failure')

class ProjectiveFour:
    def __init__(self,joint=True):
        self.joint=joint;self.H=None;self.shape=None;self.pending=[]
        self.last_seen=None;self.last_t=None;self.velocity=np.zeros(2);self.anchor=np.zeros(2)
        self.latest=None;self.initializations=0

    def step(self,t,groups,original_center=None):
        dt=0 if self.last_t is None else t-self.last_t;self.last_t=t
        if self.last_seen is not None and t-self.last_seen>PARAMS['coast_s']+1e-8:
            self.H=None;self.pending=[];self.velocity[:]=0
        previous=None
        if self.H is not None:
            q=Q*[self.shape[0],1.];previous=project(self.H,q)+self.velocity*dt
        choices=[];rejections=[]
        for name,group in groups.items():
            rs,why=ordered_four(group,previous)
            if rs is None:rejections.append(name+':'+why);continue
            fit=fit_joint(rs,self.shape)
            if fit['accepted']:choices.append((fit['rms_px'],name,rs,fit))
            else:rejections.append(name+':'+fit['reason'])
        if choices:
            _,source,rs,fit=min(choices,key=lambda q:q[0]);self.latest=fit
            if self.H is None:
                p=np.array(fit['motors']);mean=p.mean(0)
                if self.pending and np.linalg.norm(mean-self.pending[-1][0])>50:self.pending=[]
                self.pending.append((mean,fit['aspect'],fit['radius']))
                if len(self.pending)<PARAMS['init_frames']:
                    return dict(t=t,active=False,mode='initializing',streak=len(self.pending),fit=fit,source=source,rejections=rejections)
                # Freeze only normalized shape, not absolute scale, across gaps.
                self.shape=(float(np.median([a[1] for a in self.pending])),float(np.median([a[2] for a in self.pending])))
                refit=fit_joint(rs,self.shape)
                if not refit['accepted']:
                    self.pending=[];self.shape=None
                    return dict(t=t,active=False,mode='shape_freeze_rejected',fit=refit,rejections=rejections)
                fit=refit;self.initializations+=1;self.velocity[:]=0
                initialize=True
            else:initialize=False
            H=np.array(fit['H'])
            if not self.joint:
                p=np.array([[r['cx'],r['cy']] for r in rs]);q=Q*[self.shape[0],1.]
                H=cv2.getPerspectiveTransform(q.astype(np.float32),p.astype(np.float32))
            center=project(H,np.array([[0.,0.]]))[0]
            if not initialize:
                old=project(self.H,np.array([[0.,0.]]))[0]
                self.velocity=.7*self.velocity+.3*np.clip((center-old)/max(dt,.001),-2000,2000)
            if initialize:
                self.anchor=np.zeros(2)
                if original_center is not None:
                    self.anchor=np.clip(project(np.linalg.inv(H),np.array([original_center]))[0],-.6,.6)
            self.H=H;self.last_seen=t;self.latest=fit;mode='initialized' if initialize else 'four_update'
        elif self.H is None:
            self.pending=[]
            return dict(t=t,active=False,mode='uninitialized',rejections=rejections)
        else:
            T=np.eye(3);T[:2,2]=self.velocity*dt;self.H=T@self.H;mode='coast';source='none'
        return dict(t=t,active=True,mode=mode,source=source,center=project(self.H,np.array([self.anchor]))[0].tolist(),
            geometric_center=project(self.H,np.array([[0.,0.]]))[0].tolist(),
            points=project(self.H,Q*[self.shape[0],1.]).tolist(),observed_count=4 if mode!='coast' else 0,
            support_age_s=t-self.last_seen,shape=list(self.shape),fit=self.latest,
            metric_pose=False,initializations=self.initializations,rejections=rejections)

class DensityFour:
    """EVPeriscope-inspired leaky density + ROI k-means, not ROS code port.

    A forced k=4 result is a proposal ONLY. Four-circle consistency is required
    downstream. ROI comes from past/current detections, never human boxes.
    """
    def __init__(self):self.previous=None
    def detect(self,density,rotors,model_points=None):
        if model_points is not None:p=np.array(model_points)
        elif len(rotors)>=3:p=np.array([[r['cx'],r['cy']] for r in rotors])
        else:return [],dict(reason='no_causal_roi')
        majors=[r['major'] for r in rotors]
        pad=max(20.,np.median(majors)*.9) if majors else 30.
        lo=np.maximum(np.floor(p.min(0)-pad).astype(int),0)
        hi=np.minimum(np.ceil(p.max(0)+pad).astype(int),[density.shape[1]-1,density.shape[0]-1])
        crop=density[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        nonzero=crop[crop>0]
        if len(nonzero)<100:return [],dict(reason='sparse_roi')
        mask=np.uint8(crop>float(nonzero.mean()+nonzero.std()))*255
        mask=cv2.erode(mask,np.ones((2,2),np.uint8));mask=cv2.dilate(mask,np.ones((2,2),np.uint8))
        yy,xx=np.nonzero(mask)
        if len(xx)<120:return [],dict(reason='sparse_binary_roi')
        points=np.c_[xx,yy].astype(np.float32)
        if len(points)>6000:points=points[::int(np.ceil(len(points)/6000))]
        # Deterministic RNG. Initialization is constrained by local density only;
        # labels still require downstream observed circle-shape support.
        cv2.setRNGSeed(41)
        _,labels,centers=cv2.kmeans(points,4,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,25,.1),2,cv2.KMEANS_PP_CENTERS)
        rs=[]
        for i in range(4):
            support=points[labels.ravel()==i]
            if len(support)<25:return [],dict(reason='weak_cluster')
            hull=cv2.convexHull(support).reshape(-1,2)
            if len(hull)<6:return [],dict(reason='nonellipse_cluster')
            try:(cx,cy),(d1,d2),angle=cv2.fitEllipseAMS(hull.reshape(-1,1,2))
            except cv2.error:return [],dict(reason='ellipse_fit_failed')
            if d1<d2:d1,d2,angle=d2,d1,angle+90
            if not np.isfinite([cx,cy,d1,d2,angle]).all() or not 12<d1<350 or d2<2:return [],dict(reason='bad_cluster_shape')
            rs.append(dict(cx=float(cx+lo[0]),cy=float(cy+lo[1]),major=float(d1),minor=float(d2),angle=float(angle%180),
                           ratio=float(d2/d1),mode='ellipse',support=len(support)))
        return rs,dict(reason='proposal_only',roi=np.r_[lo,hi].tolist(),binary_points=len(points),forced_k=4)
