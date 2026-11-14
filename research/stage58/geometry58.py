"""Unchanged author2D center solver plus diagnostic joint coplanar-circle fit."""
from pathlib import Path
import json,sys,time
import cv2,numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'external/circular_center_calibration_168b85c1'
sys.path.insert(0,str(VENDOR/'src'))
from circular_center.center2d import get_ellipse_polynomial_coeff,refine_projected_center,select_projected_center_candidates_by_homography

PARAMS=dict(points=64,max_nfev=80,radius_bounds=[.02,.99],aspect_bounds=[.25,4.],
    author_defaults='search_ratio=.5,directions16,levels(2,.5,.1),nms_scale.08 unchanged; unit marker diameter2 only scales chord-distance loss, no inferred metric radius.',
    joint='General projective H8 + unknown common radius + unknown rectangular frame aspect. Equal coplanar circles, linear Sampson residual on complete fitted conic perimeters; not arbitrary4-point residual.',
    limits='Joint residual is a model-consistency diagnostic, not true motor identity or calibrated confidence. Raw lens distortion is unknown. Author known-K solver is used only for explicitly rectified synthetic data.')

def verify_vendor():
    import hashlib
    p=json.loads((VENDOR/'UPSTREAM_SNAPSHOT.json').read_text(encoding='utf-8-sig'))
    assert p['commit']=='168b85c1cb241937a0c3e563ef382c20ad0be456'
    for r in p['files']:assert hashlib.sha256((VENDOR/r['path']).read_bytes()).hexdigest()==r['sha256']
    return p

def parameters(ellipse):
    (x,y),(a,b),angle=ellipse
    if a<b:a,b=b,a;angle+=90
    return np.array([x,y,a/2,b/2,np.deg2rad(angle)],float)

def from_rotor(r):return ((r['cx'],r['cy']),(r['major'],r['minor']),r['angle'])

def perimeter(ellipse,count=None):
    x,y,a,b,angle=parameters(ellipse);phase=np.linspace(0,2*np.pi,count or PARAMS['points'],endpoint=False)
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return np.c_[a*np.cos(phase),b*np.sin(phase)]@rotation.T+[x,y]

def author_center(ellipse,second,K,contour=None,second_contour=None):
    start=time.perf_counter();ell=parameters(ellipse);poly=get_ellipse_polynomial_coeff(ellipse)
    result=refine_projected_center(ell,poly,K,2.,input_is_rectified=True)
    chosen,alternative=select_projected_center_candidates_by_homography(result.candidates,ell,
        perimeter(ellipse).T if contour is None else contour.T,
        perimeter(second).T if second_contour is None else second_contour.T,
        nominal_radius_ratio=1.,clamp_degenerate=False)
    return dict(center=chosen.tolist(),alternative=alternative.tolist(),candidates=result.candidates.tolist(),scores=result.scores.tolist(),
        ambiguous=result.ambiguous,evaluations=result.evaluations,elapsed_ms=(time.perf_counter()-start)*1000,
        assumptions='Known rectified K, two equal-radius coplanar circles; no GT branch selection')

def joint_fit(ellipses):
    start=time.perf_counter()
    if len(ellipses)!=4:return dict(valid=False,reason='needs_four_full_ellipses')
    centers=np.array([e[0] for e in ellipses],float);order=np.argsort(np.arctan2((centers-centers.mean(0))[:,1],(centers-centers.mean(0))[:,0]))
    if len(cv2.convexHull(centers.astype(np.float32)))!=4:return dict(valid=False,reason='not_convex_four')
    origin=centers.mean(0);scale=float(np.sqrt(np.mean(np.sum((centers-origin)**2,axis=1))))
    if scale<1:return dict(valid=False,reason='collapsed_extent')
    observed=[(perimeter(ellipses[i])-origin)/scale for i in order]
    points=[np.c_[p,np.ones(len(p))] for p in observed]
    template=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    H=cv2.getPerspectiveTransform(template.astype(np.float32),((centers[order]-origin)/scale).astype(np.float32)).astype(float)
    if not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12:return dict(valid=False,reason='singular_initial_homography')
    H/=H[2,2];magnification=float(np.mean(np.linalg.svd(H[:2,:2],compute_uv=False)))
    radius=float(np.clip(np.median([parameters(e)[2] for e in ellipses])/scale/max(magnification,1e-8),.021,.98))
    initial=np.r_[H.reshape(-1)[:8],np.log(radius),0.]
    def unpack(p):return np.r_[p[:8],1.].reshape(3,3),np.exp(p[8]),np.exp(p[9])
    def residual(p):
        H,radius,aspect=unpack(p)
        try:inverse=np.linalg.inv(H)
        except np.linalg.LinAlgError:return np.full(len(points)*len(points[0]),1e6)
        result=[]
        for xy,edge in zip(template,points):
            x,y=xy*np.array([1.,aspect]);circle=np.array([[1.,0,-x],[0,1.,-y],[-x,-y,x*x+y*y-radius*radius]])
            conic=inverse.T@circle@inverse;cp=edge@conic
            values=np.sum(cp*edge,axis=1)/np.maximum(2*np.linalg.norm(cp[:,:2],axis=1),1e-12)
            result.extend(values*scale)
        return np.asarray(result)
    lower=np.r_[np.full(8,-np.inf),np.log(PARAMS['radius_bounds'][0]),np.log(PARAMS['aspect_bounds'][0])]
    upper=np.r_[np.full(8,np.inf),np.log(PARAMS['radius_bounds'][1]),np.log(PARAMS['aspect_bounds'][1])]
    try:fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=PARAMS['max_nfev'],loss='linear',x_scale='jac')
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as error:return dict(valid=False,reason=type(error).__name__+': '+str(error))
    H,radius,aspect=unpack(fit.x);q=np.c_[template*np.array([1.,aspect]),np.ones(4)]@H.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:,2])<1e-9):return dict(valid=False,reason='nonfinite_projected_hubs')
    fitted=q[:,:2]/q[:,2,None]*scale+origin;back=np.empty((4,2));back[order]=fitted
    singular=np.linalg.svd(fit.jac,compute_uv=False)
    return dict(valid=bool(np.isfinite(fit.fun).all()),reason='diagnostic_fit',optimizer_success=bool(fit.success),status=int(fit.status),nfev=int(fit.nfev),
        rms_px=float(np.sqrt(np.mean(fit.fun**2))),max_abs_residual_px=float(np.max(np.abs(fit.fun))),
        projected_hubs=back.tolist(),geometric_center=(H[:2,2]*scale+origin).tolist(),
        radius_in_template_units=float(radius),frame_aspect=float(aspect),jacobian_min_max_ratio=float(singular[-1]/max(singular[0],1e-20)),
        homography_condition=float(np.linalg.cond(H)),same_projective_halfspace=bool(np.all(q[:,2]>0)),elapsed_ms=(time.perf_counter()-start)*1000,
        confidence_calibrated=False,metric_pose_estimated=False)
