"""Projective four-hub geometry. No assumption that the target is stationary."""
import numpy as np

TEMPLATE = np.array([[-.45, -.45], [.45, -.45], [.45, .45], [-.45, .45]])
K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])

def rotation(rx, ry, rz=0.):
    x,y,z=np.deg2rad([rx,ry,rz]);cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]]) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1.,0,0],[0,cx,-sx],[0,sx,cx]])

def homography(R, translation):
    return K @ np.column_stack([R[:,0], R[:,1], translation])

def project(H, points):
    q=np.c_[np.atleast_2d(points),np.ones(len(np.atleast_2d(points)))]@H.T
    return q[:,:2]/q[:,2,None]

def intersection(points):
    """Ordered polygon vertices; opposite pairs are (0,2) and (1,3)."""
    p=np.asarray(points,float);a=p[...,2,:]-p[...,0,:];b=p[...,3,:]-p[...,1,:]
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    denom=cross(a,b);numer=cross(p[...,1,:]-p[...,0,:],b)
    with np.errstate(invalid='ignore',divide='ignore'):
        c=p[...,0,:]+(numer/denom)[...,None]*a
        sine=np.abs(denom)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
    return c,sine

def ellipse_centers(H, radius=.16):
    inv=np.linalg.inv(H);result=[]
    for x,y in TEMPLATE:
        circle=np.array([[1.,0,-x],[0,1.,-y],[-x,-y,x*x+y*y-radius*radius]])
        C=inv.T@circle@inv
        result.append(np.linalg.solve(C[:2,:2],-C[:2,2]))
    return np.asarray(result)

def diagnostics(points, sigmas, rng, repeats=128):
    """Convex ordering is only a geometric hypothesis, not certified rotor identity."""
    import cv2
    p=np.asarray(points,float)
    if p.shape!=(4,2) or not np.isfinite(p).all():return dict(accepted_geometry=False,reason='not_four_finite_current_hubs')
    indices=cv2.convexHull(p.astype(np.float32),returnPoints=False).reshape(-1)
    if len(indices)!=4:return dict(accepted_geometry=False,reason='one_point_inside_or_collinear')
    ordered=p[indices];s=np.asarray(sigmas,float)[indices];c,sine=intersection(ordered)
    if not np.isfinite(c).all() or sine<1e-8:return dict(accepted_geometry=False,reason='collapsed_diagonals')
    perturbed=ordered[None,:,:]+rng.normal(size=(repeats,4,2))*s[None,:,None]
    jitter,_=intersection(perturbed);error=np.linalg.norm(jitter-c,axis=1);finite=np.isfinite(error)
    singular=np.linalg.svd(ordered-ordered.mean(0),compute_uv=False)
    return dict(accepted_geometry=True,center=c.tolist(),centroid=p.mean(0).tolist(),ordered_indices=indices.tolist(),diagonal_sine=float(sine),
        hub_axis_ratio=float(singular[1]/max(singular[0],1e-12)),centroid_difference_px=float(np.linalg.norm(c-p.mean(0))),
        assumed_noise_center_p95_px=float(np.quantile(error[finite],.95)) if finite.any() else None,
        noise_finite_fraction=float(finite.mean()),reason='convex_quad_hypothesis_only')
