"""Current correlation peak geometry; an aperture uncertainty proxy.

Keep the old4px² best-direction floor. Relative inverse curvature supplies
anisotropy; a uniform search interval bounds unresolved directions. This is
not calibrated covariance and never divides by the number of event pixels.
"""
import numpy as np


def assess(score,index,search_px,full_neighborhood=False):
    y,x=index;maximum=max(4.,float(search_px)**2/3.)
    if y<=0 or x<=0 or y>=score.shape[0]-1 or x>=score.shape[1]-1:
        return dict(available=False,covariance_px2=(np.eye(2)*maximum).tolist(),reason='peak_neighborhood_unavailable')
    a=np.asarray(score[y-1:y+2,x-1:x+2],float)
    gradient=np.array([(a[1,2]-a[1,0])/2,(a[2,1]-a[0,1])/2])
    hxy=(a[2,2]-a[2,0]-a[0,2]+a[0,0])/4
    H=np.array([[a[1,2]-2*a[1,1]+a[1,0],hxy],[hxy,a[2,1]-2*a[1,1]+a[0,1]]])
    eigen,V=np.linalg.eigh(-H);largest=max(float(eigen[-1]),0.)
    variance=np.full(2,maximum)
    positive=eigen>max(largest*1e-12,1e-15)
    if largest>0:variance[positive]=np.clip(4*largest/eigen[positive],4.,maximum)
    C=(V*variance)@V.T;refinement=None
    if positive.all():
        value=np.linalg.solve(-H,gradient)
        bound=1. if full_neighborhood else .5
        if np.isfinite(value).all() and np.max(abs(value))<=bound:refinement=value.tolist()
    return dict(available=True,covariance_px2=C.tolist(),negative_hessian_eigenvalues=eigen.tolist(),
        joint_refinement_xy=refinement,gradient_xy=gradient.tolist(),
        best_direction_floor_variance_px2=4.,search_uniform_variance_bound_px2=maximum,
        covariance_calibrated=False,pixel_count_information_multiplier=False)
