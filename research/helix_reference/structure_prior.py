"""Weak-perspective four-rotor prior; an inferred ellipse is never a detection."""
import numpy as np


def image_ellipse(H,r,velocity,age):
    a=np.deg2rad(r['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    M=R@np.diag([r['major']/2,r['minor']/2])/80
    W=np.eye(3);W[:2,:2]=M
    W[:2,2]=np.array([r['cx'],r['cy']])+velocity*age-M@np.array([128.,128.])
    K=np.linalg.inv(W@np.asarray(H));C=K.T@np.diag([1.,1.,-1.])@K
    A=C[:2,:2];center=-np.linalg.solve(A,C[:2,2]);rhs=center@A@center-C[2,2]
    shape=rhs*np.linalg.inv(A)
    eig=np.linalg.eigvalsh(shape)
    if eig.min()<=0 or not np.isfinite(shape).all():return None
    return center,shape


def missing_ellipse(observed,missing,opposite):
    """3 current model conics -> missing parallelogram vertex and size prior."""
    if set(observed)!=(set(range(4))-{missing}):return None
    adjacent=[i for i in observed if i!=opposite[missing]]
    center=observed[adjacent[0]][0]+observed[adjacent[1]][0]-observed[opposite[missing]][0]
    shapes=[s for _,s in observed.values()]
    distances=[sum(np.linalg.norm(a-b) for b in shapes) for a in shapes]
    shape=shapes[int(np.argmin(distances))]
    eig,U=np.linalg.eigh(shape);axes=2*np.sqrt(eig)
    margin=np.sqrt(np.diag(shape))
    if (center-margin<0).any() or (center+margin>np.array([1280,720])).any():return None
    return dict(cx=float(center[0]),cy=float(center[1]),major=float(axes[1]),minor=float(axes[0]),
                angle=float(np.rad2deg(np.arctan2(U[1,1],U[0,1]))),
                inferred_only=True,source='three CURRENT phase-supported model conics; weak-perspective parallelogram prior')
