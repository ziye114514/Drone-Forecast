"""Exact bounds of a projected circular disk, with quantization padding."""
import numpy as np


def circle_bounds(H, radius=1.5):
    H=np.asarray(H,float);g,h,f=H[2]
    A=f*f-radius*radius*(g*g+h*h)
    if f<=0 or A<=1e-10:return None
    result=[]
    for a,b,c in H[:2]:
        B=2*(radius*radius*(a*g+b*h)-c*f)
        C=c*c-radius*radius*(a*a+b*b)
        disc=B*B-4*A*C
        if disc<0:return None
        delta=np.sqrt(disc)
        result.append(((-B-delta)/(2*A),(-B+delta)/(2*A)))
    return np.array(result)


def prefilter(xy, rotors, r, rotation):
    M=rotation@np.diag([r['major']/2,r['minor']/2])/80
    W=np.eye(3);W[:2,:2]=M;W[:2,2]=np.array([r['cx'],r['cy']])-M@np.array([128.,128.])
    padding=.5*np.abs(M).sum(axis=1)+1e-5
    keep=np.zeros(len(xy),bool)
    for H in rotors:
        bounds=circle_bounds(W@H)
        if bounds is None:return np.ones(len(xy),bool)
        bounds[:,0]-=padding;bounds[:,1]+=padding
        keep|=((xy>=bounds[:,0])&(xy<=bounds[:,1])).all(axis=1)
    return keep
