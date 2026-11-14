"""Actual axis-aligned extent of a rotated image ellipse, sensor coordinates."""
import numpy as np

def ellipse_fov_clipped(r,width=1280,height=720):
    values=np.array([r['cx'],r['cy'],r['major'],r['minor'],r['angle']],float)
    if not np.isfinite(values).all() or min(values[2:4])<=0:return True
    c=values[:2];a,b=values[2:4]/2;angle=np.deg2rad(values[4]);co,si=np.cos(angle),np.sin(angle)
    extent=np.sqrt([a*a*co*co+b*b*si*si,a*a*si*si+b*b*co*co])
    return bool(np.any(c-extent<0) or np.any(c+extent>=[width,height]))
