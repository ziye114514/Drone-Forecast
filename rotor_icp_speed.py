"""Experimental EV-Tach-inspired bidirectional event-cloud registration.

Reference: https://arxiv.org/abs/2209.02205 . Independently written limited
adapter, NOT author code or a faithful reproduction. Ellipse rectification,
time-axis scaling, finite point cap and quality gates are local choices.
It does not replace the live frequency proxy or measure metric thrust.
"""
import time
import numpy as np
from scipy.spatial import cKDTree

PARAMS=dict(slice_ms=2., initial_step_ms=1., points_per_slice=180,
            iterations=30, time_axis_scale=.5, retained_match_fraction=.8,
            max_bidirectional_disagreement_rad=.12, blade_symmetry=2,
            maximum_relative_IQR=.2, minimum_pairs=4)


def _cloud(xy, stamps, begin, length):
    keep=(stamps>=begin)&(stamps<begin+length)
    q=np.column_stack([xy[keep],(stamps[keep]-begin)*1e-3*PARAMS['time_axis_scale']])
    if len(q)>PARAMS['points_per_slice']:
        q=q[np.linspace(0,len(q)-1,PARAMS['points_per_slice']).astype(int)]
    return q


def _align(p,q):
    if min(len(p),len(q))<25:return None
    tree=cKDTree(q);points=p.copy();total=np.eye(3)
    for _ in range(PARAMS['iterations']):
        distance,index=tree.query(points)
        keep=distance<=np.quantile(distance,PARAMS['retained_match_fraction'])
        a,b=points[keep],q[index[keep]];ca=a.mean(0);cb=b.mean(0)
        u,_,vt=np.linalg.svd((a-ca).T@(b-cb))
        rotation=vt.T@u.T
        if np.linalg.det(rotation)<0:vt[-1]*=-1;rotation=vt.T@u.T
        translation=cb-rotation@ca
        points=points@rotation.T+translation;total=rotation@total
        if np.linalg.norm(rotation-np.eye(3))<1e-5:break
    return float(np.arctan2(total[1,0],total[0,0]))


def estimate(events,rotor,velocity=(0.,0.),cutoff_s=None):
    start=time.perf_counter()
    if rotor.get('ratio',rotor['minor']/rotor['major'])<.2:
        return dict(qualified=False,reason='ellipse_rectification_degenerate')
    t=events['t'].astype(float)
    end=float(t.max()+1) if cutoff_s is None else cutoff_s*1e6
    xy=np.column_stack([events['x'],events['y']]).astype(float)
    xy+=(end-t)[:,None]*1e-6*np.asarray(velocity)
    angle=np.deg2rad(rotor.get('angle',0.));c,s=np.cos(angle),np.sin(angle)
    xy=(xy-[rotor['cx'],rotor['cy']])@np.array([[c,-s],[s,c]])
    xy/=np.array([rotor['major'],rotor['minor']])/2
    radius=np.linalg.norm(xy,axis=1);keep=(radius>.15)&(radius<1.05)
    xy,t=xy[keep],t[keep]
    if len(t)<150:return dict(qualified=False,reason='too_few_local_events',events=len(t))
    values=[];disagreements=[]
    for begin in np.arange(max(t.min(),end-30000),end-4000,3000):
        p=_cloud(xy,t,begin,2000);q=_cloud(xy,t,begin+1000,2000)
        a=_align(p,q);b=_align(q,p)
        if a is None or b is None:continue
        disagreement=abs(a+b);disagreements.append(disagreement)
        if disagreement>PARAMS['max_bidirectional_disagreement_rad']:continue
        initial=(a-b)/2
        if abs(initial)<.02 or abs(initial)>=np.pi/2:continue
        # Refine at a larger baseline, respecting the two-blade half-turn.
        step=min(3000.,max(1000.,.75*(np.pi/2)/abs(initial)*1000.))
        p=_cloud(xy,t,begin,2000);q=_cloud(xy,t,begin+step,2000)
        a=_align(p,q);b=_align(q,p)
        if a is None or b is None or abs(a+b)>PARAMS['max_bidirectional_disagreement_rad']:continue
        theta=(a-b)/2
        if abs(theta)>=np.pi/2:continue
        values.append(abs(theta)/(step*1e-6)*60/(2*np.pi))
    median=float(np.median(values)) if values else None
    iqr=float(np.subtract(*np.percentile(values,[75,25]))/max(median,1e-9)) if values else None
    qualified=len(values)>=PARAMS['minimum_pairs'] and iqr<=PARAMS['maximum_relative_IQR']
    return dict(qualified=bool(qualified),reason='registration_consistent' if qualified else 'insufficient_or_inconsistent_registration',
        mechanical_rpm_estimate=median,independent_RPM_truth=False,
        accepted_pairs=len(values),relative_IQR=iqr,pair_RPM=values,
        bidirectional_disagreements_rad=disagreements,local_events=len(t),
        CPU_ms=(time.perf_counter()-start)*1000)
