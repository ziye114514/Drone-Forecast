"""Geometry-only sensitivity audit, not an inferred camera calibration.

Cone circular-section eigensolution: Chen et al. (2004), as documented by
https://conics.readthedocs.io/en/latest/pose_estimation.html .
No third-party source is copied. K here is an explicit hypothetical camera.
It must not silently become a calibrated runtime default.
"""
from pathlib import Path
import sys,json,itertools,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from five_circle_model import ellipse_tensors


def conic(r):
    S=ellipse_tensors([r])[0];A=np.linalg.inv(S);c=np.array([r['cx'],r['cy']])
    C=np.empty((3,3));C[:2,:2]=A;C[:2,2]=C[2,:2]=-A@c;C[2,2]=c@A@c-1
    return C


def candidates(r,K):
    Q=K.T@conic(r)@K;lam,U=np.linalg.eigh(Q);order=np.argsort(lam)[::-1];lam=lam[order];U=U[:,order]
    l1,l2,l3=lam
    if not l1>=l2>0>l3:raise ValueError('not an elliptic cone')
    h=np.sqrt(max(0.,(l1-l2)/(l1-l3)));g=np.sqrt((l2-l3)/(l1-l3))
    result=[]
    for sign in [-1.,1.]:
        n=U@np.array([sign*h,0.,g]);n*=1 if n[2]>=0 else -1
        result.append(n)
    return result


def projected(n,point,K):
    q=np.linalg.solve(K,np.r_[point,1.]);q=q[:2]/q[2]
    p=np.array([n[0]-q[0]*n[2],n[1]-q[1]*n[2]])
    if p[1]>0:p=-p
    return p


def angle(a,b):
    return float(np.rad2deg(np.arccos(np.clip(abs(a@b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-15),-1.,1.))))


def synthetic_check(K):
    errors=[];rng=np.random.default_rng(270)
    for _ in range(30):
        n=rng.normal(size=3);n[2]=abs(n[2])+.2;n/=np.linalg.norm(n)
        z=np.array([1.,0.,0.]);a=np.cross(n,z);a/=np.linalg.norm(a);b=np.cross(n,a)
        c=np.r_[rng.uniform(-2,2,2),rng.uniform(4,8)]
        H=K@np.column_stack((a,b,c));inv=np.linalg.inv(H);C=inv.T@np.diag([1.,1.,-1.])@inv
        A=C[:2,:2];p=-np.linalg.solve(A,C[:2,2]);scale=p@A@p-C[2,2]
        eig,U=np.linalg.eigh(A/scale);axes=2/np.sqrt(eig)
        r=dict(cx=p[0],cy=p[1],major=axes[0],minor=axes[1],angle=np.rad2deg(np.arctan2(U[1,0],U[0,0])))
        errors.append(min(angle(x,n) for x in candidates(r,K)))
    assert max(errors)<1e-4,max(errors)
    return max(errors)


if __name__=='__main__':
    OUT=ROOT.parent/'output/projection270_20261108';OUT.mkdir(exist_ok=False)
    K=np.array([[1280.,0.,639.5],[0.,1280.,359.5],[0.,0.,1.]])
    result=dict(hypothetical_K=K.tolist(),actual_intrinsics_available=False,
        raw_to_APS_homography_is_not_intrinsics=True,synthetic_best_branch_max_error_deg=synthetic_check(K),
        no_predictor_changed=True,no_focal_sweep_or_trajectory_tuning=True,cases={})
    for name in ['13','15','16']:
        records=[]
        for row in map(json.loads,source(name).read_text('utf8').splitlines()):
            rotors=[r for r in row['geometry'].get('rotors',[]) if r.get('current_observation',True) and not r.get('fov_clipped') and r.get('ratio',0)>.16 and
                (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))][:6]
            if len(rotors)<2:continue
            pairs=[candidates(r,K) for r in rotors];hyp=[]
            for branches in itertools.product([0,1],repeat=len(rotors)):
                ns=np.array([p[b] for p,b in zip(pairs,branches)]);mean=ns.mean(0);mean/=np.linalg.norm(mean)
                scatter=np.mean([angle(x,mean)**2 for x in ns]);hyp.append((scatter,mean))
            hyp.sort(key=lambda x:x[0]);best=hyp[0]
            point=np.asarray(row['body']['center']);n=projected(best[1],point,K)
            old=row['structure'].get('axis_xy');old=None if old is None else np.array(old)
            records.append(dict(t=row['t'],count=len(rotors),best_scatter_rms_deg=float(np.sqrt(best[0])),
                alternative_scatter_rms_deg=float(np.sqrt(hyp[1][0])),
                alternative_projected_axis_difference_deg=angle(n,projected(hyp[1][1],point,K)),
                axis_difference_deg=None if old is None else angle(n,old),
                magnitude_ratio=None if old is None else float(np.linalg.norm(n)/max(np.linalg.norm(old),1e-12)),
                old_normal=None if old is None else old.tolist(),candidate_normal=n.tolist()))
        usable=[r for r in records if r['axis_difference_deg'] is not None]
        result['cases'][name]=dict(current_conic_frames=len(records),
            median_axis_difference_deg=float(np.median([r['axis_difference_deg'] for r in usable])),
            p95_axis_difference_deg=float(np.percentile([r['axis_difference_deg'] for r in usable],95)),
            median_intra_frame_normal_scatter_deg=float(np.median([r['best_scatter_rms_deg'] for r in usable])),
            records=records)
        print(name,{k:v for k,v in result['cases'][name].items() if k!='records'})
    (OUT/'results.json').write_text(json.dumps(result,indent=2),'utf8')
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
