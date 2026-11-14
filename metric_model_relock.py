"""Retain a drone's intrinsic five-circle metric, reacquire CURRENT pose.

Object shape does not expire with pose. Three current non-collinear centres
can relock a past model only when their observed circle tensors also agree
with the induced affine metric. Slot permutations are anonymous; symmetry
that gives the same plane is not a heading ambiguity. No hidden RPM is made.
"""
from itertools import combinations,permutations
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from five_circle_model import diagonal_center,ellipse_tensors


def sqrt_spd(S,inverse=False):
    eig,U=np.linalg.eigh((S+S.T)*.5)
    if eig.min()<=1e-9:raise ValueError('Degenerate circle metric')
    return (U*(1/np.sqrt(eig) if inverse else np.sqrt(eig)))@U.T


def log_spd(S):
    eig,U=np.linalg.eigh((S+S.T)*.5)
    if eig.min()<=1e-9:raise ValueError('Degenerate metric')
    return (U*np.log(eig))@U.T


def exp_spd(L):
    eig,U=np.linalg.eigh((L+L.T)*.5);return (U*np.exp(eig))@U.T


def plane(S):
    eig,U=np.linalg.eigh(S);ratio=np.sqrt(eig[0]/eig[1]);n=U[:,0]*np.sqrt(max(0.,1-ratio**2))
    if n[1]>0:n=-n
    return n,float(ratio)


class MetricModelRelock:
    def __init__(self):
        self.template=None;self.metric_log=None;self.calibrations=0

    def step(self,t,body,structure,geometry):
        out=copy.deepcopy(structure);info=dict(applied=False,reason='no_current_metric_pose',
            single_target_intrinsic_model_retained=True,pose_is_not_carried_as_measurement=True,
            hidden_motor_frequencies_measured=0,physical_heading_required=False)
        if not body.get('detected') or body.get('center_provisional'):
            return out,dict(info,reason='no_current_native_target_observation')
        rotors=[r for r in geometry.get('rotors',[]) if r.get('current_observation',True) and not r.get('fov_clipped') and
            r.get('ratio',0)>.16 and (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))]
        if self.template is None:
            if structure.get('mode')!='four' or not structure.get('continuous',{}).get('updated') or len(rotors)<4:
                return out,dict(info,reason='need_one_confirmed_four_circle_model')
            hubs=np.asarray(structure.get('model_centers'),float)
            if hubs.shape!=(4,2):return out,info
            points=np.array([[r['cx'],r['cy']] for r in rotors]);cost=np.linalg.norm(hubs[:,None]-points[None],axis=2);ii,jj=linear_sum_assignment(cost)
            if any(cost[i,j]>max(4.,.15*rotors[j]['major']) for i,j in zip(ii,jj)):return out,dict(info,reason='initial_hubs_not_supported_by_current_conics')
            self.origin=diagonal_center(hubs)
            if self.origin is None:return out,info
            self.scale=float(np.sqrt(np.mean(np.sum((hubs-self.origin)**2,axis=1))))
            self.template=(hubs-self.origin)/self.scale;self.design=np.c_[np.ones(4),self.template]
            selected=[rotors[j] for i,j in sorted(zip(ii,jj))]
            self.metric_log=np.mean([log_spd(S) for S in ellipse_tensors(selected)],axis=0);self.calibrations=1
        if len(rotors)<3:return out,dict(info,reason='two_or_line_uses_existing_side_logic')
        # Same existing front's six-candidate bound; never search the image.
        rotors=sorted(rotors,key=lambda r:r.get('geometry_quality',0.),reverse=True)[:6]
        points=np.array([[r['cx'],r['cy']] for r in rotors]);shapes=ellipse_tensors(rotors);metric=exp_spd(self.metric_log);accepted=[]
        for count in ([4,3] if len(rotors)>=4 else [3]):
            for observed in combinations(range(len(rotors)),count):
                y=points[list(observed)]
                for slots in permutations(range(4),count):
                    X=self.design[list(slots)]
                    if np.linalg.matrix_rank(X)<3:continue
                    beta=np.linalg.lstsq(X,y,rcond=None)[0];A=beta[1:].T/self.scale;S=A@metric@A.T
                    try:root=sqrt_spd(S);inv=sqrt_spd(S,True);normal,ratio=plane(S)
                    except (ValueError,np.linalg.LinAlgError):continue
                    if ratio<=.16:continue
                    residual=np.linalg.norm(X@beta-y,axis=1)
                    if any(error>max(4.,.15*rotors[j]['major']) for error,j in zip(residual,observed)):continue
                    score=float(np.mean(residual**2));good=True
                    for j in observed:
                        relative=inv@shapes[j]@inv;scale=np.sqrt(np.linalg.eigvalsh(relative))
                        if scale.min()<.65 or scale.max()>1.5:good=False;break
                        distance=float(np.trace(S+shapes[j]-2*sqrt_spd(root@shapes[j]@root)))
                        score+=max(0.,distance)/count
                    if good:accepted.append(dict(score=score,normal=normal,ratio=ratio,S=S,A=A,beta=beta,observed=observed,slots=slots))
            if accepted:break
        if not accepted:return out,dict(info,reason='current_centres_and_circles_reject_intrinsic_model')
        accepted.sort(key=lambda r:r['score']);best=accepted[0];near=[r for r in accepted if r['score']<=best['score']+4.]
        angles=[np.arccos(np.clip(abs(r['normal']@best['normal'])/max(np.linalg.norm(r['normal'])*np.linalg.norm(best['normal']),1e-12),-1.,1.)) for r in near]
        if max(angles)>np.deg2rad(15.):return out,dict(info,reason='slot_ambiguity_changes_the_plane',near_hypotheses=len(near))
        hubs=self.design@best['beta'];count=len(best['observed']);center=best['beta'][0]
        out.update(axis_xy=best['normal'].tolist(),pose_source='current_metric_template_relock',axis_source='current_rigid_hubs_with_shared_circle_metric',
            mode='four' if count==4 else 'three',model_centers=hubs.tolist(),
            metric_model_current_center=center.tolist(),metric_model_center_is_independent_observation=False,
            metric_model_inferred_slots=sorted(set(range(4))-set(best['slots'])),
            metric_model_current_slots=list(best['slots']),axis_unsigned=True,full_3d_pose=False,
            metric_axis_sigma_proxy_deg=max(5.,float(np.rad2deg(max(angles)))),
            axis_support=dict(valid=False,reason='current_metric_pose_replaces_unresolved_body_axis'))
        info.update(applied=True,reason='current_circle_metric_relock',current_count=count,
            observed_candidate_indices=list(best['observed']),score_px2=best['score'],near_equivalent_hypotheses=len(near),
            normal_xy=best['normal'].tolist(),ratio=best['ratio'],calibration_observations=self.calibrations,
            intrinsic_model_age_is_not_a_pose_measurement=True,uncertainty_is_calibrated=False)
        if count==4:
            invA=np.linalg.inv(best['A']);mean=np.mean([log_spd(invA@shapes[j]@invA.T) for j in best['observed']],axis=0)
            self.metric_log=(self.calibrations*self.metric_log+mean)/(self.calibrations+1);self.calibrations+=1
        return out,info
