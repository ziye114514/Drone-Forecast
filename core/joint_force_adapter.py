"""Optional current ring-phase replacement of the existing observed term.

The original filtered native position observations and five-circle mode remain
unchanged. Missing joint observations fall back to the original rotor signal.
No inferred or retained motor is counted as a currently measured force.
"""
import copy
import numpy as np


def adapt(row,period,refine_normal=False):
    u=period.get('u');meta=period.get('meta') or {};structure=copy.deepcopy(row['structure'])
    if refine_normal:
        normals=[]
        for j,o in enumerate(meta.get('records',[])):
            # Direction and amplitude have different observability. A pair of
            # disjoint CURRENT heldout-confirmed conics can constrain the plane
            # before either motor has a continuous frequency history.
            if not o.get('qualified') or j in meta.get('ambiguous_candidates',[]):continue
            g=o['result']['arms']['joint_pose']['inferred_geometry'];angle=np.deg2rad(g['angle'])
            n=np.array([-np.sin(angle),np.cos(angle)])*np.sqrt(max(0.,1-(g['minor']/g['major'])**2))
            if n[1]>0:n=-n
            normals.append(n)
        if len(normals)>=2:
            unit=np.asarray([n/np.linalg.norm(n) for n in normals]);dot=np.clip(unit@unit.T,-1.,1.)
            if np.arccos(dot).max()<=np.deg2rad(15.):
                structure.update(axis_xy=np.mean(normals,axis=0).tolist(),pose_source='current_qualified_visible_conics',
                    normal_refinement_source='two_or_more_disjoint_current_late_confirmed_joint_rings',
                    inferred_motor_count_unchanged=True,hidden_rpm_measured=False)
    if u is None:
        old=row['force'].get('input_u')
        return None if old is None else np.asarray(old),copy.deepcopy(row['period_selection']),structure
    meta=copy.deepcopy(meta);meta.pop('records',None);meta['baseline_rotor_signal_replaced_by_current_joint_observation']=True
    return np.asarray(u),meta,structure
