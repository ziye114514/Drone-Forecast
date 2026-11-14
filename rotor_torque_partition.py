"""Visibility-partition-invariant observed torque CHANGE, not total torque.

New/lost source baselines transfer to the unknown angular response. Only
changes of sources measured in BOTH consecutive packets enter the known input.
No heading, fabricated hidden frequency, or constant zero hidden torque is used.
The instantaneous normal acceleration still comes from the measured plane fit.
"""
import numpy as np
from axis_amplitude_bridge import current_axis_qualified
from rotor_torque_response import RotorTorqueResponse


def components(row,period):
    structure=row['structure'];center=structure.get('current_geometric_center')
    kind='current_five_circle_center_prior_conditioned'
    if center is None:
        center=row['body'].get('center');kind='current_visible_envelope_center_not_COM'
    if center is None or not row['body'].get('detected') or not current_axis_qualified(structure):return {},kind
    center=np.asarray(center,float);out={}
    for record in (period.get('meta') or {}).get('records',[]):
        if not record.get('qualified') or not record.get('current_force_admitted'):continue
        f=record.get('proxy_twofold_hz');g=record.get('result',{}).get('arms',{}).get('joint_pose',{}).get('inferred_geometry') or {}
        if not f or g.get('fov_clipped') or not g.get('major'):continue
        p=np.asarray(g.get('projected_shaft_xy',[g.get('cx'),g.get('cy')]),float)
        if p.shape!=(2,) or not np.isfinite(p).all():continue
        out[str(record['track_id'])]=-(float(f)/300)**2/4*(p-center)/(float(g['major'])/2)
    return out,kind


class RotorTorquePartition:
    def __init__(self):
        self.learner=RotorTorqueResponse();self.previous={};self.q=np.zeros(2)
        self.last_t=None;self.last_epoch=None;self.gauge=0

    def observe(self,row,period):
        t=row['t'];epoch=row['structure'].get('geometry_epoch');current,reference=components(row,period)
        reset=(self.last_t is None or t-self.last_t>.040001 or epoch!=self.last_epoch or not self.previous)
        if reset:self.q[:]=0.;self.gauge+=1
        shared=set(current)&set(self.previous) if not reset else set()
        added=set(current)-set(self.previous);lost=set(self.previous)-set(current)
        delta=sum((current[k]-self.previous[k] for k in shared),np.zeros(2))
        self.q+=delta
        moment=dict(qualified=bool(current),source_ids=[f'partition_gauge:{self.gauge}'],
            normal_acceleration_contrast_xy=self.q.tolist())
        info=self.learner.observe(t,row['structure'],moment)
        info.update(visibility_partition_gauge=True,current_measured_sources=sorted(current),
            current_source_count=len(current),source_changes_credit=sorted(shared),
            newly_visible_baseline_transferred_to_unknown=sorted(added),
            lost_sources_transferred_to_unknown=sorted(lost),
            observed_torque_change_xy=delta.tolist(),known_torque_change_gauge_xy=self.q.tolist(),
            source_membership_impulse_added=False,known_absolute_torque_invented=False,
            current_hidden_frequency_measured=0,body_reference=reference,
            body_reference_is_COM_truth=False,propeller_coefficients_are_equal_priors_not_measurements=True)
        self.previous=current;self.last_t=t;self.last_epoch=epoch
        return info
