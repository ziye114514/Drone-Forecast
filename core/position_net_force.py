"""Fallback image net acceleration from already observed body positions.

One physical rollout is retained. This is NOT a rotor-thrust measurement.
Known input, held force and qualified flow acceleration always take priority.
The existing .35s history,8point/.23s curve gates, Huber loss and3000 cap are reused.
"""
from collections import deque
import numpy as np
from fast_huber import fit
from live_motion import cap
from adaptive_force_memory import kernel


def regression(rows,t,degree):
    stamps=np.array([r[0] for r in rows]);span=float(np.ptp(stamps))
    x=(stamps-t)/span;A=np.column_stack((np.ones(len(x)),x,.5*x*x))[:,:degree+1]
    p=np.array([r[1] for r in rows]);origin=p.mean(0)
    J=np.kron(A,np.eye(2))/6.;target=((p-origin)/6.).ravel()
    initial=np.linalg.lstsq(J,target,rcond=None)[0]
    beta,ok,_=fit(J,target,initial)
    error=J@beta-target
    return beta.reshape(degree+1,2),J,error,origin,span,ok


class PositionNetForce:
    def __init__(self):self.positions=deque();self.last=None;self.epoch=None
    def step(self,row):
        t=row['t'];body=row['body'];info=dict(applied=False,reason='position_net_force_warmup',
            rotor_thrust_measured=False,second_position_filter=False,known_force_path_unchanged=True)
        if self.last is not None and t<=self.last:raise ValueError('Increasing cutoffs required')
        if self.last is not None and t-self.last>.16:self.positions.clear()
        self.last=t
        identity=(body.get('body_track_epoch'),body.get('body_track_id')) if body.get('body_track_id') is not None else None
        if identity is not None and self.epoch is not None and identity!=self.epoch:self.positions.clear()
        if identity is not None:self.epoch=identity
        if body.get('detected') and body.get('confidence',0)>=.25:
            self.positions.append((t,np.asarray(body['center'],float)))
        while self.positions and t-self.positions[0][0]>.35+1e-8:self.positions.popleft()
        force=row['force'];m=row.get('image_motion') or {};a=m.get('acceleration') or {}
        if (not row['valid'] or row['coefficient'] is None or not body.get('detected') or body.get('confidence',0)<.25 or
            force.get('active') or force.get('short_gap_acceleration_held') or a.get('applied') or
            (row.get('adaptive_force_memory') or {}).get('corrections_px')):
            return row['physical'],force,dict(info,reason='existing_force_or_current_flow_preserved')
        rows=list(self.positions)
        if len(rows)<8 or rows[-1][0]-rows[0][0]<.23-1e-8:return row['physical'],force,info
        if max(np.diff([r[0] for r in rows]))>.08+1e-8:return row['physical'],force,dict(info,reason='body_position_gap')
        beta,J,error,origin,span,ok=regression(rows,t,2)
        if not ok or np.mean(abs(error)<=8.07/6.)<.9 or np.median(abs(error))>1.:
            return row['physical'],force,dict(info,reason='body_position_curve_not_supported')
        losses=[]
        for degree in (1,2):
            b,_,_,o,s,success=regression(rows[:-2],t,degree)
            if not success:return row['physical'],force,dict(info,reason='past_curve_fit_failed')
            x=(np.array([r[0] for r in rows[-2:]])-t)/s
            design=np.column_stack((np.ones(2),x,.5*x*x))[:,:degree+1]
            residual=design@b+o-np.array([r[1] for r in rows[-2:]])
            losses.append(float(np.sum((residual/6.)**2)))
        if losses[1]>=losses[0]:return row['physical'],force,dict(info,reason='curve_does_not_predict_last_two_observations_better')
        # Positions can share a localization gauge. Do not obtain an artificial
        # 1/N precision gain from a high source rate.
        C=np.linalg.pinv(J.T@J)*len(rows);se=np.sqrt(np.maximum(np.diag(C)[4:],0))/span**2
        acceleration=beta[2]/span**2
        shrink=np.maximum(0.,1.-se*se/(acceleration*acceleration+1e-12));value=cap(acceleration*shrink)
        if np.linalg.norm(value)<1e-8:return row['physical'],force,dict(info,reason='net_acceleration_below_correlated_position_uncertainty',standard_error_proxy=se.tolist())
        c=row['coefficient'];p=np.asarray(c['position']);v=np.asarray(c['velocity']);tau=float(force['acceleration_correlation_s'])
        predictions={str(h):(p+h*v+kernel(-1/tau,h)*value).tolist() for h in (.1,.2,.4)}
        revised=dict(force,acceleration_px_s2=value.tolist(),acceleration_source='qualified_body_position_net_acceleration')
        return predictions,revised,dict(info,applied=True,reason='qualified_body_position_net_acceleration',
            rows=len(rows),span_s=span,acceleration_px_s2=value.tolist(),standard_error_proxy=se.tolist(),
            held_out_linear_loss=losses[0],held_out_curve_loss=losses[1],future_force_command_known=False)
