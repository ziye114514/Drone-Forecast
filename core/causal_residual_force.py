"""Causal residual net force when no calibrated rotor-force channel exists.

Matured OWN image positions supervise the earlier0.2s rollout. No external
labels, no separate position filter. Correlated residuals use mean scatter,
not variance divided by sample count. Visibility gauge changes invalidate rows.
"""
from collections import deque
import numpy as np
from adaptive_force_memory import kernel


class CausalResidualForce:
    def __init__(self):
        self.pending=deque();self.rows=deque();self.previous=None;self.last_origin=None;self.last_t=None
    def step(self,row):
        t=row['t'];p=row.get('force_observed_position');f=row['force'];m=row.get('image_motion') or {}
        obs=m.get('observation') or {};valid=bool(row['valid'] and p is not None and obs.get('valid'))
        eligible=bool(not f.get('active') and not f.get('short_gap_acceleration_held') and not f.get('gain_ever_learned'))
        reanchor=obs.get('anchor_upgrade') or obs.get('visibility_reanchor') or obs.get('contained_weak_patch_motion')
        if reanchor or (self.last_t is not None and t-self.last_t>.5):self.__init__()
        self.last_t=t;updated=False
        if valid and eligible:
            z=np.asarray(p,float)
            if self.previous is not None:
                prev,old=self.previous
                while self.pending and self.pending[0][0]<=t+1e-8:
                    target,base=self.pending.popleft()
                    if not prev-1e-8<=target<=t+1e-8 or t-prev>.08+1e-8:continue
                    alpha=0. if t==prev else (target-prev)/(t-prev)
                    self.rows.append((t,(1-alpha)*old+alpha*z-base));updated=True
            self.previous=(t,z.copy())
            if row['physical'] and (self.last_origin is None or t-self.last_origin>=.199):
                self.pending.append((t+.2,np.asarray(row['physical']['0.2'],float)));self.last_origin=t
        while self.rows and t-self.rows[0][0]>3.:self.rows.popleft()
        info=dict(applied=False,reason='no_matured_reliable_residual',updated=updated,matured_rows=len(self.rows),
            external_labels_used=False,metric_force_measured=False,residuals_are_correlated=True,
            visibility_gauge_reset=bool(reanchor),new_position_filters=0)
        if not valid or not eligible or not self.rows:return row['physical'],f,info
        age=np.array([t-r[0] for r in self.rows]);w=np.exp(-age/.35);w/=w.sum()
        errors=np.array([r[1] for r in self.rows]);mean=w@errors
        variance=np.sum(w[:,None]*(errors-mean)**2,axis=0)+72.
        delta=mean*np.maximum(0.,1.-variance/(mean*mean+1e-12))
        tau=float(f.get('acceleration_correlation_s',.35));K=kernel(-1/tau,.2)
        bias=delta/K;current=np.asarray(f['acceleration_px_s2']);total=current+bias
        length=np.linalg.norm(total)
        if length>3000.:total*=3000./length
        bias=total-current
        if np.linalg.norm(bias)<1e-8:return row['physical'],f,dict(info,reason='residual_not_above_common_image_noise')
        revised={h:(np.asarray(pred)+kernel(-1/tau,float(h))*bias).tolist() for h,pred in row['physical'].items()}
        force=dict(f,acceleration_px_s2=total.tolist(),acceleration_source='causal_prediction_residual_net_force')
        return revised,force,dict(info,applied=True,reason='matured_image_residual_corrects_net_force',
            correction_px_s2=bias.tolist(),matured_mean_residual_px=mean.tolist(),residual_variance_proxy_px2=variance.tolist())
