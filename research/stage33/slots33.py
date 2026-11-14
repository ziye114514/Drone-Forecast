"""Association-only ablation of occupied stale rotor slots; legacy sensor reused."""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.rotor_sensing import RotorSensing

PARAMS=dict(stale_replacement_age_s=.08,maximum_tracks=4,legacy_association_cost_unchanged=True,
    calibration_reference_reset_on_retirement=True,source='Reuse RotorSensing/SciPy Hungarian. Replacement80ms from existing stage9 reacquire delay; not tuned on stage33 results.')


class ReplaceStale(RotorSensing):
    def __init__(self,cfg):super().__init__(cfg);self.retired=[];self.calibration_epoch=0;self.audit={}

    def step(self,observation,history,t):
        current=observation.get('rotors',[]);tracks=[q for q in self.tracks if t-q.seen<=self.cfg.track_expiry_s]
        retired=[];unmatched=0
        if len(tracks)==PARAMS['maximum_tracks'] and current:
            cost=np.array([[np.linalg.norm(q.position(t)-[r['cx'],r['cy']])+.1*abs(q.rotor['major']-r['major']) for r in current] for q in tracks])
            a,b=linear_sum_assignment(cost);accepted=[(i,j) for i,j in zip(a,b) if cost[i,j]<max(20.,tracks[i].rotor['major']*.75)]
            live={i for i,j in accepted};unmatched=len(current)-len(accepted)
            stale=[i for i,q in enumerate(tracks) if i not in live and t-q.seen>=PARAMS['stale_replacement_age_s']-1e-8]
            stale.sort(key=lambda i:tracks[i].seen)
            retired=[tracks[i].id for i in stale[:unmatched]]
            if retired:
                self.tracks=[q for q in self.tracks if q.id not in retired];self.reference_total=None;self.calibration_epoch+=1
                self.retired.extend(dict(t=t,id=i) for i in retired)
        result=super().step(observation,history,t)
        self.audit=dict(retired_ids=retired,unmatched_current_before_replacement=unmatched,calibration_epoch=self.calibration_epoch)
        return result


def geometry_record(sensing):
    keys=['rotor_id','cx','cy','major','minor','angle','observed_now','last_seen_s','image_side']
    return [{k:r[k] for k in keys} for r in sensing['rotors']]
