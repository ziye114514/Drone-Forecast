"""Causal normalized-shape refinement after diagnosing premature freezing.

Separate follow-up protocol: replay already decoded RAW-only observations;
no human boxes or future keypoints. Does not alter the first frozen experiment.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research/stage5'));sys.path.insert(0,str(ROOT/'src'))
from projective import ProjectiveFour,ordered_four,fit_joint,project,Q
from study import OUT,CASES,save,verify
from event_drone_forecast.motion import KinematicFilter
from event_drone_forecast.config import Config

class TemporalShape(ProjectiveFour):
    def __init__(self):super().__init__(joint=True);self.shape_history=[];self.shape_updates=0
    def step(self,t,groups,original_center=None):
        previous=None
        if self.H is not None:previous=project(self.H,Q*[self.shape[0],1.])+self.velocity*(0. if self.last_t is None else t-self.last_t)
        rs,why=ordered_four(groups['original'],previous)
        free=fit_joint(rs) if rs is not None else None
        if free and free['accepted']:
            self.shape_history.append((t,free['aspect'],free['radius']))
        self.shape_history=[v for v in self.shape_history if t-v[0]<=1.]
        if len(self.shape_history)>=10 and self.shape is not None:
            target=np.median(np.array(self.shape_history)[:,1:],axis=0)
            old=np.array(self.shape);new=old+.15*np.clip(target-old,-.15,.15)
            if self.H is not None:
                # Changing canonical aspect is a coordinate change. Keep all
                # projected motor points and the stored target anchor continuous.
                self.H[:,0]*=old[0]/new[0];self.anchor[0]*=new[0]/old[0]
            self.shape=tuple(new);self.shape_updates+=1
        r=super().step(t,groups,original_center)
        r['shape_calibration']=dict(qualified_past_views=len(self.shape_history),updates=self.shape_updates,
            uses_current_four_only=True,free_fit_rms_px=free.get('rms_px') if free else None)
        return r

def main():
    verify();folder=OUT/'temporal_shape';folder.mkdir(exist_ok=True)
    if (folder/'protocol.json').exists():raise RuntimeError('Follow-up already frozen')
    save(folder/'protocol.json',dict(reason='Initial five-view shape estimate differed from later four-circle fits; investigate premature parameter freezing.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observations_sha256={c:hashlib.sha256((OUT/c/'trace.jsonl').read_bytes()).hexdigest() for c in CASES},
        window_s=1.,minimum_qualified_views=10,blend=.15,max_target_delta=.15,
        scope='Post-hoc designed causal follow-up on seen development clips, not independent test.',
        input='Existing RAW-derived current/past observations only, no evaluation boxes.',
        four_current_points_required=True,no_new_native_raw_decode=True,shape_is_normalized_not_true_metric=True))
    summaries={}
    for case in CASES:
        dest=folder/case;dest.mkdir(exist_ok=True);tr=TemporalShape();kf=KinematicFilter(Config(),2);seen=None;count=0;active=0;updates=0;start=time.perf_counter();modes={}
        with (dest/'trace.jsonl').open('w',encoding='utf-8') as f:
            for line in (OUT/case/'trace.jsonl').open():
                o=json.loads(line)['observation'];t=o['t'];m=tr.step(t,{'original':o['rotors']},o.get('center'))
                z=m.get('center') if m['active'] and m['mode']!='coast' else None
                if z is not None:seen=t
                if seen is not None and t-seen>.5:kf=KinematicFilter(Config(),2);seen=None
                x=kf.update(t,z,confidence=.75);a=x is not None and seen is not None and t-seen<=.3+1e-8
                forecast=dict(active=bool(a),state=x.tolist() if x is not None else None)
                f.write(json.dumps(dict(t=t,model=m,forecast=forecast),allow_nan=False)+'\n')
                count+=1;active+=int(a);updates+=int(m['mode'] in ['initialized','four_update']);modes[m['mode']]=modes.get(m['mode'],0)+1
        summaries[case]=dict(frames=count,active_forecasts=active,qualified_current_four_updates=updates,shape_updates=tr.shape_updates,modes=modes,wall_s=time.perf_counter()-start)
        print(case,summaries[case],flush=True)
    save(folder/'summary.json',summaries)

if __name__=='__main__':main()
