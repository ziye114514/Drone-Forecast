"""Weak-perspective image dynamics with relative scale and delayed updates.

Mass, thrust coefficient, metric size and wind are not separately identifiable.
A shared bias-only model is the required control for any proposed physics gain.
"""
from collections import deque
import numpy as np
from residual148 import ResidualExpert

PARAMS=dict(scale_min_current_tracks=3,scale_min_observations=3,
    scale_history_s=.30,scale_min_span_s=.12,max_log_scale_rate=.50,
    model_gain_bounds=[0.,5000.],disturbance_bounds=[-2500.,2500.],
    train_horizon_s=.20,minimum_updates=10,loss_window_s=1.2,
    min_same_origin_pairs=10,event_margin=.05,max_weight=.35,
    max_axis_rate_rad_s=3.,axis_rate_decay_s=.15,
    principal_point='image centre assumption',metric_parameters_identified=False)

class RelativeScale:
    def __init__(self):
        self.reference={}; self.history=deque()
    def step(self,t,geometry):
        candidates=[]
        for r in geometry.get('observed_rotors',[]):
            if not r.get('current_observation') or r.get('association_ambiguous'): continue
            if r.get('temporal_observations',0)<PARAMS['scale_min_observations']: continue
            if r.get('mode')=='line' or r.get('ratio',0)<.16 or r.get('fov_clipped'): continue
            if r.get('inferred_shape') and not (r.get('curvature_validated') and r.get('visible_arc_fraction',0)>=.65): continue
            identifier=r.get('visual_track_id')
            if identifier is None: continue
            candidates.append((identifier,float(r['major'])))
        # References for new tracks are tied to the previous common scale,
        # rather than resetting the whole object's apparent size to one.
        previous=float(np.exp(self.history[-1][1])) if self.history else 1.
        for identifier,size in candidates:
            if identifier not in self.reference: self.reference[identifier]=size/previous
        ratios=[size/self.reference[i] for i,size in candidates]
        valid=len(ratios)>=PARAMS['scale_min_current_tracks']
        if valid:
            logs=np.log(ratios); mid=np.median(logs)
            valid=bool(np.max(abs(logs-mid))<.15)
        if valid:
            value=float(np.median(np.log(ratios)))
            self.history.append((t,value))
        while self.history and t-self.history[0][0]>PARAMS['scale_history_s']+1e-9: self.history.popleft()
        gamma=0.; span=0.
        if valid and len(self.history)>=5:
            ts=np.array([q[0]-t for q in self.history]); ys=np.array([q[1] for q in self.history])
            span=float(np.ptp(ts))
            if span>=PARAMS['scale_min_span_s']:
                slope,offset=np.polyfit(ts,ys,1)
                residual=ys-(slope*ts+offset)
                if np.median(abs(residual))<.025:
                    gamma=float(np.clip(slope,-PARAMS['max_log_scale_rate'],PARAMS['max_log_scale_rate']))
        return dict(valid=valid,relative_scale=float(np.exp(self.history[-1][1])) if valid else 1.,
            log_scale_rate_per_s=gamma,history_span_s=span,current_track_count=len(candidates),
            source='same visual-ID rotor major axes; weak perspective',physical_size_known=False)

class Core:
    def __init__(self):
        self.theta=np.array([200.,0.,200.]); self.P=np.diag([1e5,1e5,1e5])
        self.queue=deque(); self.loss={h:deque() for h in (.1,.2,.4)}; self.updates=0
    def mature(self,t,body):
        while self.queue and self.queue[0]['target_t']<=t+1e-8:
            q=self.queue.popleft()
            if abs(t-q['target_t'])>.011 or not body.get('detected'): continue
            truth=np.asarray(body['center'],float)
            self.loss[q['h']].append((t,float(np.linalg.norm(truth-q['baseline'])),
                float(np.linalg.norm(truth-q['bias'])),float(np.linalg.norm(truth-q['physical']))))
            if abs(q['h']-.2)>1e-8: continue
            A=q['A']; error=np.clip(truth-q['known']-A@self.theta,-60.,60.)
            S=np.eye(2)*16.+A@self.P@A.T
            K=np.linalg.solve(S,A@self.P).T
            self.theta+=K@error
            self.theta[0]=np.clip(self.theta[0],0.,5000.)
            self.theta[1:]=np.clip(self.theta[1:],-2500.,2500.)
            self.P=(self.P-K@A@self.P)/.998
            self.P=(self.P+self.P.T)*.5; self.updates+=1
        for loss in self.loss.values():
            while loss and t-loss[0][0]>PARAMS['loss_window_s']: loss.popleft()

class ProjectedExpert:
    def __init__(self,w=1280,h=720):
        self.scale=RelativeScale(); self.bias=ResidualExpert()
        self.scaled=Core(); self.unscaled=Core(); self.axis_history=deque()
        self.principal=np.array([w/2.,h/2.])
    def step(self,t,body,geometry,proxy,coefficient,baseline):
        bias=self.bias.step(t,body,geometry,dict(valid=False),baseline)
        self.scaled.mature(t,body); self.unscaled.mature(t,body)
        scale=self.scale.step(t,geometry)
        available=bool(coefficient is not None and proxy.get('valid') and geometry.get('axis_xy') is not None)
        fused=dict(bias['bias_only_fused']); physical={}; plain={}; scale_only={}
        selections={h:'bias_only' if fused.get(h)!=baseline.get(h) else 'baseline' for h in baseline}
        weights={h:0. for h in baseline}; diagnostic={}
        rate=0.
        if available:
            axis=np.asarray(geometry['axis_xy'],float)
            angle=float(np.arctan2(axis[1],axis[0]))
            while self.axis_history and t-self.axis_history[0][0]>.16: self.axis_history.popleft()
            if self.axis_history and t-self.axis_history[-1][0]<=.06:
                old=self.axis_history[0]; elapsed=t-old[0]
                difference=float(np.arctan2(np.sin(angle-old[1]),np.cos(angle-old[1])))
                if elapsed>=.04: rate=float(np.clip(difference/elapsed,-3.,3.))
            self.axis_history.append((t,angle))
            p=np.asarray(coefficient['position'],float); v=np.asarray(coefficient['velocity'],float)
            f2=float(np.clip(proxy['relative_f2_proxy'],.2,5.))
            for horizon in (.1,.2,.4):
                key=str(horizon)
                if key not in baseline: continue
                base=np.asarray(baseline[key]); bias_p=np.asarray(bias['bias_only_predictions'][key])
                candidates=[]
                for core,with_scale in ((self.scaled,True),(self.unscaled,False)):
                    s=scale['relative_scale'] if with_scale and scale['valid'] else 1.
                    gamma=scale['log_scale_rate_per_s'] if with_scale and scale['valid'] else 0.
                    factor=float(np.exp(gamma*horizon))
                    known=self.principal+factor*((p-self.principal)+(v-gamma*(p-self.principal))*horizon)
                    # Midpoint integration of a decaying image-axis angular rate.
                    integral=np.zeros(2)
                    for j in range(12):
                        tau=(j+.5)*horizon/12
                        turn=rate*.15*(1-np.exp(-tau/.15)); c,sn=np.cos(turn),np.sin(turn)
                        future_axis=np.array([[c,-sn],[sn,c]])@axis
                        integral+=(horizon-tau)*f2*future_axis*horizon/12
                    A=s*factor*np.array([[integral[0],.5*horizon*horizon,0.],
                                        [integral[1],0.,.5*horizon*horizon]])
                    predicted=known+A@core.theta
                    # An uncertain new model is kept observable but cannot
                    # pull a selected forecast farther than 40 native pixels.
                    correction=predicted-base
                    predicted=base+correction*min(1.,40./max(np.linalg.norm(correction),1e-9))
                    core.queue.append(dict(target_t=t+horizon,h=horizon,A=A,known=known,
                        baseline=base.copy(),bias=bias_p.copy(),physical=predicted.copy()))
                    core.queue=deque(sorted(core.queue,key=lambda q:q['target_t']))
                    candidates.append(predicted)
                physical[key]=candidates[0].tolist(); plain[key]=candidates[1].tolist()
                gamma=scale['log_scale_rate_per_s'] if scale['valid'] else 0.
                correction=(np.exp(gamma*horizon)-1)*(base-self.principal)-np.exp(gamma*horizon)*gamma*(p-self.principal)*horizon
                correction*=min(1.,20./max(np.linalg.norm(correction),1e-9))
                scale_only[key]=(base+correction).tolist()
                losses=list(self.scaled.loss[horizon]); weight=0.
                if self.scaled.updates>=10 and len(losses)>=10:
                    means=np.mean(np.array([q[1:] for q in losses]),axis=0)
                    if means[2]<(1-.05)*min(means[0],means[1]):
                        weight=min(.35,max(0.,(means[0]-means[2])/max(means[0],1.)))*geometry.get('confidence',0.)
                        if proxy.get('hidden_speed_assumption_used'): weight*=.5
                        fused[key]=((1-weight)*base+weight*candidates[0]).tolist()
                        selections[key]='projected_physics'
                weights[key]=float(weight)
                diagnostic[key]=dict(past_same_origin_pairs=len(losses),
                    past_error_means_baseline_bias_physics=np.mean(np.array([q[1:] for q in losses]),axis=0).tolist() if losses else None)
        return dict(active=any(weights.values()),weight=max(weights.values(),default=0.),
            weights_by_horizon=weights,fused=fused,physical=physical or None,
            unscaled_predictions=plain,scale_only_predictions=scale_only,
            bias_only_predictions=bias['bias_only_predictions'],bias_only_fused=bias['bias_only_fused'],
            selections_by_horizon=selections,diagnostics=diagnostic,
            scale=scale,axis_rate_rad_s=rate,feature_available=available,
            updates=self.scaled.updates,parameters_effective_pixel_units=self.scaled.theta.tolist(),
            unscaled_parameters=self.unscaled.theta.tolist(),force_calibrated=False,
            adaptation_uses='only matured same-origin .2 s body observations; no labels',
            model='relative-scale normalized transverse dynamics; image-up normal sign prior',
            physical_parameters_identified=False)
