"""Forecast-baseline residual correction, with an explicit bias-only ablation.

Each horizon learns from ITS OWN already arrived target. Event features may
earn weight only by outperforming both the baseline and its shared bias model
on the same past origins. Pixel corrections are not calibrated physical force.
"""
from collections import deque
import numpy as np

PARAMS=dict(horizons=[.1,.2,.4],bias_variance=100.,event_variance=100.,forget=.995,
            max_correction_px=20.,past_loss_window_s=1.2,min_past_pairs=10,
            max_fusion_weight=.35,event_margin=.05,feature_history_s=.20)

class ResidualExpert:
    def __init__(self):
        self.queue=deque();self.features=deque()
        self.models={h:dict(bias=np.zeros(2),biasP=25.,B=np.zeros((2,2)),P=np.eye(2)*10000.,
                           updates=0,event_updates=0,loss=deque()) for h in PARAMS['horizons']}

    @staticmethod
    def bounded(value):
        value=np.asarray(value,float);length=np.linalg.norm(value)
        return value*min(1.,PARAMS['max_correction_px']/max(length,1e-9))

    def step(self,t,body,geometry,proxy,baseline):
        while self.queue and self.queue[0]['target_t']<=t+1e-8:
            old=self.queue.popleft()
            if abs(t-old['target_t'])>.011 or not body.get('detected'):continue
            model=self.models[old['h']];truth=np.asarray(body['center'],float)
            error=truth-old['baseline'];innovation=np.clip(error-model['bias'],-60,60)
            gain=model['biasP']/(model['biasP']+PARAMS['bias_variance'])
            model['bias']=np.clip(model['bias']+gain*innovation,-40,40)
            model['biasP']=(1-gain)*model['biasP']/PARAMS['forget']+.05
            model['updates']+=1
            if old['event_valid']:
                z=old['z'];P=model['P'];R=PARAMS['event_variance']
                K=P@z/(R+z@P@z)
                residual=np.clip(truth-old['bias_prediction']-model['B']@z,-60,60)
                model['B']+=np.outer(residual,K);model['B']=np.clip(model['B'],-2000,2000)
                model['P']=(P-np.outer(K,z)@P)/PARAMS['forget']
                model['P']=(model['P']+model['P'].T)*.5;model['event_updates']+=1
            model['loss'].append((t,float(np.linalg.norm(truth-old['baseline'])),
                float(np.linalg.norm(truth-old['bias_prediction'])),
                float(np.linalg.norm(truth-old['event_prediction'])) if old['event_valid'] else None))
        for model in self.models.values():
            while model['loss'] and t-model['loss'][0][0]>PARAMS['past_loss_window_s']:model['loss'].popleft()
        event_valid=bool(proxy.get('valid') and geometry.get('axis_xy') is not None)
        current=np.asarray(geometry['axis_xy'])*np.clip(proxy['relative_f2_proxy'],.2,5.) if event_valid else None
        while self.features and t-self.features[0][0]>PARAMS['feature_history_s']:self.features.popleft()
        feature_delta=np.zeros(2)
        if event_valid:
            if len(self.features)>=3:feature_delta=current-np.mean([q[1] for q in self.features],axis=0)
            self.features.append((t,current.copy()))
        fused={};bias_predictions={};bias_fused={};event_predictions={};weights={};selections={};diagnostics={}
        for h,model in self.models.items():
            key=str(h)
            if key not in baseline:continue
            base=np.asarray(baseline[key],float);z=.5*h*h*feature_delta
            bias_prediction=base+self.bounded(model['bias'])
            event_prediction=base+self.bounded(model['bias']+model['B']@z)
            losses=list(model['loss']);weight=0.;selection='baseline';chosen=base
            bmean=cmean=emean=None;paired=[p for p in losses if p[3] is not None]
            if len(losses)>=PARAMS['min_past_pairs']:
                bmean=float(np.mean([p[1] for p in losses]));cmean=float(np.mean([p[2] for p in losses]))
                if cmean<bmean:
                    weight=min(PARAMS['max_fusion_weight'],(bmean-cmean)/max(bmean,1.))
                    selection='bias_only';chosen=bias_prediction
            bias_fused[key]=((1-weight)*base+weight*chosen).tolist()
            if event_valid and len(paired)>=PARAMS['min_past_pairs']:
                before=float(np.mean([p[1] for p in paired]));bias=float(np.mean([p[2] for p in paired]));emean=float(np.mean([p[3] for p in paired]))
                if emean<(1-PARAMS['event_margin'])*min(before,bias):
                    weight=min(PARAMS['max_fusion_weight'],(before-emean)/max(before,1.))*geometry.get('confidence',0.)
                    selection='event_residual';chosen=event_prediction
            fused[key]=((1-weight)*base+weight*chosen).tolist();weights[key]=float(weight);selections[key]=selection
            bias_predictions[key]=bias_prediction.tolist()
            if event_valid:event_predictions[key]=event_prediction.tolist()
            diagnostics[key]=dict(updates=model['updates'],event_updates=model['event_updates'],past_pairs=len(losses),
                event_past_matched_pairs=len(paired),past_baseline_mean=bmean,past_bias_mean=cmean,past_event_mean=emean,
                bias_px=model['bias'].tolist(),event_gain=model['B'].tolist())
            self.queue.append(dict(target_t=t+h,h=h,baseline=base,bias_prediction=bias_prediction,
                event_prediction=event_prediction,event_valid=event_valid,z=z.copy()))
        # Horizons are inserted per origin, not in target-time order.
        self.queue=deque(sorted(self.queue,key=lambda q:q['target_t']))
        return dict(active=any(w>0 for w in weights.values()),weight=max(weights.values(),default=0.),
                    weights_by_horizon=weights,fused=fused,physical=event_predictions or None,
                    bias_only_predictions=bias_predictions,bias_only_fused=bias_fused,selections_by_horizon=selections,diagnostics=diagnostics,
                    model='baseline residual in pixels; shared bias + centered event feature',feature_delta_xy=feature_delta.tolist(),
                    force_calibrated=False,updates=max([m['updates'] for m in self.models.values()]),
                    adaptation_uses='only same-horizon already matured body observations',
                    event_advantage_requires_same_origins_bias_ablation=True)
