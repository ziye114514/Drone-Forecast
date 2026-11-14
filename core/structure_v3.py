"""Adapter selecting mechanism3, with explicit lumped-disturbance controls."""
from pathlib import Path
import copy
import hashlib
import numpy as np
from five_circle_model import FiveCircleModel,PARAMS as STRUCTURE
from lumped_disturbance import LumpedDisturbance,DelayedResidualDisturbance,forecasts,PARAMS as DISTURBANCE
from live_structure_bridge import V7StructurePair
from structure_activity import WholeBodyPeriodicity
from whole_body_axis import measure as body_axis


class StructureV3:
    def __init__(self,strategy=3):
        self.strategy=strategy;self.models={i:FiveCircleModel(i) for i in (1,2,3)}
        self.motion={i:LumpedDisturbance() for i in self.models}
        self.with_input=LumpedDisturbance();self.bias={};self.last_epoch={}
        self.delayed=DelayedResidualDisturbance();self.delayed_no_input=DelayedResidualDisturbance()
        self.live_pair=V7StructurePair()
        self.activity=WholeBodyPeriodicity()

    def protocol(self):
        root=Path(__file__).resolve().parent
        files=['run_five_circle_v3.py','structure_v3.py','five_circle_model.py','lumped_disturbance.py','live_structure_bridge.py','structure_activity.py','whole_body_axis.py']
        return dict(selected_mechanism=self.strategy,structure=STRUCTURE,disturbance=DISTURBANCE,
            sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files},
            force_input='assumed200px/s2 gain times conditional relative frequency squared; NOT known Newton force',
            controls=['independent_rotors','four_only_rigid','visibility_switching','switching_without_thrust_proxy'],
            no_target_IMU=True,no_metric_mass_or_wind_identification=True)

    def step(self,t,body,coefficient,baseline,measurement,image=None,prepared=None):
        result=copy.deepcopy(measurement);geometry=measurement['geometry'];models={};motion={}
        extra=result.setdefault('extra_predictions',{})
        for strategy,model in self.models.items():
            g=model.step(t,body,geometry);models[str(strategy)]=g
            if strategy==3 and image is not None and g['current_geometric_center'] is None and g['mode']!='edge_on':
                axis=body_axis(image,body,prepared);g['whole_body_axis_attempt']=axis
                if axis['valid']:
                    g.update(mode='edge_on',axis_xy=axis['normal_candidates'][0],geometric_weight=.35,
                        reason='native_whole_body_axis_without_rotor_segmentation',axis_support=axis,
                        mode_changed=g['previous_mode']!='edge_on')
                    model.mode='edge_on'
            if self.last_epoch.get(strategy)!=g['geometry_epoch']:
                self.bias.pop(strategy,None);self.last_epoch[strategy]=g['geometry_epoch']
            geometric=g['current_geometric_center'];body_point=g['body_fallback']
            if g['mode'] in ('four','independent') and g['observed_count']==4 and geometric is not None and body_point is not None:
                self.bias.setdefault(strategy,np.asarray(geometric)-np.asarray(body_point))
            point=(np.asarray(geometric)-self.bias[strategy]).tolist() if geometric is not None and strategy in self.bias else body_point
            c,audit=self.motion[strategy].step(t,point,identity=g['geometry_epoch'])
            motion[str(strategy)]=dict(coefficient=c,diagnostic=audit,position_input=point,
                gauge='First full geometry-to-body offset fixed per identity; body fallback uses same image gauge')
            if audit['ready']:extra['structure'+str(strategy)+'_disturbance']=forecasts(c)
        chosen=models[str(self.strategy)];base_motion=motion[str(self.strategy)]
        proxy=measurement['thrust_proxy'];axis=chosen.get('axis_xy');u=None
        activity=self.activity.step(t,measurement.get('rates',[]),chosen['geometry_epoch'])
        proxy_source='legacy_per_rotor_proxy'
        if not proxy.get('valid') and activity['valid']:
            proxy=activity;proxy_source='whole_object_periodic_feature'
        input_assumptions=[]
        if self.strategy==3 and proxy.get('valid') and axis is not None:
            axis=np.asarray(axis,float)
            if axis[1]>0:axis=-axis
            u=200.*np.clip(proxy['relative_f2_proxy'],.2,5.)*axis
            input_assumptions=['image_up_normal_sign','default_effective_gain200',
                'conditional_frequency_proxy_not_verified_mechanical_RPM']
            if proxy.get('hidden_speed_assumption_used'):input_assumptions.append('hidden_speed_hypothesis')
        c,audit=self.with_input.step(t,base_motion['position_input'],u,identity=chosen['geometry_epoch'])
        if audit['ready']:extra['structure3_assumed_thrust_disturbance']=forecasts(c)
        delayed=self.delayed.step(t,body,baseline,u,identity=chosen['geometry_epoch'])
        delayed_control=self.delayed_no_input.step(t,body,baseline,identity=chosen['geometry_epoch'])
        extra['structure3_delayed_disturbance']=delayed['predictions']
        extra['structure3_delayed_no_input']=delayed_control['predictions']
        result['five_circle_v3']=dict(selected_mechanism=self.strategy,selected=chosen,
            models=models,motion=motion,assumed_thrust_motion=dict(coefficient=c,diagnostic=audit,
                input_assumptions=input_assumptions),
            delayed_disturbance=delayed,delayed_no_input=delayed_control,
            object_periodicity=activity,input_proxy_source=proxy_source,
            preferred_research_forecast='structure3_delayed_disturbance',
            physics_input_is_measured_force=False,baseline_predictions_unchanged=True)
        live_pair,live_predictions=self.live_pair.step(t,body,chosen,geometry.get('observed_rotors',[]),image)
        result['five_circle_v3']['live_position_pair']=live_pair
        extra.update(live_predictions)
        return result
