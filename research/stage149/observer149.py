"""148 current arc evidence plus 149 inertial identity and projected dynamics."""
from temporal148 import BranchObserver as Observer148,GeometryAdapter,TemporalRates
from residual148 import ResidualExpert
from inertia149 import InertialBank,PARAMS as INERTIA_PARAMS
from dynamics149 import ProjectedExpert,PARAMS as DYNAMICS_PARAMS

PARAMS=dict(inertia=INERTIA_PARAMS,dynamics=DYNAMICS_PARAMS,default_promoted=False)

class BranchObserver(Observer148):
    def reset_temporal(self):
        self.bank=InertialBank()
        self.geometry=GeometryAdapter(self.bank,self.w,self.h)
        self.temporal_rates=TemporalRates(self.bank)
        self.residual_expert=ResidualExpert()
        self.projected_expert=ProjectedExpert(self.w,self.h)
    def step(self,events,t,prepared,body,coefficient,baseline_predictions):
        result=super().step(events,t,prepared,body,coefficient,baseline_predictions)
        result['residual_expert148']=result['expert']
        result['inertia149']=self.bank.diagnostic
        result['expert']=self.projected_expert.step(t,body,result['geometry'],
            result['thrust_proxy'],coefficient,baseline_predictions)
        result['extra_predictions'].update(
            rotor_residual148=result['residual_expert148']['fused'],
            rotor_bias=result['expert']['bias_only_predictions'],
            rotor_bias_fused=result['expert']['bias_only_fused'],
            rotor_projected_unscaled=result['expert']['unscaled_predictions'],
            rotor_scale_only=result['expert']['scale_only_predictions'])
        return result
