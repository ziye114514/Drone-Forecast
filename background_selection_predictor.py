"""One rotor front; static and moving-background coordinate hypotheses.

Only current/past native background evidence supplies the hypothesis weight.
The original two-force mechanics and fixed native response model are retained.
These states share rotor observations and are never treated as independent.
"""
import copy
import time
from background_reference_predictor import BackgroundEndpointPredictor
from camera_reference_selection import ReferenceEvidenceWeight,ReferenceMemoryWeight,mix
from equilibrium_gain_prior import CovarianceOnlyObserver
from inverse_normal_response import InverseNormalResponse
from anisotropic_response import AnisotropicResponse
from learned_rotor_predictor import load_model
from plane_conditioned_response import PlaneResponse
from reference_qualification import QualifiedReferenceResponse
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation


class BackgroundSelectionPredictor(BackgroundEndpointPredictor):
    reference_weight_type=ReferenceEvidenceWeight
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.force.static_observer=CovarianceOnlyObserver()
        model,_=load_model(kwargs['response_model'])
        if not model.get('plane_conditioned'):
            raise ValueError('Use the unchanged native plane response model')
        self.static_inverse=InverseNormalResponse();self.static_disk=AnisotropicResponse()
        self.static_response=QualifiedReferenceResponse(PlaneResponse(model))
        self.static_boot=BootstrapNativeReference();self.static_visible=VisibleEnvelopeResponse()
        self.static_window=WindowPositionObservation(33333,20000)
        self.reference_weight=self.reference_weight_type()

    def step(self,events,t):
        began=time.perf_counter();camera,frame=super().step(events,t)
        r=copy.deepcopy(camera)
        for k in ('conditional_unknown_force','visibility_response','plane_conditioned_response',
                  'future_drive_response','native_fov_observation','physical',
                  'window_position_observation','native_reference_bootstrap',
                  'background_frame_motion','background_relative_motion'):
            r.pop(k,None)
        c,v,f,s=copy.deepcopy(self.force.static_result)
        s['two_point_initialization_selected_by_partial_entry']=camera['state'].get(
            'two_point_initialization_selected_by_partial_entry',False)
        r.update(coefficient=c,valid=bool(v),force=f,state=s,
            normal_turn_forecast=s.get('normal_turn_forecast'))
        r=self.static_disk.step(self.static_inverse.step(r))
        r=self.static_response.step(r);r=self.static_boot.step(r)
        r=self.static_window.step(self.static_visible.step(r))
        info=self.reference_weight.step(frame['background_reference_evidence'])
        result=mix(r,camera,info)
        result['compute_ms']=(time.perf_counter()-began)*1000
        return result,frame


class BackgroundMemorySelectionPredictor(BackgroundSelectionPredictor):
    reference_weight_type=ReferenceMemoryWeight
