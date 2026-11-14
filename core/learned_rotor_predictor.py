"""Explicit experimental wrapper: native rotor pipeline + learned unknown rate.

Old factories are preserved. Model selection is a caller-provided file path,
never a runtime recording-name decision. Cross-fit evaluation models and the
shared model for a future unseen recording must not be conflated.
"""
from pathlib import Path
import json,hashlib,time,numpy as np
from run_rotor_only import RotorPredictor
from conditional_unknown_force import ConditionalUnknownForce


def load_model(path):
    p=Path(path);raw=p.read_bytes();model=json.loads(raw.decode('utf8'))
    w=np.asarray(model.get('weights') or [],float)
    visibility=bool(model.get('visibility_conditioned'))
    plane=bool(model.get('plane_conditioned'))
    if plane and not visibility:raise ValueError('Plane conditioning requires the visibility model')
    if w.shape!=((12,) if plane else (10,) if visibility else (5,)) or not np.isfinite(w).all():raise ValueError('Expected five native, ten visibility, or twelve plane-conditioned response coefficients')
    if visibility and not np.isfinite(float(model.get('visibility_context_mean',float('nan')))):raise ValueError('Finite training context mean required')
    if 'future_drive_multiplier' in model and (not np.isfinite(float(model['future_drive_multiplier'])) or float(model['future_drive_multiplier'])<0):raise ValueError('Future-drive multiplier must be finite and nonnegative')
    if model.get('reference_motion_conditioned'):raise ValueError('Research reference-motion model is not supported by this entry')
    if model.get('mode') not in ('disk','camera') or model.get('manual_labels_used') is not False:raise ValueError('Expected native-teacher disk/fixed-image response model, no annotation training')
    if model.get('net_force_history_used'):raise ValueError('History model is not supported by this five-coefficient public wrapper')
    return model,dict(sha256=hashlib.sha256(raw).hexdigest(),filename=p.name,training_samples=model.get('training_samples'),
        inference_input_is_native_rotor_only=True,metric_force_measured=False,accuracy_on_new_recordings_not_guaranteed=True,
        visibility_conditioned=visibility,plane_conditioned=plane,coefficient_count=len(w),
        future_drive_calibrated='future_drive_multiplier' in model,evaluation_sources_excluded_from_fit=model.get('all6_evaluation_sources_excluded_from_fit',False))


class LearnedRotorPredictor(RotorPredictor):
    def __init__(self,*args,response_model,**kwargs):
        model,self.response_model_info=load_model(response_model)
        if not kwargs.get('inverse_normal_response') or not kwargs.get('anisotropic_response'):
            raise ValueError('Learned response requires the inverse+anisotropic rotor-force candidate')
        super().__init__(*args,**kwargs)
        if model.get('plane_conditioned'):
            from plane_conditioned_response import PlaneResponse
            self.learned_response=PlaneResponse(model)
        elif 'future_drive_multiplier' in model:
            from future_drive_response import FutureDriveResponse
            self.learned_response=FutureDriveResponse(model)
        elif model.get('visibility_conditioned'):
            from visibility_response import VisibilityResponse
            self.learned_response=VisibilityResponse(model)
        else:self.learned_response=ConditionalUnknownForce(model)
    def step(self,events,t):
        started=time.perf_counter()
        row,frame=super().step(events,t);row=self.learned_response.step(row)
        row['learned_response_model']=self.response_model_info
        if row.get('conditional_unknown_force'):
            row.update(output_reference='visible_rotor_envelope_center',output_position_is_COM_truth=False)
        row['compute_ms']=(time.perf_counter()-started)*1000
        return row,frame
