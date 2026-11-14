"""Evaluate the SAME Singer/normal/force-memory rollout at actual output time.

No interpolation between saved forecasts and no extra trajectory filter. This
supports the target-candidate mechanics; unsupported trend models are rejected.
"""
import copy
import numpy as np
from adaptive_force_memory import kernel


def freeze(row,last_image_t):
    result=dict(t=row['t'],valid=row['valid'],last_image_t=last_image_t,
        coefficient=copy.deepcopy(row['coefficient']),force=copy.deepcopy(row['force']),
        normal_turn_forecast=copy.deepcopy(row.get('normal_turn_forecast')),
        adaptive_force_memory=copy.deepcopy(row.get('adaptive_force_memory')),
        thrust_trend_forecast=copy.deepcopy(row.get('thrust_trend_forecast')))
    state=row.get('state') or {}
    if state.get('layout_reference_is_uncertain_offset'):
        result['output_reference']=state['output_reference']
    if row.get('conditional_unknown_force'):
        result['conditional_unknown_force']=copy.deepcopy(row['conditional_unknown_force'])
        result['output_reference']='visible_rotor_envelope_center'
    if row.get('future_drive_response'):
        result['future_drive_response']=copy.deepcopy(row['future_drive_response'])
    if row.get('native_reference_motion'):
        result['native_reference_motion']=copy.deepcopy(row['native_reference_motion'])
    if row.get('native_fov_observation'):
        result['native_fov_observation']=copy.deepcopy(row['native_fov_observation'])
        result['output_reference']='predicted_visible_native_rotor_envelope_center'
    if row.get('background_unknown_response'):
        result['background_unknown_response']=copy.deepcopy(row['background_unknown_response'])
    if row.get('rotor_scale_response'):
        result['rotor_scale_response']=copy.deepcopy(row['rotor_scale_response'])
    if row.get('window_position_observation'):
        result['window_position_observation']=copy.deepcopy(row['window_position_observation'])
        result['output_reference']=row.get('output_reference','native_envelope_center_in_requested_observation_window')
    if row.get('background_frame_motion'):
        result['background_frame_motion']=copy.deepcopy(row['background_frame_motion'])
    if row.get('background_reference_model_average'):
        result['background_reference_model_average']=copy.deepcopy(row['background_reference_model_average'])
    return result


def evaluate(row,horizons):
    c=row['coefficient'];f=row['force']
    if c is None:return {}
    if row.get('background_reference_model_average'):
        from camera_reference_selection import predict
        return predict(row,horizons)
    if row.get('window_position_observation'):
        from window_position_observation import predict
        return predict(row,horizons)
    if row.get('native_fov_observation'):
        from native_fov_observation import predict
        return predict(row,horizons)
    if row.get('background_frame_motion'):
        from background_relative_force import predict
        return predict(row,horizons)
    if row.get('rotor_scale_response'):
        from rotor_scale_response import predict
        return predict(row,horizons)
    if row.get('native_reference_motion'):
        from native_reference_motion import predict
        return predict(row,horizons)
    if row.get('background_unknown_response'):
        from background_unknown_response import predict
        return predict(row,horizons)
    if row.get('rotor_residual_gp'):
        from rotor_residual_gp import predict
        return predict(row,horizons)
    if row.get('conditional_unknown_force'):
        from conditional_unknown_force import predict
        return predict(row,horizons)
    if row.get('future_drive_response'):
        from future_drive_response import predict
        return predict(row,horizons)
    if row.get('calibrated_force_rollout'):
        from calibrated_force_rollout import predict
        return predict(row,horizons)
    if f.get('response_model_mixture'):
        horizons=list(horizons);result={str(float(h)):np.zeros(2) for h in horizons}
        for component in f['response_model_mixture']:
            values=evaluate(component,horizons)
            for h,p in values.items():result[h]+=float(component['weight'])*np.asarray(p)
        return {h:p.tolist() for h,p in result.items()}
    if f.get('anisotropic_disk_response') or f.get('visible_force_rate'):
        if (row.get('adaptive_force_memory') or {}).get('applied'):raise ValueError('Anisotropic response with legacy memory is undefined')
        from anisotropic_response import rollout
        return rollout(c,f,row.get('normal_turn_forecast') or {},horizons)
    if (row.get('thrust_trend_forecast') or {}).get('applied'):raise ValueError('Thrust trend rollout not supported by this publisher')
    if f.get('future_net_force_model') in ('matern32','integrated_ou_rate'):
        if (row.get('normal_turn_forecast') or {}).get('applied') or (row.get('adaptive_force_memory') or {}).get('applied'):
            raise ValueError('Additional legacy force-memory corrections are not defined for Matern32')
        if f['future_net_force_model']=='matern32':
            from matern_force_observer import coefficients
        else:
            from integrated_force_rate import coefficients
        result={};p=np.asarray(c['position']);v=np.asarray(c['velocity']);a=np.asarray(f['acceleration_px_s2']);j=np.asarray(f['jerk_px_s3'])
        for h in horizons:
            h=float(h)
            if not np.isfinite(h) or not 0<=h<=1.:raise ValueError('Rollout horizon must be0..1s')
            ka,kj=coefficients(h);result[str(h)]=(p+h*v+ka*a+kj*j).tolist()
        return result
    tau=f.get('acceleration_correlation_s')
    if tau is None:raise ValueError('Publisher requires the Singer force model')
    lam=-1/float(tau);p=np.asarray(c['position']);v=np.asarray(c['velocity']);a=np.asarray(f['acceleration_px_s2'])
    normal=row.get('normal_turn_forecast') or {};memory=row.get('adaptive_force_memory') or {}
    omega=normal.get('trend',{}).get('shrunk_angular_rate_rad_s',0.) if normal.get('applied') else 0.
    alpha=normal.get('trend',{}).get('angular_acceleration_rad_s2',0.) if normal.get('applied') else 0.
    def turn_kernel(rate,h):
        if f.get('future_normal_response_model')=='drag_filtered_normal_derivative':
            from inverse_normal_response import normal_drive_kernel
            angle_tau=normal.get('trend',{}).get('angular_acceleration_correlation_s',normal.get('trend',{}).get('torque_innovation_correlation_s'))
            return normal_drive_kernel(rate,h,omega,alpha,angle_tau)
        tilt=normal.get('projected_tilt_trend') or {}
        if tilt.get('qualified'):
            from projected_tilt_trend import projected_normal_kernel
            return projected_normal_kernel(rate,h,omega,tilt['inclination_rad'],tilt['inclination_rate_rad_s'])
        if alpha:
            angle_tau=normal.get('trend',{}).get('angular_acceleration_correlation_s',normal.get('trend',{}).get('torque_innovation_correlation_s'))
            if angle_tau:
                from rotor_torque_innovation import rotation_kernel
                return rotation_kernel(rate,h,omega,alpha,angle_tau)
            from endpoint_normal_trend import rotation_kernel
            return rotation_kernel(rate,h,omega,alpha)
        return kernel(rate+1j*omega,h)
    known=complex(*(float(f['effective_gain_px_s2'])*np.asarray(f['input_u']))) if f.get('active') and f.get('input_u') is not None else 0j
    if normal.get('applied') and f.get('normal_turn_vector_px_s2') is not None:
        known=complex(*f['normal_turn_vector_px_s2'])
    use_memory=bool(memory.get('corrections_px'))
    long=float(memory.get('weight',0.)) if use_memory else 0.
    short=float(memory.get('short_weight',0.)) if use_memory else 0.
    result={}
    for h in horizons:
        h=float(h)
        if not np.isfinite(h) or not 0<=h<=1.:raise ValueError('Rollout horizon must be0..1s')
        k=kernel(lam,h);value=p+h*v+k*a
        rotate=(turn_kernel(lam,h)-k)*known
        value=value+np.array([rotate.real,rotate.imag])
        if use_memory:
            delta=(kernel(0.,h)-k)*a
            z=((turn_kernel(0.,h)-kernel(0.,h))-(turn_kernel(lam,h)-k))*known
            delta=delta+np.array([z.real,z.imag]);value=value+long*delta
            if short:
                fast=-10.;delta=(kernel(fast,h)-k)*a
                z=((turn_kernel(fast,h)-kernel(fast,h))-(turn_kernel(lam,h)-k))*known
                value=value+short*(delta+np.array([z.real,z.imag]))
        if f.get('external_unknown_rate_px_s3') is not None:
            from separated_force_response import rate_kernel
            value+=rate_kernel(lam,-1/float(f['external_rate_correlation_s']),h)*np.asarray(f['external_unknown_rate_px_s3'])
        result[str(h)]=value.tolist()
    return result


def publication(state,source_now,deadline,index):
    result=dict(index=index,t=source_now,deadline_t=deadline,valid=False,
        physical={},source_t=None,source_age_s=None,current_image_observation=False,
        output_is_model_projection=True,new_detector_or_trajectory_branch=False)
    if state is None:return dict(result,reason='awaiting_first_state')
    if state.get('output_reference'):
        result.update(output_reference=state['output_reference'],output_position_is_COM_truth=False,
            layout_center_is_output_position=False)
    age=source_now-state['t'];last=state.get('last_image_t')
    result.update(source_t=state['t'],source_age_s=age,last_image_t=last)
    if not state['valid'] or age<0 or age>.16 or last is None or source_now-last>.16:
        return dict(result,reason='no_recent_valid_observed_state')
    values=evaluate(state,[age,age+.1,age+.2,age+.4])
    result.update(valid=True,reason='same_physical_rollout_at_actual_output_time',
        position_now=values[str(float(age))],
        physical={str(h):values[str(float(age+h))] for h in (.1,.2,.4)},
        target_times={str(h):source_now+h for h in (.1,.2,.4)})
    return result
