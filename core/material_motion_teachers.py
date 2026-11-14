"""Matured native displacement teachers, excluding support/reference changes.

No evaluation annotations, physical COM truth, or predicted future controls.
The target sums actual image matches; the force observer is not its teacher.
"""
import numpy as np


def material_displacement(rows,start,end):
    previous=rows[start];total=np.zeros(2);used=0;gap=0.
    if not previous['motion'].get('current_observation') or not previous['body'].get('center'):
        return None,dict(qualified=False,reason='source_without_current_material_observation')
    for q in rows[start+1:end+1]:
        if not q['motion'].get('current_observation'):continue
        motion=q['motion'];estimate=motion.get('estimate') or {};dt=motion.get('dt_s')
        if (dt is None or not 0<dt<=.12+1e-8 or abs(dt-(q['t']-previous['t']))>1e-6 or
            q['body'].get('measurement_source') not in ('filtered_native_translation','expanded_rotor_support_reference') or
            not estimate.get('current_displacement_qualified',True) or estimate.get('peak_at_search_boundary')):
            return None,dict(qualified=False,reason='disconnected_or_unqualified_material_reference')
        delta=np.array([estimate.get('dx',np.nan),estimate.get('dy',np.nan)])
        shift=np.asarray(q['body'].get('measurement_reference_shift_xy') or [0.,0.])
        observed=np.asarray(q['body']['center'])-previous['body']['center']-shift
        if not np.isfinite(delta).all() or np.max(abs(observed-delta))>1e-6:
            return None,dict(qualified=False,reason='stored_match_does_not_explain_reference_change')
        total+=delta;used+=1;gap=max(gap,dt);previous=q
    if previous['t']!=rows[end]['t']:
        return None,dict(qualified=False,reason='endpoint_not_connected')
    return total,dict(qualified=True,matched_factors=used,maximum_matched_gap_s=gap,
        bbox_shape_not_used=True,reference_rebases_removed=True,independent_COM_truth=False)
