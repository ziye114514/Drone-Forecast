"""Current phase/native slot union with explicit estimator provenance.

Same slot is never counted twice. A pooled whole-body component is replaced
when resolved current sources exist, not added on top of them. Switching an
estimator changes component ownership so the existing observer transfers its
baseline to the unknown term instead of inventing a thrust step.
"""
import copy
import numpy as np


def merge(row,period):
    original=row['period_selection'];u=row['force'].get('input_u')
    old=copy.deepcopy(original)
    if not (period.get('meta') or {}).get('observed_components'):
        return None if u is None else np.asarray(u,float),.7 if u is not None else 0.,old
    current=period['meta'];epoch=row['structure'].get('geometry_epoch',0)
    phase={};slots=set();frequencies={};records=[]
    for record in current.get('records',[]):
        key=str(record.get('track_id'));vector=current['observed_components'].get(key)
        if vector is None or not record.get('current_force_admitted'):continue
        slot=int(record['model_slot']);new_key=f'phase_slot:{epoch}:{slot}'
        phase[new_key]=vector;slots.add(slot);frequencies[str(slot)]=record['proxy_twofold_hz']
        r=copy.deepcopy(record);r['phase_instance_id']=r['track_id'];r['track_id']=new_key;records.append(r)
    if not phase:return None if u is None else np.asarray(u,float),.7 if u is not None else 0.,old
    native={k:v for k,v in old.get('observed_components',{}).items() if k.startswith(f'slot:{epoch}:') and int(k.split(':')[-1]) not in slots}
    components=dict(native,**phase);value=np.sum(list(components.values()),axis=0)
    sigma=float(np.sqrt((old.get('input_sigma_proxy',0.) if native else 0.)**2+
        .04**2*sum(np.dot(v,v) for v in phase.values())))
    meta=dict(current,source='current_disjoint_phase_native_slot_union',observed_components=components,
        input_sigma_proxy=sigma,input_uncertainty_calibrated=False,records=records,
        learning_input_current=bool(not native or old.get('learning_input_current',False)),
        current_phase_model_slots=sorted(slots),current_phase_frequency_hz=frequencies,
        whole_body_pool_added_to_resolved_sources=False,overlapping_native_slots_removed=True,
        estimator_switch_is_unknown_force_transfer=True,hidden_frequencies_measured=0)
    return value,.7,meta


def torque_period(row,period):
    """Stable geometry-slot provenance; phase restarts do not rename a motor."""
    result=copy.deepcopy(period);epoch=row['structure'].get('geometry_epoch',0)
    for r in result.get('meta',{}).get('records',[]):
        r['phase_instance_id']=r['track_id'];r['track_id']=f'phase_slot:{epoch}:{r["model_slot"]}'
    return result
