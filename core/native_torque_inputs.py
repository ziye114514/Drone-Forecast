"""Cheap current slot-force evidence for the opt-in RAW torque observer.

No author phase DLL or host event retention is needed. Native pixel periods
remain harmonic-ambiguous proxies, which are reported as such. Pooled whole-
body frequencies are never replicated into artificial individual motors.
"""
import numpy as np
from axis_amplitude_bridge import current_axis_qualified


def selected(row,period):
    st=row['structure'];assignments={a['model_id']:a['region_index'] for a in period.get('mapping',{}).get('assignments',[])}
    samples={int(i):float(f) for i,f in period.get('current_individual_frequency_hz',{}).items()}
    n=np.asarray(st.get('axis_xy') if st.get('axis_xy') is not None else [],float)
    if not samples and n.shape==(2,) and n@n>1e-12:
        for key,value in period.get('observed_components',{}).items():
            if not key.startswith('slot:'):continue
            weight=float(np.asarray(value)@n/(n@n))
            if weight>0:samples[int(key.split(':')[-1])]=np.sqrt(weight*360000)
    found=[]
    for i,f in samples.items():
        index=assignments.get(i)
        if index is None:continue
        g=row['geometry']['rotors'][index]
        if g.get('fov_clipped') or not g.get('current_observation',True):continue
        found.append((f'slot:{st.get("geometry_epoch",0)}:{i}',f,g))
    return found


def components(row,period):
    st=row['structure'];center=st.get('current_geometric_center');reference='current_five_circle_center_prior_conditioned'
    if center is None:center=row['body'].get('center');reference='current_visible_envelope_center_not_COM'
    if center is None or not row['body'].get('detected') or not current_axis_qualified(st):return {},reference
    center=np.asarray(center,float);out={}
    for key,f,g in selected(row,period):
        p=np.asarray([g['cx'],g['cy']],float);out[key]=-(f/300)**2/4*(p-center)/(g['major']/2)
    return out,reference
