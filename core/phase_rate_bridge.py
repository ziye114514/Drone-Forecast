"""Current spatial phase may arbitrate spectral candidates; no hidden RPM."""
import copy
import numpy as np


def phase_supported_rates(structure, rotors, rates, phase):
    result=copy.deepcopy(rates);accepted=[]
    hubs=structure.get('model_centers')
    if hubs is None:return result,dict(selected=accepted)
    hubs=np.asarray(hubs,float)
    for p in phase.get('rotors',[]):
        if not p.get('qualified') or not p.get('current_slot_observed'):continue
        slot=int(p['model_id']);f=p['frequency_hz']
        distances=[np.linalg.norm(np.array([r['cx'],r['cy']])-hubs[slot]) for r in rotors]
        if not distances:continue
        j=int(np.argmin(distances));r=rotors[j];rate=result[j]
        if distances[j]>max(4.,.15*r['major']) or not rate.get('valid'):continue
        candidates=[rate['frequency_hz']]+rate.get('spectral_candidates_hz',[])
        if not any(v and abs(np.log(f/v))<.08 for v in candidates):continue
        rate.update(ndft_frequency_hz=rate['frequency_hz'],frequency_hz=float(f),
            spatial_phase_supported=True,phase_support_source='current_rectified_spatial_moment',
            independent_RPM_truth=False)
        accepted.append(dict(region_index=j,model_id=slot,frequency_hz=float(f)))
    return result,dict(selected=accepted,hidden_frequencies_added=0)
