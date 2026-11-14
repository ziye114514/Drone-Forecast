"""Bounded local temporal evidence without ellipses or a shared rotor plane.

Experimental CPU component, not a calibrated RPM/thrust sensor. Callers must
provide events already supported by the rotor front. Common periodic lighting
can pass this temporal test; geometry/rotor support is not replaced by it.
"""
import numpy as np


def local_temporal_phase(events, cutoff_s, center, velocity, candidates_hz,
        *, rotor_support_confirmed=False, cell_px=24., window_us=33333,
        max_cells=8, max_events_per_cell=4096):
    empty=dict(regions=[],shared_plane_required=False,rotor_identity_required=False,
        output_is_rotor_count=False,metric_RPM_or_thrust_measured=False,
        future_events_used=False,default_pipeline_connected=False)
    if not rotor_support_confirmed:return dict(empty,reason='requires_upstream_rotor_support')
    f=np.unique(np.asarray(candidates_hz,float))
    if f.ndim!=1 or len(f)>8 or not len(f) or not np.isfinite(f).all() or np.any((f<60)|(f>1200)):
        raise ValueError('Provide one to eight finite60..1200Hz current candidates')
    c=np.asarray(center,float);v=np.asarray(velocity,float)
    if c.shape!=(2,) or v.shape!=(2,) or not np.isfinite(np.r_[c,v]).all():
        raise ValueError('Finite image center and velocity required')
    end=round(cutoff_s*1e6);begin=end-window_us
    e=events[(events['t']>=begin)&(events['t']<end)]
    if not len(e):return dict(empty,reason='no_current_supported_events')
    # Translation-only body coordinates: no ellipse inversion at edge-on views.
    age=(end-e['t']).astype(float)/1e6
    xy=np.column_stack((e['x'],e['y'])).astype(float)+age[:,None]*v-c
    grid=np.floor(xy/cell_px).astype(np.int64)
    cells,ids,count=np.unique(grid,axis=0,return_inverse=True,return_counts=True)
    results=[]
    for cell in np.argsort(-count,kind='stable')[:max_cells]:
        selected=e[ids==cell];n=len(selected)
        entry=dict(cell=cells[cell].tolist(),events=n,qualified=False,candidates=[])
        if n>max_events_per_cell:
            results.append(dict(entry,reason='capacity_exceeded_no_temporal_subsampling'));continue
        times=(selected['t']-begin).astype(float)/1e6
        first=times<window_us/2e6;second=~first
        if min(first.sum(),second.sum())<8:
            results.append(dict(entry,reason='insufficient_half_window_support'));continue
        if min(np.ptp(times[first]),np.ptp(times[second]))<window_us/4e6:
            results.append(dict(entry,reason='insufficient_time_span'));continue
        polarity=2*(selected['p']>0).astype(float)-1
        a=polarity[first];b=polarity[second];a=a-a.mean();b=b-b.mean()
        harmonics=np.arange(1.,3.)
        phase=2*np.pi*times[:,None,None]*f[None,:,None]*harmonics[None,None]
        coefficients=np.mean(a[:,None,None]*np.exp(-1j*phase[first]),axis=0)
        template=np.real(np.sum(coefficients[None]*np.exp(1j*phase[second]),axis=2))
        template-=template.mean(0)
        projection=np.sum(b[:,None]*template,axis=0)
        denom=np.sqrt(np.sum(b*b)*np.sum(template*template,axis=0))
        corr=np.divide(projection,denom,out=np.zeros_like(projection),where=denom>1e-12)
        # First-harmonic support rejects unsupported subharmonic explanations;
        # it cannot uniquely infer blade count in symmetric real observations.
        first_strength=np.abs(coefficients[:,0])
        valid=(corr>=.65)&(first_strength>=.15)
        for j,value in enumerate(f):
            entry['candidates'].append(dict(frequency_hz=float(value),
                validation_correlation=float(corr[j]),first_harmonic_strength=float(first_strength[j]),
                phase_at_window_begin_rad=float(np.angle(coefficients[j,0])),
                qualified=bool(valid[j])))
        winners=np.flatnonzero(valid)
        if len(winners):
            best=int(winners[np.argmax(corr[winners])])
            entry.update(qualified=True,frequency_hz=float(f[best]),
                reason='current_local_temporal_template_validated',
                ambiguous_candidates=int(len(winners)),hypothesis_not_mechanical_RPM=True)
        else:entry['reason']='no_current_candidate_validated'
        results.append(entry)
    return dict(empty,regions=results,reason='current_local_temporal_evidence',
        current_window_us=window_us,qualified_regions=sum(r['qualified'] for r in results),
        cell_px=cell_px,max_cells=max_cells,max_events_per_cell=max_events_per_cell,
        current_candidates_only=True,previous_frequency_latch_used=False,
        polarity_template_first_half_then_validation_second_half=True)
