"""Map a source-window position forecast to a requested observation window.

The original cutoff/target timestamps remain unchanged. Under a window-centre
approximation, source33.333ms and target20ms positions have a6.6665ms offset.
This is an observation model, not future-event access, a fitted lag, or a
claim that a native event envelope is an exact uniformly exposed image.
"""
import copy


class WindowPositionObservation:
    def __init__(self,source_support_us=33333,target_support_us=20000):
        if not 0<=target_support_us<=source_support_us:raise ValueError('Target support must be within0..source support')
        self.source=int(source_support_us);self.target=int(target_support_us)
    def step(self,row):
        r=copy.deepcopy(row)
        if not r.get('coefficient'):return r
        r['window_position_observation']=dict(source_support_us=self.source,target_support_us=self.target,
            effective_time_difference_s=(self.source-self.target)*.5e-6,
            window_centres_are_approximation=True,fitted_delay=False,
            source_cutoff_and_requested_target_timestamps_unchanged=True,
            internal_p_v_a_force_detector_unchanged=True)
        r['output_reference']='native_visible_rotor_envelope_center_in_requested_observation_window'
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r


def predict(row,horizons):
    from physical_publication import evaluate
    r=copy.deepcopy(row);info=r.pop('window_position_observation');delta=float(info['effective_time_difference_s'])
    original=[float(h) for h in horizons];shifted=[h+delta for h in original]
    values=evaluate(r,shifted)
    return {str(h):values[str(s)] for h,s in zip(original,shifted)}
