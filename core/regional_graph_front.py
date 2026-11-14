"""Optional corrected regional fallback in the existing CUDA service."""
import numpy as np
from cuda_regional_graph import RegionalGraph

class RegionalGraphFallback:
    def __init__(self,primary):
        self.primary=primary;self.regional=RegionalGraph()

    def process(self,events,t,window_us=33333):
        activity,period,native,meta=self.primary.process(events,t,window_us)
        # Do not replace established per-pixel evidence with regional evidence.
        if meta['accepted_pixels']>=12 and len(native)>=12:
            meta['regional_graph_fallback']=dict(used=False,reason='individual_pixel_entry_available')
            return activity,period,native,meta
        arrays,summary,info=self.regional.measure(events,t,window_us)
        seeds=arrays[5].reshape(-1)>0
        if not info['qualified_groups'] or np.count_nonzero(seeds)<3:
            meta['regional_graph_fallback']=dict(info,used=False,reason='no_qualified_regional_native_seed')
            return activity,period,native,meta
        on,off,on20,off20,support,_=arrays.reshape(6,-1)
        active=np.flatnonzero(on+off);p=np.flatnonzero(support)
        activity=np.stack((active,on[active],off[active],np.zeros(len(active)),
            (on[active]+off[active])>=6,on20[active],off20[active]),1).astype(np.float32)
        native=np.stack((p,seeds[p]),1).astype(np.float32)
        # No candidate is assigned to a motor or per-pixel phase bank. Hidden
        # and uncalibrated thrust remain in the unknown-force state.
        period=np.empty((0,12),np.float32)
        meta.update(accepted_pixels=0,eligible_pixels=0,regional_seed_groups=info['qualified_groups'],
            regional_seed_pixels=int(seeds.sum()),regional_only_motor_frequency_unobserved=True,
            regional_graph_fallback=dict(info,used=True,reason='corrected_regional_timing_on_original_count_support'),
            parameters=dict(meta['parameters'],corrected_regional_graph=True,
                vote_cap='one_minimum_time_independent_hash_vote_per_pixel_polarity_short_block',
                mechanical_RPM_measured=False))
        return activity,period,native,meta
