"""SDK-qualified brightness frequency as a relative input; no mechanical RPM."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage88'),str(ROOT/'research/stage82'),str(ROOT/'research/stage79'),str(ROOT/'research/stage72')]
from paired88 import Paired
from increment72 import ContinuousFrequency
from stream82 import StreamingActivity
from native79 import engine,checked_process,summarize


class NativeFrequency(ContinuousFrequency):
    def step(self,t,activity):
        nd=activity['native_sdk96'];candidates=[]
        if nd['qualified']:
            # These two fields are only scalar weights in the reused continuity
            # tracker. Unit second weight is NEUTRAL, not measured phase.
            candidates=[dict(qualified=True,frequency_hz=nd['frequency_hz'],tone_fraction=nd['mode_fraction'],phase_coherence=1.)]
        value,diagnostic=super().step(t,dict(spectral_candidates=candidates))
        diagnostic.update(measurement_source='installed_SDK_strict7',scalar_weight='mode_fraction times neutral1; no phase measurement',
            accepted_pixels=nd.get('mode_pixels',0),mechanical_RPM_verified=False)
        return value,diagnostic


class NativeActivity(StreamingActivity):
    def __init__(self,w,h):
        super().__init__(w,h);self.native96=engine(w,h,'strict7');self.output96=self.native96.get_empty_output_buffer()
    def step(self,e,bbox,t,velocity):
        v=super().step(e,bbox,t,velocity)
        v['native_sdk96']=summarize(checked_process(self.native96,e,self.output96,self.width,self.height),bbox,t,self.width,self.height)
        return v


class NativePaired(Paired):
    def reset(self):
        super().reset();self.inc.frequency=NativeFrequency()
