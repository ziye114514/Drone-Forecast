"""Two independent geometry arms, without the old process-global detector swap.

Only the prepared counts are shared read-only. Each arm owns its frontend,
track/recovery state and structure model. Results join before forecasting.
"""
from pathlib import Path
import sys,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage138'),str(ROOT/'research/stage136'),str(ROOT/'research/stage129')]
from tiles138 import TileFrontend,count_detect,occupancy_detect,MergeRecovery,polarity
from boundary129 import bounded
from fast_geometry136 import baseline_for
_ORIGINAL_POLARITY=polarity.PolarityFrontend.detect_prepared
PARAMS=dict(workers=2,shared_input='read-only current ON/OFF counts',
    mutable_state='separate frontend and model per source',join_before_prediction=True,
    default_promoted=False)

def instance_detect(self,events,t,prepared):
    detector=getattr(self,'isolated_detector156',None)
    if detector is None:return _ORIGINAL_POLARITY(self,events,t,prepared)
    self.prepared=prepared;self.candidates=[]
    observation,total=detector(self,events,t)
    if observation['detected']:
        for serialized,measured in zip(observation['rotors'],self.previous_rotors):
            serialized['activity']=float(measured['activity'])
    return observation,total

class IsolatedFrontend(TileFrontend):
    def __init__(self,w,h,mode):
        super().__init__(w,h,mode)
        self.isolated_detector156=occupancy_detect if mode.startswith('occupancy') else count_detect
    def detect_prepared(self,events,t,prepared):
        # Retain the recovery/provenance wrapper; bypass only its outer global
        # function-switching adapters. Polarity dispatch now belongs to self.
        return MergeRecovery.detect_prepared(self,events,t,prepared)

def fronts(w,h):
    polarity.PolarityFrontend.detect_prepared=instance_detect
    return dict(count=IsolatedFrontend(w,h,'count_global'),occupancy=IsolatedFrontend(w,h,'occupancy_roi'))

def one(front,model,e,t,prepared,body,w,h,use_geometry):
    tick=time.perf_counter()
    obs=front.detect_prepared(e,t,prepared)[0] if use_geometry else dict(t=t,detected=False,rotors=[],reason='geometry_not_scheduled')
    frontend_ms=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
    rotors,boundary=bounded(obs.get('rotors',[]),w,h)
    baseline=baseline_for(model,t,rotors,body)
    _,diagnostic=model.step(t,body,rotors,dict(valid=False),baseline)
    return dict(observation=obs,rotors=rotors,boundary=boundary,diagnostic=diagnostic,
        frontend_ms=frontend_ms,model_ms=(time.perf_counter()-tick)*1000)

class Pool:
    def __init__(self):self.executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix='geometry156')
    def step(self,fronts,models,e,t,prepared,body,w,h,use_geometry):
        for value in prepared.values():value.setflags(write=False)
        jobs={name:self.executor.submit(one,front,models[name],e,t,prepared,body,w,h,use_geometry) for name,front in fronts.items()}
        return {name:job.result() for name,job in jobs.items()}
    def close(self):self.executor.shutdown(wait=True)
