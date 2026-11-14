"""Two independently learned supports, one position/bias state.

Never mix count-core and occupancy-rim dimensions into one learned template.
Each source's first accepted geometric center is aligned to its current NN
box once. This is an image-coordinate gauge, not COM calibration.
"""
from pathlib import Path
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage122')]
from worker122 import PartialModel,geom,CenterBank,NAMES
PARAMS=dict(rule='Prefer current count-model geometry; otherwise use current independently learned occupancy-model geometry; otherwise count-model hold/body fallback.',
    structure='Two unchanged119 affine3 models, each learns its own five-full-frame dimensionless shape. Never insert occupancy ellipses into count learned shape.',
    coordinates='Keep each source first current body-minus-geometric offset while identity survives; aligned current geometry=source geometric center+source offset. Single107 bank; no position/velocity reset on source switch.',
    identity='Clear source offsets and shared bank if last current body gap>.5s, same original107 reset interval.',
    caveat='Offsets use NN BOX centers, not real COM. May retain source-dependent bias. Standard multi-model observation management, not a novelty claim.',default_promoted=False)
class MeasurementFusion:
    def __init__(self):self.anchors={};self.bank=CenterBank();self.last_body=None
    def step(self,t,body,axis,count,occupancy):
        if self.last_body is not None and t-self.last_body>.5:self.__init__()
        if body.get('detected'):self.last_body=t
        for name,d in [('count',count),('occupancy',occupancy)]:
            if d.get('current') and d.get('model_geometric_center') is not None and name not in self.anchors:
                self.anchors[name]=np.asarray(body['center'])-np.asarray(d['model_geometric_center'])
        name='count' if count.get('current') else 'occupancy' if occupancy.get('current') else 'body_fallback'
        d=copy.deepcopy(occupancy if name=='occupancy' else count)
        if d.get('current'):
            source='occupancy' if name=='occupancy' else 'count'
            d['model_geometric_center']=(np.asarray(d['model_geometric_center'])+self.anchors[source]).tolist()
        q=self.bank.step(t,body,d,axis)
        return dict(chosen=name,center=d,forecast=q,source_offsets={k:v.tolist() for k,v in self.anchors.items()},
            same_position_filter=True,source_switch_resets_velocity=False,source_offsets_are_COM_truth=False)
class Independent:
    def __init__(self):
        self.models={m:PartialModel('affine3') for m in ['count','occupancy']};self.fusion=MeasurementFusion()
    def step(self,t,body,axis,count,occupancy):
        outputs={}
        for name,rotors in [('count',count),('occupancy',occupancy)]:
            m=geom.rigid104.measure_four(rotors,body);_,d=self.models[name].step(t,body,rotors,axis,m);outputs[name]=d
        result=self.fusion.step(t,body,axis,outputs['count'],outputs['occupancy']);result['independent_models']=outputs;return result
