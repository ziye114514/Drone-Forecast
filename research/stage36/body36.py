"""Isolated support/reseeding variants; execute the immutable stage35 implementation."""
from pathlib import Path
import copy,importlib.util

ROOT=Path(__file__).resolve().parents[2]


def load_body(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'research/stage35/body35.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


strict=load_body('strict_body35_for36')
METHODS=['baseline','similarity_long','translation_long','support3_similarity','support3_translation','replenish3_similarity']
PARAMS=dict(stage35=copy.deepcopy(strict.PARAMS),new_minimum_points=3,replenish_below=40,
    methods=METHODS,replenish='Only after accepted current body motion, find stronger current corners inside its box; keep original independent-geometry age, original extent and oriented corners. No reseeding after rejection; no G/semantic identity guarantee.')


class Replenished:
    def __init__(self,w,h,method):
        self.method=method;module=load_body('variant_body35_'+method);module.PARAMS['minimum_points']=PARAMS['new_minimum_points']
        self.body=module.BodyTracker(w,h,'translation_long' if method=='support3_translation' else 'similarity_long')

    def step(self,image,current,strong,t,o,n):
        q,a=self.body.step(image,current,strong,t,o,n);a=dict(a,minimum_points=3,feature_reseeded=False)
        if self.method=='replenish3_similarity' and a['source']=='body_flow' and len(self.body.points)<PARAMS['replenish_below']:
            points=self.body.points.copy();corners=self.body.corners.copy();independent=self.body.last_geometry;extent=self.body.initial_extent
            self.body.seed(image,strong,t,q)
            if self.body.points is None or len(self.body.points)<=len(points):self.body.points=points
            else:a.update(feature_reseeded=True,points_after_reseed=len(self.body.points))
            self.body.corners=corners;self.body.last_geometry=independent;self.body.initial_extent=extent
        return q,a


def factory(w,h,method):
    if method in ['similarity_long','translation_long']:return strict.BodyTracker(w,h,method)
    return Replenished(w,h,method)
