"""Prefer original current full geometry; use occupancy only when it qualifies.

Selection uses current conics, not future errors. This is a conservative
measurement fallback, not a learned gate or a claim of novel fusion.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage122')]
from worker122 import PartialModel,geom,CenterBank,NAMES
PARAMS=dict(selection='Use count_global whenever its unchanged104 full-four check is valid. Otherwise use occupancy_roi only if its same104 check is valid. Otherwise retain count_global actual observations for original partial/hold logic.',
    primary='Does complementary support close initialization gaps without harming successful count measurements? No threshold/fit/lock change, no future label in selection.',
    missing='No fake or transported fourth observation, no fixed-k cluster, no count->thrust conversion.',
    model='Existing119 affine3 and107 fixed-bias bank, same as124 controls. No adaptive122 selection claim.',default_promoted=False)
class Hybrid:
    def __init__(self):self.model=PartialModel('affine3');self.bank=CenterBank()
    def step(self,t,body,axis,count,occupancy,count_measurement=None,occupancy_measurement=None):
        cm=count_measurement if count_measurement is not None else geom.rigid104.measure_four(count,body)
        om=occupancy_measurement if occupancy_measurement is not None else geom.rigid104.measure_four(occupancy,body)
        choose='occupancy_roi' if not cm['valid'] and om['valid'] else 'count_global'
        rotors,m=(occupancy,om) if choose=='occupancy_roi' else (count,cm)
        _,d=self.model.step(t,body,rotors,axis,m);q=self.bank.step(t,body,d,axis)
        return dict(chosen=choose,count_qualified=cm['valid'],occupancy_qualified=om['valid'],input_rotors=rotors,center=d,forecast=q)
