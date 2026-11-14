import numpy as np
from occlusion25 import OracleMask


def run():
    m=OracleMask();rotors=[dict(cx=x,cy=y,major=10.,minor=5.,angle=0.) for x,y in [(40,40),(80,40),(80,80),(40,80)]]
    m.update(dict(detected=True,rotors=rotors))
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);e=np.zeros(5,dtype)
    e['x']=[40,80,80,40,120];e['y']=[40,40,80,80,120]
    adjacent,ok=m.erase(e,'adjacent');assert ok and len(adjacent)==3
    opposite,ok=m.erase(e,'opposite');assert ok and len(opposite)==3
    assert adjacent['x'].tolist()!=opposite['x'].tolist() or adjacent['y'].tolist()!=opposite['y'].tolist()
    assert len(e)==5
    empty=OracleMask();q,ok=empty.erase(e,'opposite');assert not ok and len(q)==5
    return dict(two_slots_erased_with_pattern_distinction=True,reference_events_not_modified=True,unavailable_mask_declared=True,
        scope='Mask-generator invariants; no real-occlusion or motor-ID accuracy claim.')


if __name__=='__main__':print(run())
