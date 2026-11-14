"""Compare bridge batches and ROI against the actual SDK, not visual guesses."""
from pathlib import Path
import json
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import RoiFilterAlgorithm
out=Path(__file__).resolve().parents[2]/'output/paper_reproduction_v030/eeppr_benchmark'
results=[]
for name in ('velcro_front','velcro_side'):
    meta=json.loads((out/f'{name}_input.json').read_text());events=np.load(out/f'{name}.npy',mmap_mode='r');times=events['t']
    bounds=np.searchsorted(np.ascontiguousarray(times),np.arange(0,1000001,100))
    equal=True;count=0;chunks=0
    for i,actual in enumerate(EventsIterator(meta['raw'],delta_t=100,max_duration=1000000)):
        lo,hi=bounds[i:i+2];expected=events[lo:hi]
        equal &= np.array_equal(actual,expected);count+=len(actual);chunks+=1
    roi=RoiFilterAlgorithm(**meta['roi'],output_relative_coordinates=False);buf=roi.get_empty_output_buffer()
    # All border pixels are checked using actual decoded sensor coordinates.
    small=np.array(events[:1000000],copy=True);roi.process_events(small,buf);actual=buf.numpy()
    r=meta['roi'];expected=small[(small['x']>=r['x0'])&(small['x']<=r['x1'])&(small['y']>=r['y0'])&(small['y']<=r['y1'])]
    results.append(dict(sequence=name,batches_compared=chunks,events_compared=count,iterator_exactly_equal=bool(equal),
                        roi_exactly_equal=bool(np.array_equal(actual,expected)),roi_test_events=len(small)))
(out/'sdk_bridge_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(results)
if not all(q['iterator_exactly_equal'] and q['roi_exactly_equal'] for q in results):raise AssertionError(results)
