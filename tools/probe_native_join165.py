"""Exact native bytes vs structured concatenation; saved packet only."""
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_decode import join_native
if __name__=='__main__':
    with np.load(ROOT.parent/'output/rotor_pipeline_20261101/cache/side/0000.npz') as z:e=z['events'].copy()
    pieces=[e[:300000],e[300000:]];outputs={};records=[]
    for native in [False,True,True,False,False,True]:
        tick=time.perf_counter()
        outputs[native]=join_native(pieces,e.dtype) if native else np.concatenate(pieces,dtype=e.dtype)
        records.append(dict(native=native,ms=(time.perf_counter()-tick)*1000))
    assert outputs[True].dtype==outputs[False].dtype
    assert all(np.array_equal(outputs[True][k],outputs[False][k]) for k in e.dtype.names)
    assert np.array_equal(join_native([e[::2]],e.dtype)['t'],e['t'][::2])
    result=dict(events=len(e),native_itemsize=e.dtype.itemsize,all_fields_equal=True,noncontiguous_copy_equal=True,
        records=records,median_ms={str(v):float(np.median([r['ms'] for r in records if r['native']==v])) for v in [False,True]})
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
