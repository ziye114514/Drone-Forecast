"""Read-only RAW rechunk integrity against three previously saved packets."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches


if __name__=='__main__':
    base=ROOT.parent/'output/rotor_pipeline_20261101/cache/side';records=[]
    for i,(t,e) in enumerate(native_batches(Path('E:/dataset/train/6/media/event.raw'),110.,.1,
            False,33333,reader_backend='iterator')):
        if i>=3:break
        with np.load(base/f'{i:04d}.npz') as z:expected=z['events'];old_t=float(z['t'])
        assert t==old_t and len(e)==len(expected)
        assert e.dtype.itemsize==16 and e.dtype.fields['t'][1]==8
        assert all(np.array_equal(e[k],expected[k]) for k in ['x','y','p','t'])
        digest=hashlib.sha256()
        for k in ['x','y','p','t']:digest.update(e[k].tobytes())
        records.append(dict(index=i,t=t,n=len(e),canonical_fields_sha256=digest.hexdigest(),
            every_native_field_equal=True,sdk16byte_layout=True))
    assert len(records)==3
    result=dict(records=records,no_RAW_written=True,no_SDK_code_modified=True,
        no_event_sampling_or_timestamp_shift=True)
    (ROOT.parent/'output/side_temporal164_20261101/rechunk_integrity.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result))
