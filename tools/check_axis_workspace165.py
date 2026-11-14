"""Check input/order equality before enabling the reused side-axis workspace."""
from pathlib import Path
import sys,json,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from native_axis_workspace import NativeAxisWorkspace


if __name__=='__main__':
    workspace=NativeAxisWorkspace();rng=np.random.default_rng(165)
    values=np.arange(4096,dtype=np.float32).reshape(64,64)
    np.testing.assert_array_equal(workspace.render(values),np.uint8(np.clip(np.log1p(values)/4,0,1)*255))
    folder=ROOT.parent/'output/native_plane165_20261104';records=[]
    for path in sorted((folder/'native_maps').glob('*.npz')):
        with np.load(path) as a:density=np.minimum(a['on'],a['off'])
        for threshold in [1.4,3.,6.]:
            mask=density>=threshold;y,x=np.nonzero(mask);expected=np.column_stack((x,y)).astype(float)
            if len(expected)>5000:expected=expected[::int(np.ceil(len(expected)/5000))]
            np.testing.assert_array_equal(workspace.points(mask),expected)
        records.append(dict(input=str(path),point_sampling_exact=True))
    result=dict(integer_log_count_input_exact=True,native_points_exact=records)
    (folder/'axis_workspace_checks.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
