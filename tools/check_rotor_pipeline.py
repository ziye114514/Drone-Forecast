"""Useful invariants on real cached events: future exclusion and native contours."""
from pathlib import Path
import importlib.util,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_support_front import rotor_support,density_mask,components
from rotor_only_front import native_reconstruction


def main(output):
    out=Path(output);checks={}
    # Compare optimized spatial primitives to the frozen first implementation.
    frozen=out/'clear_v1/source_snapshot/rotor_only_front.py'
    spec=importlib.util.spec_from_file_location('frozen_native_rotor_front',frozen)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    comparisons=[]
    for case in ['clear_v1','side_v1','empty_v1']:
        for path in (out/case/'images').glob('*_evidence.npz'):
            with np.load(path) as z:
                frame=dict(on=z['on'],off=z['off'],period_mask=z['period'])
                a,_=old.native_reconstruction(frame);b,_=native_reconstruction(frame)
                assert np.array_equal(a,b)
                da,_=density_mask(z['on']+z['off']);db,_=density_mask(z['on']+z['off'],fast_spatial=True)
                assert np.array_equal(da,db)
                ca,la,fa=components(z['period']);cb,lb,fb=components(z['period'],fast_spatial=True)
                assert np.array_equal(ca,cb)
                assert sorted(c['bbox'] for c in fa)==sorted(c['bbox'] for c in fb)
                comparisons.append(str(path.relative_to(out)))
    checks['native_reconstruction_and_spatial_masks_same_saved_inputs']=comparisons
    # Actually exercise timestamp rejection with a high-count future duplicate.
    with np.load(out/'cache/clear/0008.npz') as z:
        events=z['events'];t=float(z['t'])
    future=events.copy();future['t']+=100000
    a=rotor_support(events,t,fast_spatial=True)
    b=rotor_support(np.concatenate([events,future]),t,fast_spatial=True)
    assert np.array_equal(a['period_mask'],b['period_mask'])
    assert np.array_equal(a['frequency_hz'],b['frequency_hz'])
    assert np.array_equal(a['on'],b['on']) and np.array_equal(a['off'],b['off'])
    checks['future_duplicate_events_do_not_change_current_result']=True
    # Do not claim CUDA bitwise equivalence: inspect accepted masks and centers.
    r1=[json.loads(s) for s in (out/'clear_v1/predictions.jsonl').read_text().splitlines()]
    r2=[json.loads(s) for s in (out/'clear_bank/predictions.jsonl').read_text().splitlines()]
    checks['same_clear_window_count']=len(r1)==len(r2)
    checks['CPU_vs_CUDA_max_current_center_delta_px']=max(float(np.linalg.norm(
        np.array(a['body']['center'])-b['body']['center'])) for a,b in zip(r1,r2)
        if a['body'].get('detected') and b['body'].get('detected'))
    checks['CPU_vs_CUDA_current_detection_flags_equal']=all(a['body']['detected']==b['body']['detected'] for a,b in zip(r1,r2))
    (out/'mechanism_checks.json').write_text(json.dumps(checks,indent=2),'utf-8')
    print(json.dumps(dict(real_frames_compared=len(comparisons),future_exclusion=True,
        max_center_delta_px=checks['CPU_vs_CUDA_max_current_center_delta_px'])))


if __name__=='__main__':main(sys.argv[1])
