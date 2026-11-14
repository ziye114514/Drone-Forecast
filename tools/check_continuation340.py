"""Current-observation and identity tests with true translated image patches.

Strict-pixel qualification is an INPUT here; its existing native tests are
unchanged. These checks test continuation without pretending density is RPM.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,numpy as np,cv2
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
from periodic_track_continuation import PeriodicTrackContinuation
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from curvature_visibility import seeded_outline

def frame(i,strict_count=2,foreign=False):
    rng=np.random.default_rng(340);yy,xx=np.indices((720,1280));mask=((xx-(300+4*i))/55)**2+((yy-250)/18)**2<=1
    template=(20+rng.integers(0,20,(100,140))).astype(np.float32);on=np.zeros((720,1280),np.float32);on[200:300,230+4*i:370+4*i]=template;on*=mask
    off=on.copy();strict=np.zeros((720,1280),np.uint8)
    if strict_count:
        x=900 if foreign else 300+4*i
        strict[250,x:x+strict_count]=255
    mask=seeded_outline(on,off,mask)
    return dict(on=on,off=off,native_precomputed=(mask,dict(regional_seed_groups=0,regional_seed_pixels=0)),period_mask=strict,
        counts=dict(period_pixels=strict_count),density=mask,density_local=np.zeros_like(on),frequency_hz=np.zeros_like(on)),mask

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='controls341_20261109');args=parser.parse_args()
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False)
    configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'));cv2.setNumThreads(2)
    helper=PeriodicTrackContinuation();first,mask=frame(0,strict_count=12)
    front=SimpleNamespace(previous=(first['on'],first['off']),previous_mask=mask,last_t=1.,last_velocity=np.array([120.,0.]),
        fast_geometry=True,subpixel_motion=True,coarse_motion=False,border_motion=True,centered_motion=False)
    helper.prepare(first,front,1.);helper.observe(1.,dict(detected=True,center_provisional=False));truth=[]
    for i in range(1,8):
        t=1+i/30;current,mask=frame(i);before=current['frequency_hz'].copy();info=helper.prepare(current,front,t)
        assert info['used'],(i,info);flow=current['verified_native_displacement']['estimate'];truth.append([flow['dx'],flow['dy']]);assert abs(flow['dx']-4)<.3 and abs(flow['dy'])<.3,flow
        assert np.array_equal(before,current['frequency_hz']);helper.observe(t,dict(detected=True,center_provisional=False))
        front.previous=(current['on'],current['off']);front.previous_mask=current['native_precomputed'][0];front.last_t=t
    assert helper.last_period_t-1.>.2
    no_period,_=frame(8,strict_count=0);assert not helper.prepare(no_period,front,front.last_t+1/30)['used']
    unrelated,_=frame(8,foreign=True);foreign=helper.prepare(unrelated,front,front.last_t+1/30);assert not foreign['used'] and foreign['reason']=='current_period_pixels_not_on_matched_target',foreign
    stale,_=frame(8);assert not helper.prepare(stale,front,front.last_t+.13)['used']
    cold=PeriodicTrackContinuation();candidate,_=frame(8);assert not cold.prepare(candidate,front,front.last_t+1/30)['used']
    result=dict(seven_consecutive_current_strict_confirmations_beyond_original_hold_time=True,
        actual_translated_patch_displacements=truth,current_density_without_period_rejected=True,
        foreign_period_pixels_not_on_target_rejected=True,expired_identity_and_weak_cold_start_rejected=True,
        frequency_map_not_modified=True,NN_used=False,GPU_used=False,camera_false_alarm_calibrated=False)
    configure_native(None);(out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result)
