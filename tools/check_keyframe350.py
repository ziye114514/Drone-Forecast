"""Known translation and correlated-history tests, without trajectory labels."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
from check_continuation340 import frame
from keyframe_rotor_motion import KeyframeRotorMotion
from keyframe_force_observer import KeyframeAdaptiveObserver
from adaptive_response_observer import AdaptiveResponseObserver


if __name__=='__main__':
    out=ROOT.parent/'output/controls350_20261109';out.mkdir(exist_ok=False);cv2.setNumThreads(2)
    helper=KeyframeRotorMotion();measurements=[]
    for i in range(10):
        f,mask=frame(i);t=1+i/30
        f.update(native_mask=mask,body=dict(detected=True,center_provisional=False,center=[300+4*i,250]),motion={},periodic_track_continuation=dict(used=True))
        helper.observe(f,t,t-1/30 if i else None)
        p=f['motion'].get('keyframe_rotor_observation') or {}
        if p and not p['create']:
            measurements.append(p['displacement_xy']);assert np.linalg.norm(np.asarray(p['displacement_xy'])-[4*i,0])<.6,p
    assert len(measurements)>=7,len(measurements)
    # A stochastic clone on the common translation gauge must not reduce
    # absolute uncertainty. Check the exact unobservable constant-shift mode.
    reference=KeyframeAdaptiveObserver();control=AdaptiveResponseObserver();shifted=KeyframeAdaptiveObserver()
    maximum=0.;gate_count=0
    for i in range(20):
        t=2+i/30;z=np.array([100+4*i,200.])
        packet=dict(create=True,source_t=t) if i==5 else dict(create=False,source_t=2+5/30,displacement_xy=[4*(i-5),0],covariance_proxy=(np.eye(2)*4).tolist()) if i>6 else None
        motion=dict(estimate=dict(dx=4.,dy=0.,score=1.),dt_s=1/30,current_observation=True)
        if packet:motion['keyframe_rotor_observation']=packet
        values=[]
        for m,offset in ((reference,np.zeros(2)),(shifted,np.array([47.,-19.]))):
            body=dict(detected=True,center=(z+offset).tolist(),measurement_source='rotor_region_initialization' if i==0 else 'filtered_native_translation')
            c,v,f,s=m.step(t,body,np.zeros(2),motion=motion,structure={},allow_gain_learning=False)
            assert np.linalg.eigvalsh((m.P+m.P.T)/2).min()>-1e-6
            values.append(c);gate_count+=bool(s['direct_keyframe_rotor_displacement_used'])
        maximum=max(maximum,float(np.max(abs(np.asarray(values[1]['position'])-values[0]['position']-[47.,-19.]))))
        assert np.allclose(values[0]['velocity'],values[1]['velocity'],atol=1e-8)
    assert maximum<1e-8 and gate_count>=20,(maximum,gate_count)
    # Without direct factors the augmented model reproduces old force output.
    plain=KeyframeAdaptiveObserver();max_plain=0.
    for i in range(12):
        body=dict(detected=True,center=[500+2*i,150+np.sin(i/5)],measurement_source='rotor_region_initialization' if i==0 else 'filtered_native_translation')
        motion=dict(estimate=dict(dx=2.,dy=float(np.sin(i/5)-np.sin((i-1)/5)),score=1.),dt_s=1/30,current_observation=True)
        aa=control.step(4+i/30,body,np.zeros(2),motion=motion,structure={},allow_gain_learning=False)
        bb=plain.step(4+i/30,body,np.zeros(2),motion=motion,structure={},allow_gain_learning=False)
        for key in ('position','velocity','acceleration'):max_plain=max(max_plain,float(np.max(abs(np.asarray(aa[0][key])-bb[0][key]))))
    assert max_plain<1e-7,max_plain
    result=dict(known_direct_keyframe_translations=measurements,constant_image_translation_gauge_max_error=maximum,
        no_direct_factor_old_state_parity=max_plain,direct_factors_used=gate_count,stochastic_covariance_PSD=True,
        current_position_is_not_a_new_absolute_anchor=True,NN_used=False,GPU_used=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result)
