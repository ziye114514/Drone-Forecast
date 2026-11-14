import numpy as np
from gating57 import CandidateFrontend,RecoveryFrontend

def run():
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);events=np.zeros(3,dtype);events['x']=[10,50,90];events['y']=50
    front=CandidateFrontend(100,100,'neighbor10');hint=dict(t=1.,detected=True,bbox=[40,40,60,60],center=[50,50])
    original=events.copy();diagnostic=front.set_hint(events,hint,1.01,'body')
    assert diagnostic['roi']==[35,35,65,65] and diagnostic['geometry_events']==len(events) and diagnostic['mode']=='body_candidate_roi'
    huge=dict(center=np.array([50.,50.]),major=1000.,minor=500.);outside=dict(center=np.array([10.,50.]),major=10.)
    kept=front.filter_candidates([huge,outside]);assert len(kept)==1 and kept[0] is huge
    np.testing.assert_array_equal(events,original)
    for mode,h,t in [('off',hint,1.01),('body',None,1.01),('body',hint,1.17)]:
        front.set_hint(events,h,t,mode);found=[huge,outside];assert front.filter_candidates(found) is found
    try:front.set_hint(events,hint,.99,'body')
    except AssertionError:pass
    else:raise AssertionError('future body hint accepted')
    # Execute the complete inherited frontend, rather than only checking a mask.
    import cv2
    mask=np.zeros((240,320),np.uint8)
    for center in [(90,70),(230,70),(90,170),(230,170)]:cv2.ellipse(mask,center,(22,7),12,0,360,1,-1)
    yy,xx=np.nonzero(mask);e=np.zeros(len(xx)*6,dtype);e['x']=np.repeat(xx,6);e['y']=np.repeat(yy,6);e['p']=np.tile([0,1,0,1,0,1],len(xx))
    baseline=RecoveryFrontend(320,240,'neighbor10');off=CandidateFrontend(320,240,'neighbor10');wide=CandidateFrontend(320,240,'neighbor10')
    full=dict(t=0.,detected=True,bbox=[0,0,320,240],center=[160,120])
    for tick in range(1,6):
        e['t']=tick*10000-100;t=tick*.01;expected,_=baseline.detect(e,t)
        off.set_hint(e,None,t,'off');actual,_=off.detect(e,t);assert expected==actual
        wide.set_hint(e,full,t,'body');actual,_=wide.detect(e,t);assert expected==actual
    import run_candidate_forecast as entry
    from rotor_worker57 import worker_source
    compile(worker_source(),'<57 worker>','exec')
    for mode in ['off','body']:
        namespace=entry.build(mode,32)
        try:assert namespace['METHODS']==['delayed_feedback'] and len(namespace['MODELS'])==15
        finally:namespace.clear()
    return dict(candidate_dictionary_identity_and_complete_shape_preserved=True,no_current_events_modified_or_removed=True,
        disabled_missing_expired_hint_identity=True,future_hint_rejected=True,full_inherited_frontend_off_and_wide_ROI_exact=True,
        complete15_models=True,original_full_event_frequency_history_unchanged=True)
