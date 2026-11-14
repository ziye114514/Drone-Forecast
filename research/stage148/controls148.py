"""Critical causal tracking, dropout, harmonic and overlap controls."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from temporal148 import TemporalBank,TemporalRates
from residual148 import ResidualExpert
from guided148 import QuadGuide,GuidedFrontend
import cv2

OUT=Path(__file__).resolve().parents[3]/'output/stage148_temporal_rotors_20261024'
def rotor(x,y):
    return dict(cx=float(x),cy=float(y),major=36.,minor=18.,ratio=.5,angle=0.,mode='ellipse',inferred_shape=False)
def body(x=100.):return dict(detected=True,center=[x,100.])
def votes(t,f,centers):
    points=[(int(x+dx),int(y+dy)) for x,y in centers for dx in (-5,-2,2,5) for dy in (-2,2)]
    q=np.zeros(len(points),dtype=[('x','u2'),('y','u2'),('t','i8'),('frequency','f8')]);q['x'],q['y']=np.asarray(points).T
    q['t']=round(t*1e6)-100;q['frequency']=f;return q
def main():
    OUT.mkdir(exist_ok=True);checks={};bank=TemporalBank();original=[rotor(50,50),rotor(150,50),rotor(150,150),rotor(50,150)]
    first=bank.step(.02,body(),original);ids={tuple(r['raw_center148']):r['visual_track_id'] for r in first}
    moved=[dict(r,cx=r['cx']+4) for r in original]
    second=bank.step(.04,body(104),moved[::-1])
    assert all(r['visual_track_id']==ids[(r['cx']-4,r['cy'])] for r in second)
    # No duplicate update when a geometry attempt falls back within one timestamp.
    again=bank.step(.04,body(104),moved[::-1]);assert [r['visual_track_id'] for r in again]==[r['visual_track_id'] for r in second]
    assert all(r['temporal_observations']==2 for r in again)
    checks['shuffled_order_and_body_translation_keep_ids']=True;checks['same_timestamp_fallback_not_double_counted']=True
    missing=bank.step(.06,body(104),moved[:3]);assert len(missing)==3 and bank.diagnostic['missing_tracks_emitted']==0
    checks['missing_track_never_emitted_as_current']=True
    rate=TemporalRates(bank);centers=[(r['cx'],r['cy']) for r in missing]
    q=votes(.06,300,centers);a=rate(q,missing,.06,[0,0],300);assert not any(r['valid'] for r in a)
    missing=bank.step(.08,body(104),moved[:3]);b=rate(votes(.08,300,centers),missing,.08,[0,0],300)
    assert all(r['valid'] and abs(r['frequency_hz']-300)<1 for r in b)
    empty=votes(.10,300,centers)[:0];c=rate(empty,missing,.10,[0,0],300);assert not any(r['valid'] for r in c)
    checks['two_current_supported_periods_lock_and_zero_current_rejects']=True
    for t in (.12,.14):
        missing=bank.step(t,body(104),moved[:3]);c=rate(votes(t,600,centers),missing,t,[0,0],300)
        assert not any(r['valid'] for r in c)
    missing=bank.step(.16,body(104),moved[:3]);c=rate(votes(.16,600,centers),missing,.16,[0,0],300)
    assert all(r['valid'] and abs(r['frequency_hz']-600)<1 for r in c)
    checks['half_double_transition_requires_three_current_frames']=True
    two=[rotor(100,100),rotor(100,100)];bb=TemporalBank();rr=TemporalRates(bb)
    for t in (.02,.04):
        rs=bb.step(t,body(),two);measured=rr(votes(t,300,[(100,100)]),rs,t,[0,0],300)
        assert not any(r['valid'] for r in measured)
    checks['fully_overlapping_votes_not_assigned_to_either_motor']=True
    expired=bank.step(.40,body(104),moved[:3]);assert all(r['temporal_observations']==1 for r in expired)
    checks['expired_tracks_reinitialize_without_stale_frequency']=True
    expert=ResidualExpert();geom=dict(axis_xy=[0.,-1.],confidence=.8);proxy=dict(valid=True,relative_f2_proxy=1.)
    baseline={str(h):[100.,100.] for h in (.1,.2,.4)}
    for k in range(1,101):
        t=k*.02;tilted=dict(geom,axis_xy=[.2*np.sin(t*30),-1.])
        out=expert.step(t,body(),tilted,proxy,baseline)
        assert all(np.allclose(point,[100,100]) for point in out['fused'].values())
    checks['plane_wobble_stationary_body_does_not_generate_translation']=True
    expert=ResidualExpert();biased={str(h):[106.,100.] for h in (.1,.2,.4)}
    for k in range(1,101):
        out=expert.step(k*.02,body(),geom,proxy,biased)
        if k<6:assert out['updates']==0
    assert out['bias_only_predictions']['0.4'][0]<102
    assert all(np.allclose(d['event_gain'],0) for d in out['diagnostics'].values())
    assert not any(s=='event_residual' for s in out['selections_by_horizon'].values())
    checks['bias_correction_not_misreported_as_event_gain']=True
    expert=ResidualExpert()
    for k in range(1,51):out=expert.step(k*.02,dict(detected=False),geom,proxy,biased)
    assert out['updates']==0
    checks['unobserved_targets_do_not_update_model']=True
    guide=QuadGuide();quad=[]
    for index,(x,y) in enumerate([(90,90),(190,90),(190,122),(90,122)]):
        r=rotor(x,y);r.update(major=80.,minor=40.,temporal_observations=3,visual_track_id=index);quad.append(r)
    guide.update(.02,body(),dict(state='four',observed_rotors=quad))
    mask=np.zeros((220,280),np.uint8)
    for y in (90,122):cv2.ellipse(mask,(94,y),(40,20),0,0,360,255,-1)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    recovered,info=guide.split(max(contours,key=cv2.contourArea),.04,body(104))
    assert recovered is not None,info
    assert all(a['coverage']>=.48 for a in recovered) and info['no_prior_only_output']
    one=np.zeros_like(mask);cv2.ellipse(one,(94,90),(40,20),0,0,360,255,-1)
    contours,_=cv2.findContours(one,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    assert guide.split(contours[0],.04,body(104))[0] is None
    assert guide.split(contours[0],.3,body(104))[0] is None
    checks['recent_quad_guides_two_current_arcs_not_missing_rotor']=True
    front=GuidedFrontend(280,220);front.guide=guide;front.guide_body=body(104)
    split_mask=np.zeros_like(mask);cv2.ellipse(split_mask,(94,90),(40,20),0,0,360,255,-1);split_mask[:,93:96]=0
    contours,_=cv2.findContours(split_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    def fragment(x):return dict(center=np.array([x,90.]),major=40.,minor=30.,activity=100.,shape_quality=.8)
    merged=front.merge_fragments([fragment(74),fragment(114)],contours,np.ones_like(mask,dtype=float),.04,1)
    assert len(merged)==1 and merged[0]['fragment_merged148'],merged
    negative=np.zeros_like(mask)
    for x in (74,114):cv2.ellipse(negative,(x,90),(13,10),0,0,360,255,-1)
    contours,_=cv2.findContours(negative,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    unchanged=front.merge_fragments([fragment(74),fragment(114)],contours,np.ones_like(mask,dtype=float),.04,1)
    assert len(unchanged)==2 and not any(q.get('fragment_merged148') for q in unchanged)
    checks['two_mask_fragments_one_disk_merge_but_unrelated_blobs_reject']=True
    (OUT/'controls148.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
