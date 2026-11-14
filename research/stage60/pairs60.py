"""Standard opposite-edge pairing; line hypotheses never certify rotor identity."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

PARAMS=dict(angle_max_deg=10.,overlap_min=.65,endpoint_disagreement_fraction=.25,
            width_min_px=1.,width_max_length_ratio=.30,
            cross_threshold_center_fraction=.10,cross_threshold_center_min_px=3.,
            cross_threshold_length_ratio=[.65,1.5],cross_threshold_angle_deg=10.,
            confirmation_ticks=3,track_gap_s=.025,track_center_fraction=.25,
            track_center_min_px=10.,track_angle_max_deg=20.,
            roi_max_length_fraction=.6,
            interpretation='Standard stripe-edge geometry, not novel event detector. Parallel arms/body stripes remain possible; stable is not true motor identity.')

def angle_distance(a,b):return float(abs((a-b+90)%180-90))

def pair_segments(lines):
    lines=np.asarray(lines,float).reshape(-1,2,2)
    proposals=[];cos=np.cos(np.deg2rad(PARAMS['angle_max_deg']))
    for i in range(len(lines)):
        pi=lines[i];di=pi[1]-pi[0];li=np.linalg.norm(di)
        if li<12:continue
        ui=di/li
        for j in range(i+1,len(lines)):
            pj=lines[j];dj=pj[1]-pj[0];lj=np.linalg.norm(dj)
            if lj<12 or ui@(dj/lj)>-cos:continue
            u=ui-dj/lj;u/=np.linalg.norm(u)
            if u[0]<0 or (u[0]==0 and u[1]<0):u=-u
            n=np.array([-u[1],u[0]])
            ti=np.sort(pi@u);tj=np.sort(pj@u)
            overlap=max(0.,min(ti[1],tj[1])-max(ti[0],tj[0]));extent=max(li,lj)
            overlap_fraction=overlap/extent
            mismatch=float(max(abs(ti[0]-tj[0]),abs(ti[1]-tj[1])))
            if overlap_fraction<PARAMS['overlap_min'] or mismatch>PARAMS['endpoint_disagreement_fraction']*extent+2:continue
            ni=float(np.mean(pi@n));nj=float(np.mean(pj@n));width=abs(ni-nj)
            lo,hi=(ti+tj)/2;length=float(hi-lo)
            if not PARAMS['width_min_px']<=width<=length*PARAMS['width_max_length_ratio']:continue
            center=u*((lo+hi)/2)+n*((ni+nj)/2)
            score=overlap_fraction*min(li,lj)/max(li,lj)*np.exp(-mismatch/max(extent,1))
            proposals.append(dict(edge_indices=[i,j],center=center.tolist(),major=length,minor=width,
                angle=float(np.degrees(np.arctan2(u[1],u[0]))%180),
                endpoints=(u[None,:]*np.array([lo,hi])[:,None]+n*((ni+nj)/2)).tolist(),
                ratio=width/length,overlap_fraction=overlap_fraction,endpoint_disagreement_px=mismatch,
                shape_quality=float(score),verified_rotor_identity=False,complete_endpoints_verified=False))
    # Each detected edge can support at most one stripe in this deterministic hypothesis.
    used=set();found=[]
    for p in sorted(proposals,key=lambda p:(-p['shape_quality'],p['edge_indices'])):
        if any(i in used for i in p['edge_indices']):continue
        used.update(p['edge_indices']);found.append(p)
    return found

def cross_threshold(base,strong):
    result=[dict(p,cross_threshold_stable=False,matched_strong_index=None) for p in base]
    if not base or not strong:return result
    costs=np.full((len(base),len(strong)),1e6)
    for i,a in enumerate(base):
        for j,b in enumerate(strong):
            distance=float(np.linalg.norm(np.array(a['center'])-b['center']));ratio=b['major']/a['major']
            if (distance<=max(PARAMS['cross_threshold_center_min_px'],a['major']*PARAMS['cross_threshold_center_fraction']) and
                PARAMS['cross_threshold_length_ratio'][0]<=ratio<=PARAMS['cross_threshold_length_ratio'][1] and
                angle_distance(a['angle'],b['angle'])<=PARAMS['cross_threshold_angle_deg']):
                costs[i,j]=distance/max(a['major'],1)+abs(np.log(ratio))
    rr,cc=linear_sum_assignment(costs)
    for i,j in zip(rr,cc):
        if costs[i,j]<1e6:
            result[i].update(cross_threshold_stable=True,matched_strong_index=int(j),
                threshold_center_shift_px=float(np.linalg.norm(np.array(base[i]['center'])-strong[j]['center'])))
    return result

class Confirm:
    def __init__(self):self.tracks=[];self.next_id=0
    def step(self,t,pairs,roi):
        current=[dict(p) for p in pairs if p['cross_threshold_stable']]
        old=[p for p in self.tracks if 0<t-p['t']<=PARAMS['track_gap_s']]
        costs=np.full((len(current),len(old)),1e6)
        for i,a in enumerate(current):
            for j,b in enumerate(old):
                distance=float(np.linalg.norm(np.array(a['center'])-b['center']));ratio=a['major']/b['major']
                if distance<=max(PARAMS['track_center_min_px'],a['major']*PARAMS['track_center_fraction']) and .6<=ratio<=1.7 and angle_distance(a['angle'],b['angle'])<=PARAMS['track_angle_max_deg']:
                    costs[i,j]=distance/max(a['major'],1)+abs(np.log(ratio))
        matched={}
        if current and old:
            rr,cc=linear_sum_assignment(costs)
            matched={int(i):int(j) for i,j in zip(rr,cc) if costs[i,j]<1e6}
        for i,p in enumerate(current):
            if i in matched:
                q=old[matched[i]];p.update(candidate_track_id=q['candidate_track_id'],streak=q['streak']+1)
            else:p.update(candidate_track_id=self.next_id,streak=1);self.next_id+=1
            p['t']=t
            eligible=roi is not None and p['major']<=PARAMS['roi_max_length_fraction']*max(roi[2]-roi[0],roi[3]-roi[1])
            p['confirmed_candidate']=bool(eligible and p['streak']>=PARAMS['confirmation_ticks'])
        self.tracks=current
        return current

def checks():
    # Known stripe edges are gradient-directed oppositely; permutations invariant geometrically.
    good=np.array([[[0.,-2],[100,-2]],[[100.,2],[0,2]]])
    p=pair_segments(good);assert len(p)==1 and np.allclose(p[0]['center'],[50,0])
    assert not pair_segments(np.array([good[0],good[1,::-1]]))
    assert not pair_segments(np.array([good[0],[[100,40],[0,40]]]))
    assert not pair_segments(np.array([good[0],[[100,2],[70,2]]]))
    stable=cross_threshold(p,p);assert stable[0]['cross_threshold_stable']
    moving=[dict(p[0],center=[70.,0.])];assert not cross_threshold(p,moving)[0]['cross_threshold_stable']
    c=Confirm();a=c.step(.01,stable,[-10,-10,110,10]);b=c.step(.02,stable,[-10,-10,110,10]);d=c.step(.03,stable,[-10,-10,210,10])
    assert not a[0]['confirmed_candidate'] and not b[0]['confirmed_candidate'] and d[0]['confirmed_candidate']
    assert c.step(.06,stable,[-10,-10,210,10])[0]['streak']==1
    # A bright arm stripe has the same geometry: pairing cannot resolve this semantic ambiguity.
    assert len(pair_segments(good))==1 and not p[0]['verified_rotor_identity']
    return dict(known_stripe_center_exact=True,same_gradient_rejected=True,wide_pair_rejected=True,
        one_sided_fragment_rejected=True,unstable_threshold_center_rejected=True,
        causal_confirmation_and_expiry=True,body_stripe_remains_explicit_ambiguity=True)
