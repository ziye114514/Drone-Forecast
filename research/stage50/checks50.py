import numpy as np
from layout50 import *

def observed(points):
    return dict(rotors=[dict(cx=float(p[0]),cy=float(p[1]),major=20.,minor=12.,angle=0.,ratio=.6,geometry_quality=.9,activity=100.,mode='ellipse') for p in points])
def body(t,c):return dict(t=t,detected=True,center=np.asarray(c).tolist(),bbox=np.r_[np.array(c)-150,np.array(c)+150].tolist())
def run():
    sensors={'replace_stale':ReplaceStale(PaperConfig()),'layout':LayoutSlots(PaperConfig())};errors={n:0 for n in sensors};first={};joint=0
    for j in range(80):
        t=(j+1)*.01;c=np.array([300.+(80 if j>=30 else 0.),250.]);angle=.3*np.sin(t*3);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        points=skeleton.Q@R.T*45+c;order=np.roll(np.arange(4),j%4);obs=observed(points[order])
        for name,s in sensors.items():
            q=s.step(obs,[],t,body(t,c)) if name=='layout' else s.step(obs,[],t)
            if j==0:first[name]={r['rotor_id']:int(order[k]) for k,r in enumerate(q['rotors'])}
            for r in q['rotors']:
                if r['observed_now']:
                    physical=int(np.argmin(np.linalg.norm(points-[r['cx'],r['cy']],axis=1)))
                    errors[name]+=r['rotor_id'] not in first[name] or first[name].get(r['rotor_id'])!=physical
            if name=='layout':joint+=s.audit['layout']['joint_association']
    assert errors['layout']==0 and joint>0,(errors,joint)
    assert sensors['layout'].next_id==4
    q=sensors['layout'].step(dict(rotors=[]),[],1.2,None);assert not q['rotors'] and sensors['layout'].A is None
    assert all('position' not in q.__dict__ for q in sensors['layout'].tracks)
    # No body evidence: exactly retain33 geometry and IDs, not an invented anchor.
    a=ReplaceStale(PaperConfig());b=LayoutSlots(PaperConfig())
    for j in range(30):
        t=j*.01;p=skeleton.Q*45+[300+j*2,250];o=observed(p)
        assert geometry_record(a.step(o,[],t))==geometry_record(b.step(o,[],t,None))
    return dict(controlled_motion_identity_errors=errors,joint_current_associations=joint,body_supported_layout_kept_four_ids=True,
        no_body_exact_to33=True,no_input_expires=True,temporary_position_hooks_restored=True,
        limits='Synthetic independent body measurement supplied explicitly; does not establish real UAV identity accuracy.')
