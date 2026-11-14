import numpy as np,cv2
from body36 import Replenished


def run():
    rng=np.random.default_rng(36);patch=cv2.GaussianBlur(np.uint8(rng.integers(10,240,(60,60))),(3,3),.5)
    for method in ['support3_similarity','support3_translation','replenish3_similarity']:
        b=Replenished(240,180,method);seed_t=.01
        for i in range(5):
            image=np.zeros((180,240),np.uint8);x=50+2*i;y=50-i;image[y:y+60,x:x+60]=patch
            o=dict(t=.01*(i+1),detected=i==0,center=[x+30,y+30],bbox=[x,y,x+60,y+60],rotors=[])
            q,a=b.step(image,image,image>0,o['t'],o,1000);assert q['detected'] and np.linalg.norm(np.asarray(q['center'])-[x+30,y+30])<1.5
            if i==0:b.body.points=b.body.points[:8]
            assert b.body.last_geometry==seed_t
        if method=='replenish3_similarity':assert len(b.body.points)>8
        q,a=b.step(image,image,image>0,2.02,dict(t=2.02,detected=False,rotors=[]),1000);assert not q['detected'] and a['reason']=='age_expired'
    return dict(known_translation=True,reseed_does_not_reset_independent_age=True,reseed_preserves_oriented_box_and_extent=True,age_expiry_even_after_reseed=True)
