import cv2,numpy as np
from body35 import BodyTracker,PARAMS


def run():
    rng=np.random.default_rng(35);patch=np.uint8(rng.integers(20,240,(60,60)));patch=cv2.GaussianBlur(patch,(3,3),.5)
    for method in PARAMS['variants']:
        b=BodyTracker(240,180,method);positions=[]
        for i in range(8):
            image=np.zeros((180,240),np.uint8);x=60+3*i;y=60-i;image[y:y+60,x:x+60]=patch
            o=dict(t=.01*(i+1),detected=i<3,rotors=[],center=[x+30,y+30],bbox=[x,y,x+60,y+60])
            q,a=b.step(image,image,image>0,o['t'],o,1000)
            assert q['detected']
            assert np.linalg.norm(np.asarray(q['center'])-[x+30,y+30])<1.5
            if i<3:assert q==o
            else:assert q['rotors']==[] and q['source']=='body_flow'
        image=np.zeros((180,240),np.uint8);o=dict(t=.09,detected=False,rotors=[])
        q,a=b.step(image,image,image>0,.09,o,0);assert not q['detected'] and a['reason']=='no_current_events'
        q,a=b.step(image,image,image>0,.10,o,100);assert not q['detected']
    return dict(known_translation_under_1_5px=True,geometry_exactly_preserved=True,current_feature_measurement_has_no_invented_rotors=True,zero_event_abstention=True,no_reseeding_from_flow=True)
