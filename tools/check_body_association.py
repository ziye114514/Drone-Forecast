from pathlib import Path
import os,sys
os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
import numpy as np
from body_association import BodyAssociation

def box(x,score):return np.array([[x,100.,x+80.,140.,score,0.]],np.float32)
q=BodyAssociation(25.)
a,d=q.update(box(100.,.9),0.);assert len(a)==1;identity=a[0]['body_track_id']
for t,x in [(.04,102.),(.08,104.)]:
    a,d=q.update(box(x,.15),t)
    assert len(a)==1 and a[0]['body_track_id']==identity and a[0]['low_confidence']
    assert a[0]['bbox']==box(x,.15)[0,:4].tolist()
a,d=q.update(np.empty((0,6),np.float32),.12);assert not a
fresh=BodyAssociation(25.)
a,d=fresh.update(box(100.,.15),0.);assert not a
print('PASS: installed ByteTrack high initialization/low continuation; raw current boxes; no coasting or low-only initialization')
