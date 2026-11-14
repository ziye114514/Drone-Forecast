"""Actual same-input output comparison after inference exits."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;OUT=WORK/'output/stage138_shared_objective_20261024';OLD=WORK/'output/stage137_fast_geometry_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def check(name):
    rr=rows(OUT/name/'predictions.jsonl');refname='natural6' if name=='fast100' else name;bb=rows(OLD/refname/'predictions.jsonl')[:len(rr)];assert len(rr)==len(bb)
    c=load(OUT/name/'completion.json');assert c['sources_unchanged'] and not c['labels_read'] and not c['output_caches_read'];difference=0.;center=0.
    for a,b in zip(rr,bb):
        for key in ['t','body','geometry','boundary','valid','frame_sha256']:assert a[key]==b[key],(a['t'],key)
        assert a['max_event_t'] is None or a['max_event_t']<round(a['t']*1e6)
        for source in ['count','occupancy']:
            x=a['models'][source];y=b['models'][source]
            for key in ['current','structure_ready','initialized_now']:assert x.get(key)==y.get(key),(a['t'],source,key)
            for key in ['valid','reason']:assert x['measurement'].get(key)==y['measurement'].get(key),(a['t'],source,key)
            if x.get('current'):center=max(center,float(np.max(abs(np.array(x['model_geometric_center'])-y['model_geometric_center']))))
        for method,ac in a['coefficients'].items():
            bc=b['coefficients'][method];assert (ac is None)==(bc is None)
            if ac:
                for key in ['position','velocity','acceleration']:difference=max(difference,float(np.max(abs(np.array(ac[key])-bc[key]))))
    assert difference<1e-3 and center<1e-5,(difference,center)
    if name=='negative5':assert all(not r['valid'] for r in rr)
    decoded=0
    if name=='natural6':
        cap=cv2.VideoCapture(str(OUT/name/'predictions.mp4'))
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
        cap.release();assert decoded==400
    result=dict(case=name,frames=len(rr),body_frontend_boundary_publication_exact=True,geometry_decisions_exact=True,max_coefficient_difference=difference,max_current_center_difference_px=center,negative_no_valid=name=='negative5',video_frames_decoded=decoded,no_label_or_output_cache_inputs=True,event_cutoffs=True)
    (OUT/name/'checks138.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    for name in ['fast100','natural6','side50','negative5']:check(name)
