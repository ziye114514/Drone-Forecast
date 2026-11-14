"""Bounded output check for the actual optimized entry; no new tuning."""
from pathlib import Path
import json,sys
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;OUT=WORK/'output/stage137_fast_geometry_20261024';OLD=WORK/'output/stage135_optimized_forecast_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def check(name):
    rr=rows(OUT/name/'predictions.jsonl');completion=load(OUT/name/'completion.json');assert completion['sources_unchanged'] and not completion['labels_read'] and not completion['output_caches_read'];maxforecast=0.;maxcoef=0.;maxcenter=0.;front_exact=True
    refname='natural6_fast' if name in ['natural6','legacy100'] else name
    if name=='side50':
        bb=rows(WORK/'output/stage124_occupancy_support_20261023/continuous6/records.jsonl');cc=load(OLD/'continuous6.json')
        ref=[dict(body=b['body'],t=b['t'],coefficients=c['output']['coefficients']) for b,c in zip(bb,cc)]
    else:ref=rows(OLD/'raw'/refname/'predictions.jsonl')
    assert len(rr)==completion['frames'];assert len(rr)<=len(ref)
    for a,b in zip(rr,ref):
        assert a['t']==b['t'] and a['body']==b['body'];assert a['max_event_t'] is None or a['max_event_t']<round(a['t']*1e6)
        if name!='side50':
            assert a['frame_sha256']==b['frame_sha256'] and a['valid']==b['valid'];assert a['geometry']==b['geometry'];assert a['boundary']==b['boundary']
            for source in ['count','occupancy']:
                ad=a['models'][source];bd=b['models'][source]
                for key in ['current','structure_ready','initialized_now']:assert ad.get(key)==bd.get(key),(a['t'],source,key)
                for key in ['valid','reason']:assert ad['measurement'].get(key)==bd['measurement'].get(key),(a['t'],source,key)
                if ad.get('current'):maxcenter=max(maxcenter,float(np.max(abs(np.array(ad['model_geometric_center'])-bd['model_geometric_center']))))
        for method in ['optimized_geometry','matched_body']:
            ac=a['coefficients'][method];bc=b['coefficients'][method];assert (ac is None)==(bc is None)
            if ac is not None:
                for key in ['position','velocity','acceleration']:maxcoef=max(maxcoef,float(np.max(abs(np.array(ac[key])-bc[key]))))
                for h in [.1,.2,.4]:
                    ap=np.array(ac['position'])+h*np.array(ac['velocity'])+.5*h*h*np.array(ac['acceleration']);bp=np.array(bc['position'])+h*np.array(bc['velocity'])+.5*h*h*np.array(bc['acceleration']);maxforecast=max(maxforecast,float(np.max(abs(ap-bp))))
    assert maxforecast<1e-4 and maxcoef<1e-3 and maxcenter<1e-5,(maxforecast,maxcoef,maxcenter)
    if name=='negative5':assert all(not r['valid'] for r in rr)
    decoded=0
    if name=='natural6':
        cap=cv2.VideoCapture(str(OUT/name/'predictions.mp4'))
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
        cap.release();assert decoded==len(rr)==400
    result=dict(case=name,frames=len(rr),body_exact=True,current_geometry_decisions_exact=name!='side50',frontend_observations_exact=name!='side50',maximum_center_difference_px=maxcenter,maximum_coefficient_difference=maxcoef,maximum_forecast_difference_px=maxforecast,event_cutoff_passed=True,decoded_video_frames=decoded,no_output_cache_or_label_inputs=True,negative_no_valid_predictions=name=='negative5')
    (OUT/name/'checks137.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':check(sys.argv[1])
