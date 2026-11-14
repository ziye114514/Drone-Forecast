"""Check actual140 RAW and replay clear138 observations through new predictor."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage140')]
from visible140 import Predictor,VisibleGate
OUT=WORK/'output/stage140_visible_fallback_20261024';OLD=WORK/'output/stage135_optimized_forecast_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def main():
    availability=[]
    for name,case in [('continuous6','continuous6'),('extension2','extension2'),('prefix_natural6','natural6'),('negative5_raw','negative5')]:
        actual=rows(OUT/name/'predictions.jsonl');cached=load(OUT/(case+'.json'));body=rows(WORK/'output/stage124_occupancy_support_20261023'/case/'records.jsonl');maximum=0.
        for a,c,b in zip(actual,cached,body):
            assert a['t']==c['t']==b['t'] and a['body']==b['body'];assert a['max_event_t'] is None or a['max_event_t']<round(a['t']*1e6)
            for method,value in c['output']['coefficients'].items():
                current=a['coefficients'][method];assert (value is None)==(current is None)
                if value:
                    for key in ['position','velocity','acceleration']:maximum=max(maximum,float(np.max(abs(np.array(current[key])-value[key]))))
            assert a['predictor']['dense_motion_fallback']==c['output']['dense_motion_fallback']
        assert maximum<1e-4;complete=load(OUT/name/'completion.json');assert complete['sources_unchanged'] and not complete['labels_read'] and not complete['output_caches_read'];assert len(actual)==complete['frames']
        decoded=0
        if name in ['continuous6','extension2']:
            cap=cv2.VideoCapture(str(OUT/name/'predictions.mp4'))
            while True:
                ok,_=cap.read()
                if not ok:break
                decoded+=1
            cap.release();assert decoded==len(actual)
        if name=='negative5_raw':assert all(not a['valid'] for a in actual)
        check=dict(case=case,frames=len(actual),maximum_cached_coefficient_difference=maximum,body_exact=True,gate_exact=True,event_cutoffs=True,decoded_frames=decoded,negative_no_valid=name=='negative5_raw');save(OUT/name/'checks140.json',check);availability.append(check)
    # Entire clear clip: current RAW138 observations are processed sequentially
    # by140; this is a prediction replay, not a new400-frame NN/RAW run.
    clear=rows(WORK/'output/stage138_shared_objective_20261024/natural6/predictions.jsonl');predictor=Predictor();maximum=0.;active=0
    for r in clear:
        out=predictor.step(r['t'],r['body'],r['models']);active+=out['dense_motion_fallback']['active']
        for method,value in out['coefficients'].items():
            current=r['coefficients'][method];assert (value is None)==(current is None)
            if value:
                for key in ['position','velocity','acceleration']:maximum=max(maximum,float(np.max(abs(np.array(current[key])-value[key]))))
    assert maximum<1e-4 and active==0;save(OUT/'clear_replay140.json',dict(frames=len(clear),new_RAW_run=False,actual138_observations_replayed=True,maximum_coefficient_difference=maximum,fallback_activations=active))
    # Active-branch prefix causality using the real current observations.
    side=rows(OUT/'continuous6/predictions.jsonl');prefix=Predictor()
    for a in side[:80]:
        q=prefix.step(a['t'],a['body'],a['models'])
        for method,c in q['coefficients'].items():
            ref=a['coefficients'][method]
            if c:
                for key in ['position','velocity','acceleration']:np.testing.assert_allclose(c[key],ref[key],rtol=0,atol=1e-9)
    save(OUT/'checks140.json',dict(actual_RAW865_complete=True,availability=availability,clear400_replay_exact=True,side80_active_prefix_prediction_replay_exact=True,independent_RAW_prefix_for_active_fallback=False))
    print(json.dumps(availability),flush=True)
if __name__=='__main__':main()
