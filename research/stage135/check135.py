"""Necessary RAW output checks: same input/results, prefix, empty, publication."""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage135'),str(ROOT/'research/stage124')]
from publication135 import Publication
from study124 import save,stable
OUT=ROOT.parent/'output/stage135_optimized_forecast_20261024'
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def publication_check():
    p=Publication();assert [p.step(i*.02,True) for i in range(5)]==[False]*4+[True]
    assert p.step(.2,False) and not p.step(.3,False);assert not p.step(.8,True)
    assert not p.step(.82,True) and not p.step(.84,True) and not p.step(.86,True) and p.step(.88,True)
def check(case):
    publication_check();refcase='natural6' if case in ['prefix_natural6','natural6_fast'] else case;rr=rows(OUT/'raw'/case/'predictions.jsonl');cc=json.loads((OUT/(refcase+'.json')).read_text('utf-8'))[:len(rr)];bb=rows(ROOT.parent/'output/stage124_occupancy_support_20261023'/refcase/'records.jsonl')[:len(rr)];assert len(rr)==len(cc)==len(bb)
    maximum=0.
    for r,c,b in zip(rr,cc,bb):
        assert r['t']==c['t']==b['t'];assert r['body']==b['body'];assert r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6)
        for name,value in c['output']['coefficients'].items():
            actual=r['coefficients'][name];assert (value is None)==(actual is None)
            if value is not None:
                for k in ['position','velocity','acceleration']:maximum=max(maximum,float(np.max(abs(np.array(value[k])-actual[k]))))
        assert stable(r['predictor'])==stable(c['output']),(r['t'],'optimizer differs current-input replay')
    assert maximum<1e-8
    if case=='prefix_natural6':
        full=rows(OUT/'raw/natural6/predictions.jsonl')[:len(rr)]
        for a,b in zip(rr,full):
            assert a['frame_sha256']==b['frame_sha256']
            for key in ['body','coefficients','predictions','valid','predictor']:assert stable(a[key])==stable(b[key])
    if case=='negative5':assert all(not r['valid'] for r in rr)
    completion=json.loads((OUT/'raw'/case/'completion.json').read_text('utf-8'));assert completion['sources_unchanged'] and not completion['labels_read'] and not completion['output_caches_read']
    decoded=0
    if case in ['natural6','natural6_fast']:
        cap=cv2.VideoCapture(str(OUT/'raw'/case/'predictions.mp4'))
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
        cap.release();assert decoded==len(rr)==400
    save(OUT/'raw'/case/'checks135.json',dict(case=case,frames=len(rr),actual_RAW=True,NN_same_current_input=True,predictor_exact135_replay=True,maximum_coefficient_difference=maximum,actual_prefix_exact=case=='prefix_natural6',negative=case=='negative5',decoded_frames=decoded,publication_after_identity_gap_correct=True,no_cached_features_in_inference=True))
    print(json.dumps(dict(case=case,frames=len(rr),max_coeff_difference=maximum)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['natural6','natural6_fast','prefix_natural6','negative5']);a=p.parse_args();check(a.case)
