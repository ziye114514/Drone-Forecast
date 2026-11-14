from pathlib import Path
import json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;OUT=WORK/'output/stage141_periodic_axis_20261024'
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def main():
    a=rows(OUT/'continuous6/predictions.jsonl');b=rows(WORK/'output/stage140_visible_fallback_20261024/continuous6/predictions.jsonl');assert len(a)==len(b)==475;maximum=0.
    for r,s in zip(a,b):
        for key in ['t','body','valid','coefficients','predictions']:assert r[key]==s[key],(r['t'],key)
        assert r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6)
        assert not r['periodic_axis']['forecast_uses_this_axis'] and not r['periodic_axis']['thrust_estimated'] and not r['periodic_axis']['mechanical_RPM_estimated']
    cap=cv2.VideoCapture(str(OUT/'continuous6/predictions.mp4'));decoded=0
    while True:
        ok,_=cap.read()
        if not ok:break
        decoded+=1
    cap.release();assert decoded==475
    c=json.loads((OUT/'continuous6/completion.json').read_text('utf-8'));assert c['sources_unchanged'] and not c['labels_read'] and not c['output_caches_read']
    native=[r['periodic_axis'] for r in a];result=dict(frames=len(a),forecast_exact140=True,periodic_ROI_qualified=sum(q['qualified'] for q in native),unsigned_line_candidates=sum(q['axis_valid'] for q in native),not_pose_or_RPM_truth=True,no_frequency_input_to_predictor=True,decoded_video_frames=decoded,event_cutoffs=True)
    (OUT/'checks141.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
