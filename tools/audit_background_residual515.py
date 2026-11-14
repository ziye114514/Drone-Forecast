"""Remaining coordinate-reference errors; labels are retrospective only."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];base=ROOT.parent
if __name__=='__main__':
    out=base/'output/background_residual515_20261113';out.mkdir(exist_ok=False)
    current=json.loads((base/'output/raw_bg_selection512_20261113/results.json').read_text('utf8'))
    camera=json.loads((base/'output/raw_bg_endpoints509_20261112/results.json').read_text('utf8'))
    records=[]
    for n in ('13','15','16','11','3','4'):
        rows=[json.loads(s) for s in (base/f'output/raw_bg_selection512_20261113/recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
        before={r['t']:r for r in current['cases'][n]['baseline487']['records'] if r['valid']}
        pure={r['t']:r for r in camera['cases'][n]['background']['records'] if r['valid']}
        ids={r['t'] for r in json.loads((base/'output/reference429_20261110/results.json').read_text('utf8'))['cases'][n]['control410']['records'] if r['valid']}
        for s in current['cases'][n]['background']['records']:
            t=s['t']
            if not s['valid'] or t not in ids:continue
            r=[q for q in rows if q['t']<=t+1e-9][-1];info=r['background_reference_model_average'];bg=info['background'].get('background_frame_motion') or {}
            records.append(dict(recording=n,t=t,error=s['error_px'],old_error=before[t]['error_px'],camera_error=pure[t]['error_px'],
                weight=info['weight'],weight_reason=info['reason'],rows=info['rows'],background=bg,
                motion=r['motion'],mode=r['structure']['mode'],current_position_error=s['current_position_error_px']))
    (out/'results.json').write_text(json.dumps(dict(n=len(records),records=records,retrospective_only=True,
        no_hyperparameter_fit=True,new_video_unread=True),indent=2),'utf8')
    for r in sorted(records,key=lambda q:-q['error'])[:18]:
        print(r['recording'],r['t'],'FDE',round(r['error'],2),'current',round(r['current_position_error'],2),
            'old/camera',round(r['old_error'],2),round(r['camera_error'],2),'w',round(r['weight'],4),
            r['weight_reason'],'rows',r['rows'])
