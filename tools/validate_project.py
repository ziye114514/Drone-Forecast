"""Validate frozen source/config provenance, published predictions, and delivered media."""
from pathlib import Path
import json,re,hashlib,zipfile
import numpy as np,cv2
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/raw_only_project'
protocol=json.loads((OUT/'experiment_protocol.json').read_text(encoding='utf-8'))
source_matches={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in protocol['source_files'].items()}
runs={};all_rows=0
for sid in (6,2,3,4):
    folder=OUT/'runs'/f'train{sid}'
    config_equal=(folder/'config.json').read_bytes()==(OUT/'frozen_config.json').read_bytes()
    manifest=json.loads((folder/'run_manifest.json').read_text())
    traces=[json.loads(s) for s in (folder/'forecasts.jsonl').read_text().splitlines()]
    active=[r for r in traces if r['active']]
    finite=all(np.isfinite(np.array([v for state in r['states'].values() for v in state.values()])).all() for r in active)
    evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    counts={h:{r['available'] for r in evaluation['metrics'] if r['horizon_s']==h} for h in (.1,.2,.4)}
    same_anchors=all(len(v)==1 for v in counts.values())
    runs[f'train{sid}']=dict(config_matches=config_equal,inference_raw_only=manifest['inference_external_inputs']==[manifest['input_raw']],
                            finite_states=finite,same_method_coverage=same_anchors,frames=len(traces),active_frames=len(active))
    all_rows+=len(traces)
media={}
for name in ('train6','train3','train6_close'):
    folder=OUT/'visuals'/name
    cap=cv2.VideoCapture(str(folder/'current_vs_future.mp4'))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));decoded=[]
    for index in (0,max(0,count//2),max(0,count-1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();decoded.append(bool(ok and frame.shape==(600,1600,3)))
    cap.release()
    with Image.open(folder/'current_vs_future.gif') as gif:gif_count=gif.n_frames
    media[name]=dict(video_frames=count,keyframes_decoded=decoded,gif_frames=gif_count)
page=(OUT/'查看项目结果.html').read_text(encoding='utf-8')
links=re.findall(r'(?:src|href)="([^"#]+)"',page)
missing=[x for x in links if not x.startswith(('https:','http:')) and not (OUT/x).exists()]
causality=json.loads((OUT/'causality_validation.json').read_text())
validation=dict(source_matches=source_matches,runs=runs,published_frames=all_rows,media=media,missing_report_links=missing,
    prefix_check=causality,checks='Static provenance, finite outputs, common anchors, real RAW prefix equality, and decoded videos/GIFs.')
valid=all(source_matches.values()) and not missing and all(all(r[k] for k in ('config_matches','inference_raw_only','finite_states','same_method_coverage')) for r in runs.values()) and all(all(r['keyframes_decoded']) and r['gif_frames']>10 for r in media.values()) and causality['forecasts_exactly_equal'] and causality['observations_exactly_equal']
validation['passed']=valid
(OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps(validation))
if not valid:raise SystemExit(1)
