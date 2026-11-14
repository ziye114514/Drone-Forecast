"""Copy sparse human event boxes for evaluation ONLY; no predictor imports this."""
from pathlib import Path
import json,hashlib

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/raw_only_project/evaluation_data'
OUT.mkdir(parents=True,exist_ok=True)
records=[]
for sid in (1,2,3,4,6):
    folder=Path('E:/dataset/train')/str(sid)
    source=folder/'annotations/portable_annotations.json'
    if not source.exists():continue
    data=json.loads(source.read_text(encoding='utf-8-sig'))
    samples=[]
    for s in data['samples']:
        event=s.get('event',{})
        if event.get('state') not in ('present','absent'):continue
        row=dict(t=float(s['event_time_sec']),state=event['state'],avi_frame=s['avi_frame'])
        if event['state']=='present' and event.get('box'):
            w,h=event['size'];row['bbox']=[event['box'][i]*(1280/w if i%2==0 else 720/(180 if h==184 else h)) for i in range(4)]
            # 320x184 has 320x180 effective content plus four bottom padding rows.
        elif event['state']=='present':continue
        samples.append(row)
    raw=folder/('media/event.raw' if sid==6 else 'aligned/event.raw' if sid in (2,3,4) else 'event.raw')
    if sid==1:
        m=json.loads((folder/'portable_manifest.json').read_text(encoding='utf-8-sig'))
        raw=folder/m['synchronized_input']['raw']
    info=dict(sequence=f'train{sid}',raw=str(raw),labels=samples,
              source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              label_kind='existing human event-image bounding boxes; sparse; no interpolated GT',
              effective_event_grid=[1280,720],role='development' if sid==6 else 'heldout_for_this_prototype',
              caveat='These are dataset train recordings; not an official independent dataset test split.')
    dest=OUT/f'train{sid}.json';dest.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    records.append(dict(sequence=info['sequence'],raw=str(raw),label_file=str(dest),present=sum(x['state']=='present' for x in samples),absent=sum(x['state']=='absent' for x in samples),duration_s=max(x['t'] for x in samples),role=info['role']))
(OUT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False))
