"""Frozen v0.3 real-RAW regression, sparse endpoint scoring and causality check."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,shutil,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.motion import extrapolate
OUT=ROOT.parent/'output/paper_reproduction_v030'
LABELS=ROOT.parent/'output/raw_only_project/evaluation_data'
RAWS={'train6':r'E:\dataset\train\6\media\event.raw',
      **{f'train{i}':fr'E:\dataset\train\{i}\aligned\event.raw' for i in (2,3,4)}}

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def csvsave(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def readlines(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f]

def source_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/'src/event_drone_forecast').glob('*.py')):h.update(p.read_bytes())
    return h.hexdigest()

def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'experiment_protocol.json').exists():raise RuntimeError('Already frozen')
    PaperConfig().save(OUT/'frozen_config.json')
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    hashes={}
    for folder in ('src','tests'):
        for p in (ROOT/folder).rglob('*.py'):
            rel=p.relative_to(ROOT);dest=snapshot/rel;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dest)
            hashes[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('run_paper.py',):shutil.copy2(ROOT/name,snapshot/name)
    (OUT/'evaluation_data').mkdir(exist_ok=True)
    for name in RAWS:shutil.copy2(LABELS/f'{name}.json',OUT/f'evaluation_data/{name}.json')
    save(OUT/'experiment_protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        version='0.3.0',source_sha256=source_hash(),source_files=hashes,raws=RAWS,
        role='Engineering regression on recordings already examined in v0.1/v0.2. Not an unseen holdout.',
        inference='RAW only; no human box, reconstruction or future input enters estimator.',
        subsets='All targets; anchors with physics active; four observed ellipses AND physics active; partial/line anchors.',
        fixed_visual_intervals=dict(train6_four=[22,28],train6_side=[109,114]),
        limitations=['No independent real-flight RPM, pose, thrust or metric position ground truth.',
          'Sparse box centers define endpoint errors, not dense ADE; physical rotor/body centers may differ.',
          'Weak-perspective axis, image up, two blades and nominal pixel gravity are unverified assumptions.',
          'Four-ellipse subset is selected by past detector output, not independent visibility labels.',
          'Sequential RAW consumption is causal; batch runtime is offline and not sensor capture latency.']))
    print('Frozen',source_hash(),flush=True)

def run(names):
    protocol=json.loads((OUT/'experiment_protocol.json').read_text(encoding='utf-8'))
    if source_hash()!=protocol['source_sha256']:raise RuntimeError('Runtime changed after freeze')
    for name in names:
        dest=OUT/f'runs/{name}'
        if (dest/'summary.json').exists():continue
        subprocess.run([sys.executable,str(ROOT/'core/run_paper.py'),'--raw',RAWS[name],'--output',str(dest),
                        '--config',str(OUT/'frozen_config.json')],check=True)

def evaluate():
    allrows=[];summaries=[];statistics=[]
    for name in RAWS:
        run=OUT/f'runs/{name}';traces=readlines(run/'forecasts.jsonl');obs=readlines(run/'observations.jsonl')
        labels=json.loads((OUT/f'evaluation_data/{name}.json').read_text(encoding='utf-8'))['labels']
        times=np.array([q['t'] for q in traces]);rows=[]
        methods=list(next(q['states'] for q in traces if q.get('active')))
        for label in labels:
            if label['state']!='present' or label['t']>times[-1]+.015:continue
            box=np.array(label['bbox']);truth=(box[:2]+box[2:])/2
            for h in (.1,.2,.4):
                k=np.searchsorted(times,label['t']-h,side='right')-1
                if k<0:continue
                q=traces[k];o=obs[k]
                for method in methods:
                    row=dict(sequence=name,target_time_s=label['t'],anchor_time_s=times[k],horizon_s=h,
                        actual_horizon_s=label['t']-times[k],method=method,available=q.get('active',False),
                        physics_active=q.get('physics_active',False),four_ellipses=o.get('four_ellipses',False),
                        partial_or_line=any(r.get('mode') in ('arc','line') for r in o.get('detected_rotors',[])),
                        truth_x=truth[0],truth_y=truth[1])
                    if row['available']:
                        p=extrapolate(q['states'][method],row['actual_horizon_s']);error=float(np.linalg.norm(p-truth))
                        row.update(pred_x=p[0],pred_y=p[1],error_px=error,normalized_error=error/max(np.linalg.norm(box[2:]-box[:2]),1))
                    rows.append(row)
        csvsave(run/'evaluation_details.csv',rows);allrows+=rows
        summary=json.loads((run/'summary.json').read_text());summary['sequence']=name
        summary['four_ellipse_frames']=sum(o.get('four_ellipses',False) for o in obs)
        summary['arc_observation_frames']=sum(any(r.get('mode')=='arc' for r in o.get('detected_rotors',[])) for o in obs)
        # Runtime summary reports the final tracking episode. Count positive
        # update increments across resets for whole-recording learning totals.
        previous_updates=0;whole_updates=0
        for q in traces:
            value=q.get('physical_parameters',{}).get('paper_adaptive',{}).get('updates',0)
            whole_updates+=max(0,value-previous_updates) if value>=previous_updates else value
            previous_updates=value
        summary['final_episode_physical_updates']=summary.pop('physical_updates')
        summary['whole_recording_physical_updates']=whole_updates
        statistics.append(summary)
        del traces,obs
    for sequence in [*RAWS,'all','regression_234']:
        seqrows=[r for r in allrows if sequence=='all' or (sequence=='regression_234' and r['sequence']!='train6') or r['sequence']==sequence]
        for subset in ('all','physics_active','four_ellipses_physics','partial_or_line'):
            selected=[r for r in seqrows if subset=='all' or (subset=='physics_active' and r['physics_active'])
                or (subset=='four_ellipses_physics' and r['physics_active'] and r['four_ellipses'])
                or (subset=='partial_or_line' and r['partial_or_line'])]
            for h in (.1,.2,.4):
                for method in sorted(set(r['method'] for r in seqrows)):
                    rows=[r for r in selected if r['method']==method and r['horizon_s']==h];valid=[r for r in rows if r['available']]
                    errors=[r['error_px'] for r in valid]
                    summaries.append(dict(sequence=sequence,subset=subset,horizon_s=h,method=method,targets=len(rows),available=len(valid),
                        coverage=len(valid)/max(len(rows),1),mean_error_px=float(np.mean(errors)) if errors else None,
                        median_error_px=float(np.median(errors)) if errors else None,p90_error_px=float(np.percentile(errors,90)) if errors else None))
    csvsave(OUT/'all_endpoint_errors.csv',allrows);csvsave(OUT/'metrics.csv',summaries)
    save(OUT/'evaluation.json',dict(protocol='Actual human future box centers only; no GT interpolation; previous published states only.',
        role='Previously examined recordings: engineering regression, not unseen generalization.',metrics=summaries,run_statistics=statistics))
    # Common actual target timestamps across previous versions, with new fused
    # and physical branches scored separately. No availability cherry-picking.
    common=[]
    for version,folder in [('v0.1','raw_only_project'),('v0.2','raw_only_project_integrated')]:
        previous=[]
        for name in RAWS:
            p=ROOT.parent/f'output/{folder}/runs/{name}/evaluation_details.csv'
            if p.exists():
                with p.open(encoding='utf-8-sig') as f:previous+=list(csv.DictReader(f))
        lookup={(r['sequence'],round(float(r['target_time_s']),8),float(r['requested_horizon_s'])):r
                for r in previous if r['method']=='adaptive' and r.get('error_px')}
        for subset in ('all','four_ellipses_physics'):
            for h in (.1,.2,.4):
                for method in ('cv','paper_nominal','paper_adaptive','paper_fused'):
                    now=[r for r in allrows if r['method']==method and r['horizon_s']==h and r['available']
                        and (subset=='all' or (r['four_ellipses'] and r['physics_active']))]
                    pairs=[(r,lookup[(r['sequence'],round(r['target_time_s'],8),h)]) for r in now
                           if (r['sequence'],round(r['target_time_s'],8),h) in lookup]
                    common.append(dict(previous=version,subset=subset,horizon_s=h,method=method,n=len(pairs),
                        previous_mean_error_px=float(np.mean([float(b['error_px']) for a,b in pairs])) if pairs else None,
                        current_mean_error_px=float(np.mean([a['error_px'] for a,b in pairs])) if pairs else None))
    csvsave(OUT/'common_targets_previous_versions.csv',common)
    print(json.dumps([r for r in summaries if r['sequence']=='all' and r['method'] in ('cv','paper_nominal','paper_adaptive','paper_fused') and r['horizon_s']==.4]),flush=True)

def prefix():
    dest=OUT/'causality_prefix_check'
    if not (dest/'summary.json').exists():
        subprocess.run([sys.executable,str(ROOT/'core/run_paper.py'),'--raw',RAWS['train3'],'--output',str(dest),
                        '--duration','4','--config',str(OUT/'frozen_config.json')],check=True)
    result={}
    for filename in ('observations.jsonl','forecasts.jsonl'):
        short=(dest/filename).read_bytes().splitlines();full=(OUT/'runs/train3'/filename).read_bytes().splitlines()
        result[filename]=dict(prefix_frames=len(short),exactly_equal=short==full[:len(short)])
    save(OUT/'causality_validation.json',result)
    if not all(v['exactly_equal'] for v in result.values()):raise AssertionError(result)
    print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','evaluate','prefix']);p.add_argument('--sequences',nargs='+',default=list(RAWS))
    a=p.parse_args()
    if a.action=='run':run(a.sequences)
    else:globals()[a.action]()
