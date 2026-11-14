"""Frozen matched RAW-support experiment; contemporaneous cached NN/axis."""
from pathlib import Path
import argparse,collections,copy,datetime,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage124'),str(ROOT/'research/stage122')]
from occupancy124 import SupportFrontend,PARAMS,MergeRecovery
from worker122 import PartialModel,geom,CenterBank,NAMES,Observer,open_raw,raw_chunks,ValidatedSelector
OUT=ROOT.parent/'output/stage124_occupancy_support_20261023'
REF=ROOT.parent/'output/stage123_frozen_extension_20261023'
CASES=dict(extension1=dict(raw='E:/dataset/train/1/event.raw',start=40.,duration=6.,ref=str(REF/'raw/extension1/observations_forecasts.jsonl')),
    extension2=dict(raw='E:/dataset/train/2/aligned/event.raw',start=40.,duration=6.,ref=str(REF/'raw/extension2/observations_forecasts.jsonl')),
    natural6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=8.,ref=str(REF/'raw/natural6/observations_forecasts.jsonl')),
    continuous6=dict(raw='E:/dataset/train/6/media/event.raw',start=110.,duration=9.5,ref=str(ROOT.parent/'output/stage107_center_bias_filter_20261022/raw/continuous6/observations_forecasts.jsonl')),
    negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6,ref=str(REF/'raw/negative5/observations_forecasts.jsonl')))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),'utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def sources():
    paths=[Path(__file__),ROOT/'research/stage124/occupancy124.py']
    paths += [p for d in [8,9,21,55,58,61,104,107,110,113,117,119,121,122] for p in (ROOT/f'research/stage{d}').glob('*.py')]
    paths += list((ROOT/'src/event_drone_forecast').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def freeze():
    assert not (OUT/'protocol124.json').exists()
    save(OUT/'protocol124.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,cases=CASES,
        references={k:sha(v['ref']) for k,v in CASES.items()},raw_metadata={v['raw']:dict(bytes=Path(v['raw']).stat().st_size,mtime_ns=Path(v['raw']).stat().st_mtime_ns) for v in CASES.values()},
        scope='New RAW geometry; cached contemporaneous NN/axis only. Not a new full RAW-only deployment. Historical development clips, no blind-test claim.',
        design='Four support/ROI ablations fixed after visual two-snapshot development pilot, before running intervals or reading new outcome scores. One fixed .5 binary majority threshold; no search on future endpoints. Baseline identical123 wherever original input available. Models unchanged; first four/five-frame lock remains mandatory.',
        primary='Same original conic gates, qualified-frame coverage/first lock; original negative and clear/side regressions. Rotor candidate count is not true visibility. Current NN and RAW cutoffs only. Score sparse future boxes only after all predictions/prefixes.',default_promoted=False))
def verify():
    p=json.loads((OUT/'protocol124.json').read_text('utf-8'));assert p['sources']==sources();return p
def stable(x):
    if isinstance(x,dict):return {k:stable(v) for k,v in x.items() if k not in ['elapsed_ms','processing_ms']}
    if isinstance(x,list):return [stable(v) for v in x]
    return x
def run(case,prefix=False):
    p=verify();cfg=CASES[case];assert sha(cfg['ref'])==p['references'][case];refs={round(r['t'],8):r for r in rows(cfg['ref'])}
    folder=OUT/(case+('_prefix' if prefix else ''));folder.mkdir(exist_ok=False)
    reader=open_raw(cfg['raw']);h,w=reader.get_size();fronts={};models={};banks={}
    for mode in PARAMS['modes']:
        f=Observer(w,h,'native20');f.geometry=MergeRecovery(w,h,'merge_safe') if mode=='count_global' else SupportFrontend(w,h,mode)
        fronts[mode]=f;models[mode]=PartialModel('affine3');banks[mode]=CenterBank()
    begin=max(0.,cfg['start']-.1);duration=min(cfg['duration'],1.2) if prefix else cfg['duration'];count=0;baseline_exact=0
    video=cv2.VideoWriter(str(folder/'support.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(w,h)) if not prefix else None
    with (folder/'records.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,begin,duration+cfg['start']-begin,20000):
            r=refs.get(round(t,8));body=r['body_observation'] if r else dict(t=t,detected=False);axis=r['body_axis'] if r else dict(valid=False)
            outputs={}
            for mode,front in fronts.items():
                start=time.perf_counter();g=front.geometry
                if mode!='count_global':g.set_hint(e,body,t)
                obs=front.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
                if t<=cfg['start']+1e-8:continue
                rotors=obs.get('rotors',[]);m=geom.rigid104.measure_four(rotors,body)
                _,d=models[mode].step(t,body,rotors,axis,m);forecast=banks[mode].step(t,body,d,axis)
                outputs[mode]=dict(geometry=obs,center=d,forecast=forecast,qualification=m['reason'],qualified=m['valid'],
                    candidate_count=len(g.candidates),preline=g.preline_audit,merge=g.merge_audit,group=g.group_info,
                    roi=None if mode=='count_global' else g.current_roi,processing_ms=(time.perf_counter()-start)*1000)
            if t<=cfg['start']+1e-8:continue
            assert r is not None
            body_controls=[v['forecast']['coefficients'].get(NAMES[0]) for v in outputs.values()];assert all(v==body_controls[0] for v in body_controls)
            old=r['frontends']['delayed_feedback'].get('learned_structure118')
            if old is not None:assert outputs['count_global']['geometry'].get('rotors',[])==old['input_rotors'],t;baseline_exact+=1
            rec=dict(t=t,body=body,axis=axis,valid=r['frontends']['delayed_feedback']['valid'],outputs=outputs,event_count=len(e),max_event_t=int(e['t'].max()) if len(e) else None)
            stream.write(json.dumps(rec,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n');count+=1
            if video is not None:
                im=np.full((h,w,3),255,np.uint8);pix=e['y'].astype(np.int32)*w+e['x'];on=np.bincount(pix[e['p']>0],minlength=w*h).reshape(h,w)>0;off=np.bincount(pix[e['p']<=0],minlength=w*h).reshape(h,w)>0
                im[on]=(100,100,240);im[off]=(220,170,30);im[on&off]=0
                for mode,col in [('count_roi',(0,180,220)),('occupancy_roi',(220,0,160))]:
                    for rot in outputs[mode]['geometry'].get('rotors',[]):
                        cv2.ellipse(im,((rot['cx'],rot['cy']),(rot['major'],rot['minor']),rot['angle']),col,2)
                cv2.putText(im,f'{case} {t:.2f}s | count=yellow occupancy=purple',(12,h-16),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2);video.write(im)
            if count%100==0:print(case,count,flush=True)
    if video is not None:video.release()
    assert count==round(duration*50)
    check=dict(case=case,frames=count,raw_geometry=True,cached_current_NN_axis=True,labels_read=False,baseline_rotors_exact123=baseline_exact,matched_body_KF_exact=True,sources_unchanged=verify()['sources']==sources())
    if prefix:
        full=rows(OUT/case/'records.jsonl')[:count];trunc=rows(folder/'records.jsonl');assert stable(full)==stable(trunc);check['actual_RAW_prefix_exact']=True
    save(folder/'checks124.json',check);print(json.dumps(check),flush=True)
def summarize():
    verify();table=[]
    for case in CASES:
        rr=rows(OUT/case/'records.jsonl')
        for mode in PARAMS['modes']:
            oo=[r['outputs'][mode] for r in rr];streak=longest=0
            for o in oo:streak=streak+1 if o['qualified'] else 0;longest=max(longest,streak)
            table.append(dict(case=case,mode=mode,frames=len(rr),body=sum(r['body'].get('detected',False) for r in rr),
                four=sum(len(o['geometry'].get('rotors',[]))==4 for o in oo),qualified=sum(o['qualified'] for o in oo),longest_qualified_streak=longest,
                current=sum(o['center']['current'] for o in oo),partial_current=sum(o['center']['current'] and len(o['geometry'].get('rotors',[]))==3 for o in oo),
                first_structure=next((r['t'] for r,o in zip(rr,oo) if o['center'].get('structure_ready')),None),
                reasons=dict(collections.Counter(o['qualification'] for o in oo)),median_ms=float(np.median([o['processing_ms'] for o in oo]))))
    save(OUT/'availability124.json',table);print(json.dumps(table),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','summarize']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='summarize':summarize()
    else:run(a.case,a.action=='prefix')
