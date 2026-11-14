"""Forecasts first, sparse-box endpoint scoring afterwards, exact target timestamps."""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import numpy as np
from predict22 import ROOT,PARAMS,Predictor,evaluate
from checks22 import run as checks

OUT=ROOT.parent/'output/stage22_sparse_endpoint_20261018';INPUT=ROOT.parent/'output/stage21_native_windows_20261018'
CASES=['clear6','side6','review4','negative5'];FRONTENDS=['native20','native10','native5','warp20'];MODELS=['hold','cv','cv_hist_acc']


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')


def sources():
    ff=[Path(__file__),Path(__file__).with_name('predict22.py'),Path(__file__).with_name('checks22.py'),ROOT/'research/stage14/motion.py']
    result={str(p.relative_to(ROOT)):sha(p) for p in ff}
    for p,v in load(INPUT/'protocol.json')['source_hashes'].items():assert sha(ROOT/p)==v;result[p]=v
    return result


def label_path(case):return ROOT.parent/'output/raw_only_project/evaluation_data'/('train4.json' if case=='review4' else 'train6.json')


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    assert (INPUT/'analysis.json').exists()
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,
        inputs={c:sha(INPUT/c/'observations.jsonl') for c in CASES},label_hashes={n:sha(ROOT.parent/'output/raw_only_project/evaluation_data'/n) for n in ['train4.json','train6.json']},
        frontends=FRONTENDS,models=MODELS,
        design='Frozen stage21 observations; infer causal hold/CV/history-acceleration coefficients without reading labels. Score only after all prediction files saved.',
        evaluation='For each archived present target timestamp T and nominalH, take last predictor cutoff<=T-H (gap<=25ms); evaluate saved polynomial at exactT. Actual lookahead in[H,H+25ms], reported. Boxes supply target centers, never detector future outputs. No ADE claim.',
        validity='5real measurements minimum; new forecasts expire160ms after last real observation; reset after500ms. Missing forecasts count against coverage; no replacing missing with a future estimate.',
        limits='Development sparse2D whole-box centers/time alignment are approximate, not COM/pose/RPM truth. Same data previously used to explore detection; no blind generalization claim. No G or access to yolov10-main.'))
    print('Stage22 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def infer():
    protocol=verify()
    for case in CASES:
        path=INPUT/case/'observations.jsonl';assert sha(path)==protocol['inputs'][case]
        rr=list(map(json.loads,path.read_text().splitlines()));folder=OUT/case;folder.mkdir(parents=True,exist_ok=True)
        assert not (folder/'forecasts.jsonl').exists();predictors={m:Predictor() for m in FRONTENDS};n=0
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r in rr:
                row=dict(t=r['t'],frontends={})
                for m,p in predictors.items():
                    o=r['methods'][m]['observation'];row['frontends'][m]=p.step(r['t'],o['center'] if o['detected'] else None)
                f.write(json.dumps(row,allow_nan=False)+'\n');n+=1
        save(folder/'completion.json',dict(frames=n,labels_read_by_inference=False,sources_unchanged=sources()==protocol['source_hashes']))
    print('All forecasts saved before label scoring',flush=True)


def score():
    protocol=verify();source_protocol=load(INPUT/'protocol.json');details=[];table=[];common=[]
    for name,value in protocol['label_hashes'].items():assert sha(ROOT.parent/'output/raw_only_project/evaluation_data'/name)==value
    for case in CASES:
        rows=list(map(json.loads,(OUT/case/'forecasts.jsonl').read_text().splitlines()));tt=np.array([r['t'] for r in rows])
        if case=='negative5':labels=[]
        else:
            cfg=source_protocol['cases'][case]
            labels=[r for r in load(label_path(case))['labels'] if cfg['start']<r['t']<=cfg['start']+cfg['duration']]
        for label in labels:
            for h in PARAMS['horizons']:
                query=label['t']-h;i=int(np.searchsorted(tt,query,side='right')-1);anchor_available=i>=0 and query-tt[i]<=.025
                for front in FRONTENDS:
                    q=rows[i]['frontends'][front] if anchor_available else None
                    for model in MODELS:
                        valid=anchor_available and q['valid'];d=dict(case=case,frontend=front,model=model,target_t=label['t'],nominal_h=h,
                            target_state=label['state'],anchor_available=bool(anchor_available),forecast_valid=bool(valid))
                        if anchor_available:d.update(anchor_t=float(tt[i]),actual_h=float(label['t']-tt[i]),anchor_current_measurement=q['anchor_has_current_measurement'])
                        if valid:
                            predicted=evaluate(q['coefficients'][model],label['t']-tt[i]);d['predicted']=predicted
                            if label['state']=='present':
                                b=np.array(label['bbox']);center=(b[:2]+b[2:])/2;error=float(np.linalg.norm(np.array(predicted)-center));diag=np.linalg.norm(b[2:]-b[:2])
                                d.update(error_px=error,target_center=center.tolist(),joint_success=bool(error<=diag*.25))
                            else:d['inside_image']=0<=predicted[0]<1280 and 0<=predicted[1]<720
                        details.append(d)
        for front in FRONTENDS:
            for model in MODELS:
                for h in PARAMS['horizons']:
                    dd=[d for d in details if d['case']==case and d['frontend']==front and d['model']==model and d['nominal_h']==h]
                    pp=[d for d in dd if d['target_state']=='present'];valid=[d for d in pp if d['forecast_valid']];errors=[d['error_px'] for d in valid]
                    aa=[d for d in dd if d['target_state']=='absent']
                    table.append(dict(case=case,frontend=front,model=model,nominal_h=h,present_targets=len(pp),anchors_available=sum(d['anchor_available'] for d in pp),
                        valid_forecasts=len(valid),mean_fde_px=float(np.mean(errors)) if errors else None,median_fde_px=float(np.median(errors)) if errors else None,
                        max_fde_px=max(errors) if errors else None,joint_success=sum(d.get('joint_success',False) for d in valid),
                        absent_targets=len(aa),absent_forecasts_inside=sum(d['forecast_valid'] and d.get('inside_image',False) for d in aa),
                        known_negative_published_anchors=sum(r['frontends'][front]['valid'] for r in rows) if case=='negative5' else None))
        for h in PARAMS['horizons']:
            per={(f,m):{d['target_t']:d for d in details if d['case']==case and d['frontend']==f and d['model']==m and d['nominal_h']==h and d['target_state']=='present'} for f in FRONTENDS for m in MODELS}
            shared=[t for t in per[(FRONTENDS[0],MODELS[0])] if all(p[t]['forecast_valid'] for p in per.values())]
            common.append(dict(case=case,nominal_h=h,n=len(shared),mean_fde_px={f:{m:float(np.mean([per[f,m][t]['error_px'] for t in shared])) if shared else None for m in MODELS} for f in FRONTENDS}))
    save(OUT/'analysis.json',dict(table=table,details=details,common=common,
        correction='Sparse position boxes support timestamp-aligned endpoint errors by selecting earlier anchors; dense trajectory ADE still requires denser truth.'))
    print('Sparse-box endpoint scoring complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score','all']);a=p.parse_args()
    if a.action=='all':infer();score()
    else:globals()[a.action]()
