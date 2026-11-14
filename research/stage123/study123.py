"""Frozen natural-observation extension; no feature dropout or new parameters."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research/stage122')]
import run_validated_structure_forecast as entry
from study122 import rows,save,sha,stable
OUT=ROOT.parent/'output/stage123_frozen_extension_20261023'
LABELS=ROOT.parent/'output/raw_only_project/evaluation_data'
CASES=dict(extension1=dict(raw='E:/dataset/train/1/event.raw',start=40.,duration=6.,labels='train1.json',role='Different recording from122; outside metadata-known30..35, unknown older use/training possible'),
    extension2=dict(raw='E:/dataset/train/2/aligned/event.raw',start=40.,duration=6.,labels='train2.json',role='Different recording from122; previously exposed to early pipelines, frozen extension not blind'),
    natural6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=8.,labels='train6.json',role='Known development positive control extended to30s; natural observations, no mask'),
    prefix_extension1=dict(raw='E:/dataset/train/1/event.raw',start=40.,duration=2.,labels='train1.json',role='Independent actual RAW prefix'),
    negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6,labels=None,role='Known negative control; no sparse label file'))
METHODS=['center_body_kf107','partial_body_hold_kf120','partial_affine3_kf120','partial_conic3_kf120','partial_validated_kf122']

def sources():
    ns=entry.build();r=ns['fingerprints']();ns.clear();r[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__);return r
def freeze():
    assert not (OUT/'protocol123.json').exists()
    metadata={}
    for c in CASES.values():
        p=Path(c['raw']);s=p.stat();metadata[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    save(OUT/'protocol123.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),cases=CASES,methods=METHODS,
        raw_metadata=metadata,label_sha256={c['labels']:sha(LABELS/c['labels']) for c in CASES.values() if c['labels']},
        design='Unchanged122, dropout=-1, actualRAW only. Intervals fixed by readable40s metadata, not current geometry or future errors. Two separate recordings plus known positive natural control, actual prefix, negative. No model/threshold/solver-budget changes.',
        success='At least five consecutive native20 ticks with three current observed rotors, prior learned structure and current selected geometry, followed by four-view recovery in same run. Also count full current model gaps>=100ms with recovery. Zero model locks is a failure denominator, not discarded.',
        evaluation='Only after all forecast jobs: existing sparse manual event boxes exported by prepare_evaluation.py. They measure bbox center, not COM/attitude/RPM. At each present target, use latest published valid origin<=target-h with lag<=20ms. Score coefficient position+(actual target-origin)*velocity, nominal horizons.1/.2/.4. All five listed methods have zero acceleration. Pairwise common eligible origins; record coverage, absent-label publications, first lock and current geometry separately.',
        historical_exposure='historical_exposure123.json; this is frozen extension, not a claim of unseen subjects or blind detection. Existing labels and detector training may overlap.',
        no_feature_dropout=True,no_future_labels_in_inference=True,default_promoted=False))

def verify():
    p=json.loads((OUT/'protocol123.json').read_text('utf-8'));assert p['sources']==sources()
    for raw,m in p['raw_metadata'].items():
        s=Path(raw).stat();assert m==dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return p

def run(case):
    c=verify()['cases'][case];args=[sys.executable,str(ROOT/'run_validated_structure_forecast.py'),'--raw',c['raw'],'--output',str(OUT/'raw'/case),
        '--start',str(c['start']),'--duration',str(c['duration']),'--dropout-mask','-1','--verify-frames']
    if not case.startswith('prefix'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,(case,r.returncode);check(case)

def check(case):
    c=verify()['cases'][case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');assert len(rr)==round(c['duration']*50);count=0
    for r in rr:
        q=r['frontends']['delayed_feedback'];w=q['learned_structure118'];v=w['validated122']
        assert not w['removed_one_observation'];assert w['max_event_t'] is None or w['max_event_t']<round(r['t']*1e6)
        if len(w['input_rotors'])==3:assert v['selection']['validation'] is None
        for method in METHODS:
            if method in q['coefficients']:assert q['coefficients'][method]['acceleration']==[0.,0.]
    if case.startswith('prefix'):
        full=rows(OUT/'raw/extension1/observations_forecasts.jsonl')[:len(rr)]
        for a,b in zip(rr,full):
            qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback']
            for key in ['coefficients','predictions','learned_structure118']:assert stable(qa[key])==stable(qb[key]),(a['t'],key)
            assert a['event_fields_sha256']==b['event_fields_sha256']
    else:
        cap=cv2.VideoCapture(str(OUT/'raw'/case/'forecasts.mp4'))
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==len(rr)
    if case=='negative5':assert all(not r['frontends']['delayed_feedback']['valid'] for r in rr)
    completion=json.loads((OUT/'raw'/case/'completion.json').read_text('utf-8'));assert completion['source_hashes_unchanged'] and not completion['labels_read']
    save(OUT/'raw'/case/'checks123.json',dict(case=case,frames=len(rr),actual_RAW=True,no_feature_dropout=True,all_current_cutoffs_valid=True,
        no_current_hidden_fourth=True,actual_prefix=case.startswith('prefix'),decoded_frames=count,source_unchanged=True,labels_read=False))
    print(json.dumps(dict(case=case,frames=len(rr),checked=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:globals()[a.action](a.case)
