"""Freeze/run/verify67 realRAW branches with independent old/cache controls."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage117'),str(ROOT/'research/stage113')]
import run_learned_structure_forecast as entry
from study117 import OUT as CACHE,stable,rows,save,sha
OUT=ROOT.parent/'output/stage118_RAW_learned_structure_20261022'
CASES=dict(clear6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.),
    continuous6=dict(raw='E:/dataset/train/6/media/event.raw',start=110.,duration=3.),
    prefix_clear6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=1.2),
    prefix_continuous6=dict(raw='E:/dataset/train/6/media/event.raw',start=110.,duration=1.2),
    negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6))
REFERENCE=ROOT.parent/'output/stage107_center_bias_filter_20261022/raw'
def sources():
    ns=entry.build();r=ns['fingerprints']();ns.clear();r[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__);return r
def freeze():
    assert not OUT.exists();save(OUT/'protocol118.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),cases=CASES,
        design='Actual singleRAW-input full118 NN/body/oldrotor10/old64 plus isolated current20 three structural forecasts. Preserve original64 coefficients and predictions exactly107; new worker model state exactly117 cache. Isolation prevents extra ellipse fits advancing old OpenCV RNG.',
        choice='Integrate the structurally faithful117 variant as experimental only. Clear sparse0.4s local difference16.8426→16.7800 is tiny, not established independent gain; original fusion13.7081 remains stronger. Failed113/115 variants not integrated.',
        inputs='RAW and pre-existing detector weights only in execution. RAW geometry worker forbids output cache/truth reads and advances only requested cutoffs. Offline comparison reads references after child CLI exits.',
        tests='Original64 exact, three new model/learning/filter states exact117, original bodyKF equals worker body-only. Actual independent truncatedRAW prefixes, negative no valid forecasts, decode every emitted preview frame.',default_promoted=False))
def verify():
    p=json.loads((OUT/'protocol118.json').read_text('utf-8'));assert p['sources']==sources();return p
def run(case):
    cfg=verify()['cases'][case];args=[sys.executable,str(ROOT/'run_learned_structure_forecast.py'),'--raw',cfg['raw'],'--output',str(OUT/'raw'/case),
        '--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if not case.startswith('prefix'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as log:
        r=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,(case,r.returncode);check(case)
def check(case):
    cfg=verify()['cases'][case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');base=case.removeprefix('prefix_');n=len(rr);assert n==round(cfg['duration']*50)
    old=rows(REFERENCE/base/'observations_forecasts.jsonl')[:n];cached=rows(CACHE/base/'records.jsonl')[:n]
    names={'free_structure':'model_free20_kf118','fixed_structure':'model_fixed_kf118','fixed_guarded':'model_fixed_guarded_kf118'}
    for r,o,c in zip(rr,old,cached):
        qa=r['frontends']['delayed_feedback'];qo=o['frontends']['delayed_feedback'];worker=qa['learned_structure118']
        assert qa['valid']==qo['valid']
        for field in ['coefficients','predictions']:
            for name,value in qo[field].items():assert qa[field][name]==value,(case,r['t'],field,name)
        assert r['body_observation']==o['body_observation']
        for mode,name in names.items():
            assert stable(worker['outputs'][mode])==stable(c['outputs'][mode]),(case,r['t'],mode,'new worker differs cache')
            coef=c['outputs'][mode]['forecast']['coefficients'].get('center_bias_fixed107')
            if qa['valid'] and coef is not None:assert qa['coefficients'][name]==coef
        assert worker['max_event_t'] is None or worker['max_event_t']<round(r['t']*1e6)
    if base=='negative5':assert all(not r['frontends']['delayed_feedback']['valid'] for r in rr)
    if case.startswith('prefix'):
        full=rows(OUT/'raw'/base/'observations_forecasts.jsonl')[:n]
        for a,b in zip(rr,full):
            qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback']
            assert qa['coefficients']==qb['coefficients'] and qa['predictions']==qb['predictions']
            assert stable(qa['learned_structure118'])==stable(qb['learned_structure118'])
    decoded=0
    if not case.startswith('prefix'):
        import cv2
        cap=cv2.VideoCapture(str(OUT/'raw'/case/'forecasts.mp4'))
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
        cap.release();assert decoded==n
    save(OUT/'raw'/case/'checks118.json',dict(case=case,frames=n,actual_RAW=True,original64_exact107=True,new3_and_model_states_exact117=True,
        current_cutoffs_valid=True,actual_RAW_prefix=case.startswith('prefix'),decoded_frames=decoded,default_promoted=False))
    print(json.dumps(dict(case=case,frames=n,original64_exact=True,new3_exact=True)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:globals()[a.action](a.case)
