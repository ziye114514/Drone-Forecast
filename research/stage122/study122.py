"""Actual RAW69-forecast equivalence, cache-read ban and independent prefix."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research/stage120')]
import run_validated_structure_forecast as entry
from study120 import CASES,OUT as PREVIOUS,rows,save,sha,stable
OUT=ROOT.parent/'output/stage122_RAW_validated_structure_20261023'
CACHE=ROOT.parent/'output/stage121_causal_model_validation_20261023'

def sources():
    ns=entry.build();r=ns['fingerprints']();ns.clear();r[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__);return r
def freeze():
    assert not OUT.exists();save(OUT/'protocol122.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),cases=CASES,
        design='Same200 clear masked/100 independent prefix/30 negative actualRAW ticks as120. Preserve120 original64+four controls and add121 validated observer. No independent new efficacy dataset, no threshold changes.',
        checks='All old68 present coefficients/predictions exact120, all four model/filter states exact120, selector state/coefficients exact cached121, no feature/truth reads in worker, cutoff limits, true input prefix, negative, full decode.',default_promoted=False))
def verify():
    p=json.loads((OUT/'protocol122.json').read_text('utf-8'));assert p['sources']==sources();return p
def check(case):
    cfg=verify()['cases'][case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');old=rows(PREVIOUS/'raw'/case/'observations_forecasts.jsonl');n=len(rr);assert n==len(old)==round(cfg['duration']*50)
    cached=json.loads((CACHE/'clear6_mask0.json').read_text('utf-8'))['records'][:n] if 'clear6' in case else None;names=set()
    for i,(r,o) in enumerate(zip(rr,old)):
        qa=r['frontends']['delayed_feedback'];qo=o['frontends']['delayed_feedback'];worker=qa['learned_structure118'];oldworker=qo['learned_structure118']
        for field in ['coefficients','predictions']:
            for k,v in qo[field].items():assert qa[field][k]==v,(case,r['t'],field,k)
        names.update(qo['coefficients']);assert stable(worker['outputs'])==stable(oldworker['outputs'])
        assert r['body_observation']==o['body_observation'] and worker['input_rotors']==oldworker['input_rotors']
        if cached is not None:assert stable(worker['validated122'])==stable(cached[i]['output']),(case,r['t'],'selector state differs121')
        assert worker['max_event_t'] is None or worker['max_event_t']<round(r['t']*1e6)
        if len(worker['input_rotors'])==3:
            selection=worker['validated122']['selection'];assert selection['validation'] is None and not selection['uses_four_observations_when_three_available']
    if case=='negative5':assert all(not r['frontends']['delayed_feedback']['valid'] for r in rr)
    else:assert len(names)==68
    if case.startswith('prefix'):
        full=rows(OUT/'raw/clear6/observations_forecasts.jsonl')[:n]
        for a,b in zip(rr,full):
            qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback']
            assert qa['coefficients']==qb['coefficients'] and qa['predictions']==qb['predictions'] and stable(qa['learned_structure118'])==stable(qb['learned_structure118'])
    decoded=0
    if not case.startswith('prefix'):
        import cv2
        cap=cv2.VideoCapture(str(OUT/'raw'/case/'forecasts.mp4'))
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
        cap.release();assert decoded==n
    save(OUT/'raw'/case/'checks122.json',dict(case=case,frames=n,actual_RAW=True,old68_exact120=True,new_selector_exact121=cached is not None,
        no_fourth_read_during_three=True,current_cutoff_valid=True,actual_prefix=case.startswith('prefix'),decoded_frames=decoded,default_promoted=False))
    print(json.dumps(dict(case=case,frames=n,old68_exact=True,selector_exact121=cached is not None)),flush=True)
def run(case):
    c=verify()['cases'][case];args=[sys.executable,str(ROOT/'run_validated_structure_forecast.py'),'--raw',c['raw'],'--output',str(OUT/'raw'/case),'--start',str(c['start']),'--duration',str(c['duration']),'--dropout-mask',str(c['dropout']),'--verify-frames']
    if not case.startswith('prefix'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,(case,r.returncode);check(case)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:globals()[a.action](a.case)
