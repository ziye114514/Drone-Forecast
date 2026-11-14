"""Freeze and verify actual RAW deployment of the partial-observation pilot."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage119')]
import run_partial_structure_forecast as entry
from study119 import OUT as CACHE,stable,rows,save,sha
OUT=ROOT.parent/'output/stage120_RAW_partial_structure_20261023'
PREVIOUS=ROOT.parent/'output/stage118_RAW_learned_structure_20261022/raw'
CASES=dict(clear6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=4.,dropout=0),
    prefix_clear6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=2.,dropout=0),
    negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6,dropout=-1))

def sources():
    ns=entry.build();result=ns['fingerprints']();ns.clear();result[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__);return result

def freeze():
    assert not OUT.exists();save(OUT/'protocol120.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),cases=CASES,
        design='Actual RAW/oldNN weights only. Preserve original64, add body_hold/affine3/conic3/guarded3. Same feature removal mask0 as119, first200 ticks; independent100 prefix and30 negative. Test engineering equivalence, NOT a blind new data efficacy test.',
        checks='Old64 coefficients/predictions exact118; new4 and internal model/filter/learning states exact119; parent body-only exact isolated worker; no output feature cache/truth reads; cutoff and actual prefix; decode all previews.',
        limits='Feature removal only. No realistic physical occlusion, no realCOM/RPM/attitude truth, no extreme-side recovery or realtime claim. One partial frame does not establish reliable persistent tracking.',default_promoted=False))

def verify():
    p=json.loads((OUT/'protocol120.json').read_text('utf-8'));assert p['sources']==sources();return p

def check(case):
    cfg=verify()['cases'][case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');n=len(rr);assert n==round(cfg['duration']*50)
    base=case.removeprefix('prefix_');old=rows(PREVIOUS/base/'observations_forecasts.jsonl')[:n]
    cache=rows(CACHE/f'{base}_mask0'/'records.jsonl')[:n]
    oldnames=set()
    for r,o,c in zip(rr,old,cache):
        qa=r['frontends']['delayed_feedback'];qo=o['frontends']['delayed_feedback'];worker=qa['learned_structure118']
        names=[k for k in qo['coefficients'] if k not in ['model_free20_kf118','model_fixed_kf118','model_fixed_guarded_kf118']]
        oldnames.update(names)
        for field in ['coefficients','predictions']:
            for k,v in qo[field].items():
                if k.startswith('model_') and k.endswith('118'):continue
                assert qa[field][k]==v,(case,r['t'],field,k)
        assert r['body_observation']==o['body_observation']
        assert worker['input_rotors']==c['input_rotors'] and worker['removed_one_observation']==c['removed_one_observation']
        for mode,name in zip(['body_hold','affine3','conic3','guarded3'],entry.NAMES120):
            assert stable(worker['outputs'][mode])==stable(c['outputs'][mode]),(case,r['t'],mode,'new model differs cached pilot')
            coefficient=c['outputs'][mode]['forecast']['coefficients'].get('center_bias_fixed107')
            if qa['valid'] and coefficient is not None:assert qa['coefficients'][name]==coefficient
        assert worker['max_event_t'] is None or worker['max_event_t']<round(r['t']*1e6)
    if base=='negative5':assert all(not r['frontends']['delayed_feedback']['valid'] for r in rr)
    else:assert len(oldnames)==64,('union_of_initialized_original_branches',len(oldnames))
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
    save(OUT/'raw'/case/'checks120.json',dict(case=case,frames=n,actual_RAW=True,old64_exact118=True,new4_exact119=True,
        current_cutoffs_valid=True,actual_RAW_prefix=case.startswith('prefix'),decoded_frames=decoded,default_promoted=False))
    print(json.dumps(dict(case=case,frames=n,old64_exact=True,new4_exact=True)),flush=True)

def run(case):
    c=verify()['cases'][case];args=[sys.executable,str(ROOT/'run_partial_structure_forecast.py'),'--raw',c['raw'],'--output',str(OUT/'raw'/case),
        '--start',str(c['start']),'--duration',str(c['duration']),'--dropout-mask',str(c['dropout']),'--verify-frames']
    if not case.startswith('prefix'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,(case,r.returncode);check(case)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:globals()[a.action](a.case)
