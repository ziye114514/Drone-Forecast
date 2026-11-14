"""Fixed projected-magnitude ablation on all own caches and same native194."""
from pathlib import Path
from types import FunctionType
import argparse,datetime,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage88'),str(ROOT/'research/stage94')]
from inventory88 import load,save,sha
from projected97 import ProjectedPaired
from paired88 import prior,minimal_q
import score88,study94
OUT=ROOT.parent/'output/stage97_projected_axis_magnitude_20261022'
AUTO=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
PREV=ROOT.parent/'output/stage95_official_visible_interval_20261022'
CASES=['clear6','side6','train3','extra1','extra2','extra4']


def sources():return {str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('projected97.py'),Path(study94.__file__),ROOT/'research/stage88/paired88.py',ROOT/'research/stage69/force69.py']}


def freeze():
    assert not OUT.exists();p=load(PREV/'protocol_feature95.json')
    save(OUT/'protocol_feature97.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        start_s=p['start_s'],duration_s=p['duration_s'],step_us=20000,
        related_work='https://arxiv.org/html/2604.18289v1; position thrust uses full3D unit normal, not normalised2D component.',
        hypothesis='69 normalised projected normal to unit2D, dropping tilt-dependent projected magnitude. Retain existing weak3D-normal xy length; assume1 for near-edge-on body line.',
        controls='Unit-reference and projected variants share original current publication clock; all event/constant/zero/lag/past-shuffle and learning constants retained.',
        scope='Known development recordings, not novel algorithm or new calibrated tilt measurement. No parameter search or default promotion.'))
    save(OUT/'download94.json',load(study94.OUT/'download94.json'));(OUT/'input').mkdir(exist_ok=True)
    (OUT/'input/coordinates.txt').write_bytes((study94.OUT/'input/coordinates.txt').read_bytes())
    print('97 projected-magnitude/control-clock ablation frozen',flush=True)


def verify():
    p=load(OUT/'protocol_feature97.json');assert p['source_hashes']==sources();return p


def self_run():
    verify();old_scores=load(AUTO/'scoring/analysis.json')['details'];details=[]
    for case in CASES:
        model=ProjectedPaired();values=[]
        for r in score88.rows(AUTO/case/'observations_forecasts.jsonl'):
            q=r['frontends']['delayed_feedback'];v=model.step(minimal_q(q),r['observation'],r['body_axis'],r['event_activity'],q['increment_diagnostic']['direction']['quality'])
            ref=model.reference_last
            if q['valid']:
                assert ref['coefficients']['event']==q['coefficients']['mmae_state_fresh']
                assert ref['coefficients']['constant']==q['coefficients']['mmae_state_fresh_constant']
            values.append(v)
        lookup={round(r['t'],8):r for r in values};save(OUT/'self'/case/'forecasts.json',values)
        for r in old_scores:
            if r['case']!=case or r['frontend']!='event_force' or r['model']!='mmae_state_fresh' or r['nominal_h']!=.4 or r['target_state']!='present':continue
            v=lookup.get(round(r.get('anchor_t',-100),8));valid=bool(v and v['valid'])
            for m in ['cv','trusted_acc','event','constant','zero','lag200','past_shuffle101','past_shuffle202','past_shuffle303']:
                d=dict(case=case,method=m,horizon=.4,valid=valid)
                if valid:d['fde']=float(__import__('numpy').linalg.norm(__import__('numpy').array(prior.evaluate(v['coefficients'][m],r['actual_h']))-r['target_center']))
                details.append(d)
        print(case,'97 projected/control clock complete',flush=True)
    save(OUT/'self_details97.json',details);save(OUT/'self_scores97.json',dict(table=score88.summarise(details),unit_reference_exact78=True,clock_matched=True,default_promoted=False))


def run(prefix=False):
    verify();ns=dict(study94.run.__globals__,OUT=OUT,verify=verify,sources=sources,Paired=ProjectedPaired)
    return FunctionType(study94.run.__code__,ns)(prefix)


def score():
    p=verify();assert not (OUT/'scores97.json').exists();assert load(OUT/'fred/prefix194/completion.json')['actual_event_prefix_exact']
    (OUT/'annotations').mkdir(exist_ok=True);(OUT/'annotations/194.txt').write_bytes((OUT/'input/coordinates.txt').read_bytes())
    text=__import__('inspect').getsource(score88.fred_score);a='t<20.6-1e-8 or t>49.6+1e-8';b=f't<{p["start_s"]+.6}-1e-8 or t>{p["start_s"]+p["duration_s"]-.4}+1e-8'
    assert text.count(a)==1;ns=dict(score88.fred_score.__globals__,OUT=OUT,METHODS=study94.METHODS)
    exec(compile(text.replace(a,b),'<97 fixed95 interval scoring>','exec'),ns)
    protocol=load(ROOT.parent/'output/stage88_causal_event_controls_20261022/protocol88.json');protocol['fred_cases']=[194]
    details,audit=ns['fred_score'](protocol)
    save(OUT/'endpoint_details97.json',details);save(OUT/'scores97.json',dict(table=score88.summarise(details),audit=audit,same_publication_clock=True,default_promoted=False))
    print('97 projected-magnitude scores',[(r['method'],r['mean_fde']) for r in score88.summarise(details) if r['horizon']==.4 and r['group']=='all'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','self','run','prefix','score']);a=p.parse_args()
    if a.action=='prefix':run(True)
    elif a.action=='self':self_run()
    else:globals()[a.action]()
