"""Predeclared visible-interval follow-up after the94 empty midpoint control."""
from pathlib import Path
from types import FunctionType
import argparse,datetime,inspect,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage94'),str(ROOT/'research/stage88')]
import study94 as old
from inventory88 import load,save,sha,parse,unique_rows
import score88
OUT=ROOT.parent/'output/stage95_official_visible_interval_20261022'


def sources():return {str(p):sha(p) for p in [Path(__file__),Path(old.__file__),ROOT/'research/stage88/paired88.py',ROOT/'research/stage93/online93.py',ROOT/'research/stage92/router92.py']}


def freeze():
    assert not OUT.exists();empty=load(old.OUT/'fred/sequence194/completion.json');assert empty['position_updates']==0
    labels,bad=parse((old.OUT/'input/coordinates.txt').read_text(encoding='utf-8-sig'));assert not bad;labels=unique_rows(labels)
    runs=[];begin=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]['t']-labels[i-1]['t']>.101:
            runs.append(dict(start=labels[begin]['t'],end=labels[i-1]['t'],labels=i-begin));begin=i
    chosen=max(runs,key=lambda r:(r['end']-r['start'],-r['start']))
    start=np.ceil(chosen['start']*50)/50;duration=np.floor((chosen['end']-start)*50)/50
    assert duration>1.4
    save(OUT/'protocol_feature95.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        original94_sources=old.sources(),start_s=float(start),duration_s=float(duration),step_us=20000,
        interval_rule='Longest contiguous native annotation run, gap<=.101s; earliest wins equal lengths;20ms start/end quantization. No forecast errors used.',
        all_annotation_runs=runs,selected_run=chosen,
        original_empty_midpoint_sha256=sha(old.OUT/'fred/sequence194/completion.json'),
        interpretation='Visibility/size-conditioned follow-up;94 empty midpoint preserved. Same model parameters, not a blind/full benchmark.',
        event_input=load(old.OUT/'download94.json'),scoring='Same88 actual-label ADE/FDE;100ms anchors start+.6..end-.4; no interpolation.',
        RAW_only_deployment=False,default_promoted=False))
    save(OUT/'download94.json',load(old.OUT/'download94.json'))
    (OUT/'input').mkdir(exist_ok=True);(OUT/'input/coordinates.txt').write_bytes((old.OUT/'input/coordinates.txt').read_bytes())
    print('95 visible interval frozen',start,start+duration,flush=True)


def verify():
    p=load(OUT/'protocol_feature95.json');assert p['source_hashes']==sources();assert p['original94_sources']==old.sources();return p


def run(prefix=False):
    verify();function=FunctionType(old.run.__code__,dict(old.run.__globals__,OUT=OUT,verify=verify,sources=sources))
    return function(prefix)


def score():
    p=verify();assert not (OUT/'scores95.json').exists();assert load(OUT/'fred/prefix194/completion.json')['actual_event_prefix_exact']
    (OUT/'annotations').mkdir(exist_ok=True);(OUT/'annotations/194.txt').write_bytes((OUT/'input/coordinates.txt').read_bytes())
    source=inspect.getsource(score88.fred_score);old_text='t<20.6-1e-8 or t>49.6+1e-8'
    replacement=f't<{p["start_s"]+.6}-1e-8 or t>{p["start_s"]+p["duration_s"]-.4}+1e-8'
    assert source.count(old_text)==1;source=source.replace(old_text,replacement)
    namespace=dict(score88.fred_score.__globals__,OUT=OUT,METHODS=old.METHODS)
    exec(compile(source,'<95 unchanged88 scoring with predeclared visible interval>','exec'),namespace)
    protocol=load(ROOT.parent/'output/stage88_causal_event_controls_20261022/protocol88.json');protocol['fred_cases']=[194]
    detail,audit=namespace['fred_score'](protocol);table=score88.summarise(detail)
    save(OUT/'endpoint_details95.json',detail);save(OUT/'scores95.json',dict(table=table,audit=audit,
        interval_selected_by_labels_visibility=True,no_tuning_on_forecast_error=True,same_parameters_as88_90_92=True,
        mechanical_RPM_truth=False,RAW_only_deployment=False,default_promoted=False))
    print('95 visible native diagnostic',[(r['method'],r['valid'],r['mean_fde']) for r in table if r['horizon']==.4 and r['group']=='all'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','score']);a=p.parse_args()
    if a.action=='prefix':run(True)
    else:globals()[a.action]()
