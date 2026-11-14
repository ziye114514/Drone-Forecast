"""Freeze and replay independent-support measurement fusion."""
from pathlib import Path
import argparse,collections,datetime,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage124'),str(ROOT/'research/stage127')]
from study124 import OUT as INPUT,CASES,rows,sha,save,stable
from independent127 import MeasurementFusion,PARAMS,NAMES
OUT=ROOT.parent/'output/stage127_independent_support_20261023'
def freeze():
    assert not OUT.exists();save(OUT/'protocol127.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),parameters=PARAMS,sources={p.name:sha(p) for p in [Path(__file__),ROOT/'research/stage127/independent127.py']},
        cases=list(CASES),scope='Cached current124 separately learned model observations. Rule fixed after125 no-coverage-gain but before reading futureBOX scores. Exposed development clips, no blind-test claim.',
        checks='Independent60 prefix, count-only coordinates numerically equivalent107, shared exact body control, no fake four or lowered lock/fit threshold.',default_promoted=False))
def verify():
    p=json.loads((OUT/'protocol127.json').read_text('utf-8'));assert p['sources']=={x.name:sha(x) for x in [Path(__file__),ROOT/'research/stage127/independent127.py']}
def run():
    verify();summary=[];maximum=0.
    for case in CASES:
        rr=rows(INPUT/case/'records.jsonl');f=MeasurementFusion();count_only=MeasurementFusion();output=[]
        for r in rr:
            c=r['outputs']['count_global']['center'];o=r['outputs']['occupancy_roi']['center'];a=f.step(r['t'],r['body'],r['axis'],c,o);output.append(dict(t=r['t'],output=a))
            b=count_only.step(r['t'],r['body'],r['axis'],c,dict(current=False));expected=r['outputs']['count_global']['forecast']['coefficients'].get(NAMES[2]);actual=b['forecast']['coefficients'].get(NAMES[2])
            assert (expected is None)==(actual is None)
            if expected is not None:
                error=max(abs(np.array(expected[k])-actual[k]).max() for k in ['position','velocity','acceleration']);maximum=max(maximum,float(error));assert error<1e-8
            assert a['forecast']['coefficients'].get(NAMES[0])==r['outputs']['count_global']['forecast']['coefficients'].get(NAMES[0])
        save(OUT/(case+'.json'),output);prefix=MeasurementFusion()
        for r,o in zip(rr[:60],output[:60]):
            a=prefix.step(r['t'],r['body'],r['axis'],r['outputs']['count_global']['center'],r['outputs']['occupancy_roi']['center']);assert stable(a)==stable(o['output'])
        summary.append(dict(case=case,frames=len(rr),chosen=dict(collections.Counter(o['output']['chosen'] for o in output)),current=sum(o['output']['center']['current'] for o in output),
            first_current=next((o['t'] for o in output if o['output']['center']['current']),None),prefix_exact=True))
    save(OUT/'availability127.json',dict(table=summary,count_only_max_abs_coefficient_difference=maximum,count_only_tolerance=1e-8,body_controls_exact=True));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
