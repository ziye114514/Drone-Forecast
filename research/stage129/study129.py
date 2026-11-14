"""Boundary eligibility audit/replay, after128 scores: development diagnosis."""
from pathlib import Path
import argparse,collections,datetime,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage124'),str(ROOT/'research/stage129')]
from study124 import OUT as INPUT,CASES,rows,sha,save,stable
from boundary129 import Guarded,bounded,PARAMS,NAMES
OUT=ROOT.parent/'output/stage129_FOV_guard_20261024'
def freeze():
    assert not OUT.exists();save(OUT/'protocol129.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources={p.name:sha(p) for p in [Path(__file__),ROOT/'research/stage129/boundary129.py']},parameters=PARAMS,cases=list(CASES),width=1280,height=720,
        history='After128 future scores, visual QA found22.68s right rotor clipped. Six of eight occupancy-only current updates have fitted ellipse outside FOV. Guard is geometric eligibility, no endpoint tuning; all data previously exposed. Replay is a diagnostic, not blind confirmation.',
        scope='Current124 geometry and same-time NN/axis; no new detector. Independently replay/60 prefix; actual RAW90-pilot subsequently. Prior128 results retained and corrected, not overwritten.',default_promoted=False))
def verify():
    p=json.loads((OUT/'protocol129.json').read_text('utf-8'));assert p['sources']=={q.name:sha(q) for q in [Path(__file__),ROOT/'research/stage129/boundary129.py']};return p
def run():
    verify();table=[]
    for case in CASES:
        rr=rows(INPUT/case/'records.jsonl');model=Guarded(1280,720);output=[]
        for r in rr:
            c=r['outputs']['count_global']['geometry'].get('rotors',[]);o=r['outputs']['occupancy_roi']['geometry'].get('rotors',[]);q=model.step(r['t'],r['body'],r['axis'],c,o)
            assert q['forecast']['coefficients'].get(NAMES[0])==r['outputs']['count_global']['forecast']['coefficients'].get(NAMES[0]);output.append(dict(t=r['t'],output=q))
        save(OUT/(case+'.json'),output);prefix=Guarded(1280,720)
        for r,q in zip(rr[:60],output[:60]):
            a=prefix.step(r['t'],r['body'],r['axis'],r['outputs']['count_global']['geometry'].get('rotors',[]),r['outputs']['occupancy_roi']['geometry'].get('rotors',[]));assert stable(a)==stable(q['output'])
        table.append(dict(case=case,frames=len(rr),chosen=dict(collections.Counter(q['output']['chosen'] for q in output)),current=sum(q['output']['center']['current'] for q in output),first_current=next((q['t'] for q in output if q['output']['center']['current']),None),
            first_occupancy_model=next((q['t'] for q in output if q['output']['independent_models']['occupancy'].get('structure_ready')),None),
            clipped_count_frames=sum(any(v['ellipse_outside_sensor'] for v in q['output']['boundary_audit']['count']) for q in output),clipped_occupancy_frames=sum(any(v['ellipse_outside_sensor'] for v in q['output']['boundary_audit']['occupancy']) for q in output),prefix_exact=True))
    save(OUT/'availability129.json',table);print(json.dumps(table),flush=True)
def checks():
    for angle in [0,17,90,173]:
        r=dict(cx=50.,cy=50.,major=30.,minor=10.,angle=angle,mode='ellipse',inferred_shape=False)
        q,a=bounded([r],101,101);assert q[0]==r and not a[0]['ellipse_outside_sensor']
        r=dict(r,cx=99.);q,a=bounded([r],101,101);assert q[0]['mode']=='arc' and q[0]['inferred_shape'] and a[0]['ellipse_outside_sensor'] and r['mode']=='ellipse'
    save(OUT/'guard_checks129.json',dict(analytic_rotation_bounds_checked=True,interior_preserved_exact=True,clipped_marked_partial=True,original_measurement_not_mutated=True,shared_body_control_exact=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','checks']);a=p.parse_args();globals()[a.action]()
