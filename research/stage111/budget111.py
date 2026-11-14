"""Current-geometry diagnostic: distinguish optimizer exhaustion from bad support."""
from pathlib import Path
import argparse,copy,datetime,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage110'),str(ROOT/'research/stage104')]
from study110 import OUT as INPUT,save,sha,rows
import rigid104,geometry58

OUT=ROOT.parent/'output/stage111_conic_budget_diagnostic_20261022'
SOURCE=INPUT/'continuous6/records.jsonl'

def sources():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'research/stage104/rigid104.py',ROOT/'research/stage58/geometry58.py']}

def freeze():
    assert not OUT.exists()
    save(OUT/'protocol111.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),input_sha256=sha(SOURCE),
        design='On all continuous6 merge_safe frames whose unchanged104 measurement failed ONLY because conic optimizer exhausted80 evaluations, re-evaluate the same four current ellipses with320 evaluations. Keep residual/quality/conditioning/5-frame admission thresholds. Replay same104 state with these replacements. No source file edits or parameter sweep.',
        reason='110 C112.02 restores4 measured components but optimizer status0/nfev80. Need distinguish budget failure from ellipse/model inconsistency before designing new fitting.',
        current_only='All fits use the single current frame. No previous/future ellipses, manual hubs, future labels or score-based selection.',
        scope='Cached current-geometry diagnostic and model-state replay, no RAW rerun, no new forecasts, no claim of better pose/COM or real-time performance.320 is a fixed diagnostic, not a deployment promotion.',
        parameters=dict(old_max_nfev=80,diagnostic_max_nfev=320),default_promoted=False))

def stable(x):
    if isinstance(x,dict):return {k:stable(v) for k,v in x.items() if k not in ['elapsed_ms','processing_ms']}
    if isinstance(x,list):return [stable(v) for v in x]
    return x

def run():
    protocol=json.loads((OUT/'protocol111.json').read_text('utf-8'));assert protocol['sources']==sources() and protocol['input_sha256']==sha(SOURCE)
    rr=rows(SOURCE);measure=rigid104.measure_four;original_budget=geometry58.PARAMS['max_nfev'];assert original_budget==80
    replacements={};records=[]
    try:
        # Recompute C at80 before changing the budget, exact non-timing check.
        c=next(r for r in rr if abs(r['t']-112.02)<1e-8);base=c['outputs']['merge_safe']
        check=measure(base['geometry']['rotors'],c['body']);assert stable(check)==stable(base['center']['measurement'])
        geometry58.PARAMS['max_nfev']=320
        for r in rr:
            old=r['outputs']['merge_safe'];m=old['center']['measurement']
            if m['reason']!='conic_optimizer_failed':continue
            start=time.perf_counter();new=measure(old['geometry'].get('rotors',[]),r['body']);elapsed=(time.perf_counter()-start)*1000
            replacements[r['t']]=new;records.append(dict(t=r['t'],old=m,new=new,elapsed_ms=elapsed))
            if len(records)%10==0:print(f'111 current fits {len(records)}',flush=True)
    finally:geometry58.PARAMS['max_nfev']=original_budget
    # Only this isolated diagnostic substitutes measurements; old source and
    # all110 forecasts stay unchanged. No hidden point becomes an observation.
    current=[None];rigid104.measure_four=lambda rotors,body:copy.deepcopy(current[0])
    states=[]
    try:
        model=rigid104.PlanarCenter()
        for r in rr:
            current[0]=replacements.get(r['t'],r['outputs']['merge_safe']['center']['measurement'])
            _,d=model.step(r['t'],r['body'],r['outputs']['merge_safe']['geometry'].get('rotors',[]),r['axis'])
            states.append(dict(t=r['t'],diagnostic=d))
        model=rigid104.PlanarCenter()
        for r,s in zip(rr[:60],states[:60]):
            current[0]=replacements.get(r['t'],r['outputs']['merge_safe']['center']['measurement'])
            _,d=model.step(r['t'],r['body'],r['outputs']['merge_safe']['geometry'].get('rotors',[]),r['axis']);assert stable(d)==stable(s['diagnostic'])
    finally:rigid104.measure_four=measure
    summary=dict(frames=len(rr),refitted=len(records),new_valid=sum(r['new']['valid'] for r in records),
        still_optimizer_failed=sum(r['new']['reason']=='conic_optimizer_failed' for r in records),
        newly_inconsistent=sum(r['new']['reason']=='inconsistent_coplanar_circles' for r in records),
        initializations=sum(s['diagnostic']['initialized_now'] for s in states),current_centers=sum(s['diagnostic']['current'] for s in states),
        initial_times=[s['t'] for s in states if s['diagnostic']['initialized_now']],
        unchanged_source_files=protocol['sources']==sources(),cached_prefix_exact=True,max_nfev_restored=geometry58.PARAMS['max_nfev']==80)
    save(OUT/'diagnostic111.json',dict(summary=summary,records=records,states=states));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
