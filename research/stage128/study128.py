"""Freeze and verify actualRAW independent-support augmentation."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage126')]
import run_independent_support_forecast as entry
from study126 import CASES,REF,CACHE,rows,sha,save,stable,OUT as PRIOR
OUT=ROOT.parent/'output/stage128_RAW_independent_support_20261023'
def sources():
    ns=entry.build();r=ns['fingerprints']();ns.clear();r[str(Path(__file__).relative_to(ROOT))]=sha(__file__);return r
def freeze():
    assert not OUT.exists();save(OUT/'protocol128.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),cases=CASES,
        design='Same126 planned cases, RAW only71 branches. Preserve old69 exact123 and126 hybrid exact125. Independent occupancy model plus shared127 KF exact current cached127 observations. No future labels/forced4/relaxed gates.',
        history='126 simple geometry-level fallback has no current-coverage gain in125; separately completed126 natural6. Remaining126 runs superseded by this128 which preserves it as a control.127 cached clear current236->244, first22.80->22.68, before any futureBOX scoring.',
        checks='Original69 exact123; old70 exact actual126 on overlapping natural6; all occupancy observations exact124, simple hybrid exact125, new independent output exact127, active60 actualRAW prefix, negative, all decode/cutoff/source hashes.',default_promoted=False))
def verify():
    p=json.loads((OUT/'protocol128.json').read_text('utf-8'));assert p['sources']==sources();return p
def check(case):
    c=verify()['cases'][case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');ref=rows(REF/'raw'/c['reference']/'observations_forecasts.jsonl')[:len(rr)]
    inp=rows(ROOT.parent/'output/stage124_occupancy_support_20261023'/c['reference']/'records.jsonl')[:len(rr)];hybrid=json.loads((CACHE/(c['reference']+'.json')).read_text('utf-8'))[:len(rr)]
    independent=json.loads((ROOT.parent/'output/stage127_independent_support_20261023'/(c['reference']+'.json')).read_text('utf-8'))[:len(rr)];names=set()
    prior=rows(PRIOR/'raw/natural6/observations_forecasts.jsonl')[:len(rr)] if case in ['natural6','prefix_natural6'] else None
    assert len(rr)==len(ref)==round(c['duration']*50)
    for i,(r,o,s,h,new) in enumerate(zip(rr,ref,inp,hybrid,independent)):
        q=r['frontends']['delayed_feedback'];old=o['frontends']['delayed_feedback'];w=q['learned_structure118'];added=w['support126'];result=w['support128']
        assert r['body_observation']==o['body_observation'] and r['body_axis']==o['body_axis']
        for key in ['coefficients','predictions']:
            for name,value in old[key].items():assert q[key][name]==value,(case,r['t'],key,name);names.add(name)
            if prior is not None:
                for name,value in prior[i]['frontends']['delayed_feedback'][key].items():assert q[key][name]==value,(case,r['t'],'old70',name)
        stripped={k:v for k,v in w.items() if k not in ['support126','support128']};stripped['coefficients']={k:v for k,v in stripped['coefficients'].items() if k not in ['support_hybrid_kf126','support_independent_kf128']}
        assert stable(stripped)==stable(old['learned_structure118']),(r['t'],'old122 worker changed')
        assert added['occupancy_geometry']==s['outputs']['occupancy_roi']['geometry'],(r['t'],'occupancy differs124')
        assert stable(added['hybrid'])==stable(h['output']),(r['t'],'simple hybrid differs125')
        assert stable(result['output'])==stable(new['output']),(r['t'],'independent differs127')
        assert stable(result['occupancy_model'])==stable(s['outputs']['occupancy_roi']['center']),(r['t'],'independent model differs124')
        assert added['max_event_t'] is None or added['max_event_t']<round(r['t']*1e6)
        assert not added['cache_read'] and result['no_cache_reads']
    if case=='negative5':assert all(not r['frontends']['delayed_feedback']['valid'] for r in rr)
    else:assert len(names)==69
    if case.startswith('prefix'):
        full=rows(OUT/'raw/natural6/observations_forecasts.jsonl')[:len(rr)]
        for a,b in zip(rr,full):
            assert a['event_fields_sha256']==b['event_fields_sha256'];qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback']
            for key in ['coefficients','predictions','learned_structure118']:assert stable(qa[key])==stable(qb[key])
    decoded=0
    if not case.startswith('prefix'):
        import cv2
        cap=cv2.VideoCapture(str(OUT/'raw'/case/'forecasts.mp4'))
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
        cap.release();assert decoded==len(rr)
    completion=json.loads((OUT/'raw'/case/'completion.json').read_text('utf-8'));assert completion['source_hashes_unchanged'] and not completion['labels_read']
    save(OUT/'raw'/case/'checks128.json',dict(case=case,frames=len(rr),actual_RAW=True,original69_exact123=case!='negative5',old70_exact_actual126=prior is not None,old122_worker_exact=True,
        occupancy_exact124=True,simple_hybrid_exact125=True,independent_model_exact124=True,independent_fusion_exact127=True,cutoff_and_no_cache_reads=True,active_actual_prefix=case.startswith('prefix'),decoded_frames=decoded,labels_read=False,source_hashes_unchanged=True))
    print(json.dumps(dict(case=case,frames=len(rr),checked=True)),flush=True)
def run(case):
    c=verify()['cases'][case];args=[sys.executable,str(ROOT/'core/run_independent_support_forecast.py'),'--raw',c['raw'],'--output',str(OUT/'raw'/case),'--start',str(c['start']),'--duration',str(c['duration']),'--verify-frames']
    if not case.startswith('prefix'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as log:r=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,(case,r.returncode);check(case)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check']);p.add_argument('--case',choices=CASES);a=p.parse_args();globals()[a.action](a.case) if a.action!='freeze' else freeze()
