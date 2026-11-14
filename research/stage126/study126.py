"""Actual RAW-only complementary-support deployment and equivalence checks."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage124')]
import run_support_forecast as entry
from study124 import sha,save,rows,stable
OUT=ROOT.parent/'output/stage126_RAW_complementary_support_20261023'
REF=ROOT.parent/'output/stage123_frozen_extension_20261023'
CACHE=ROOT.parent/'output/stage125_complementary_support_20261023'
CASES=dict(natural6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=8.,reference='natural6'),
    extension2=dict(raw='E:/dataset/train/2/aligned/event.raw',start=40.,duration=6.,reference='extension2'),
    negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6,reference='negative5'),
    prefix_natural6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=1.2,reference='natural6'))
def sources():
    ns=entry.build();r=ns['fingerprints']();ns.clear();r[str(Path(__file__).relative_to(ROOT))]=sha(__file__);return r
def freeze():
    assert not OUT.exists();save(OUT/'protocol126.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=CASES,sources=sources(),
        design='RAW and fixed old NN weights only. Original122 separate worker, new occupancy RAW worker and same current body/axis; add one69->70 forecast, preserve every old coefficient. Rule frozen125; same development intervals. No threshold/model/5-frame-lock change.',
        checks='All old69 branches and current body/axis/rotors exact123, new occupancy exact124 and hybrid exact125; independent active60 RAW prefix, negative, no output-cache/truth reads, eventcutoffs and all-frame decode.',
        evaluation='Only after all prediction jobs use sparse old manual futureBOX centers. New scores are development diagnostics, not COM or held-out efficacy.',default_promoted=False))
def verify():
    p=json.loads((OUT/'protocol126.json').read_text('utf-8'));assert p['sources']==sources();return p
def check(case):
    c=verify()['cases'][case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');ref=rows(REF/'raw'/c['reference']/'observations_forecasts.jsonl')[:len(rr)]
    inp=rows(ROOT.parent/'output/stage124_occupancy_support_20261023'/c['reference']/'records.jsonl')[:len(rr)];cached=json.loads((CACHE/(c['reference']+'.json')).read_text('utf-8'))[:len(rr)];names=set()
    assert len(rr)==len(ref)==round(c['duration']*50)
    for r,o,s,h in zip(rr,ref,inp,cached):
        q=r['frontends']['delayed_feedback'];old=o['frontends']['delayed_feedback'];w=q['learned_structure118'];added=w['support126']
        assert r['body_observation']==o['body_observation'] and r['body_axis']==o['body_axis']
        for key in ['coefficients','predictions']:
            for name,value in old[key].items():assert q[key][name]==value,(case,r['t'],key,name);names.add(name)
        stripped={k:v for k,v in w.items() if k!='support126'};stripped['coefficients']={k:v for k,v in stripped['coefficients'].items() if k!='support_hybrid_kf126'}
        assert stable(stripped)==stable(old['learned_structure118']),(r['t'],'old worker changed')
        assert added['occupancy_geometry']==s['outputs']['occupancy_roi']['geometry'],(r['t'],'new geometry differs124')
        assert stable(added['hybrid'])==stable(h['output']),(r['t'],'hybrid differs125')
        assert added['max_event_t'] is None or added['max_event_t']<round(r['t']*1e6)
        assert not added['cache_read'] and added['no_future_hint']
    if case=='negative5':assert all(not r['frontends']['delayed_feedback']['valid'] for r in rr)
    else:assert len(names)==69
    if case.startswith('prefix'):
        full=rows(OUT/'raw/natural6/observations_forecasts.jsonl')[:len(rr)]
        for a,b in zip(rr,full):
            assert a['event_fields_sha256']==b['event_fields_sha256']
            qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback']
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
    save(OUT/'raw'/case/'checks126.json',dict(case=case,frames=len(rr),actual_RAW=True,original69_exact123=case!='negative5',old_worker_exact=True,occupancy_exact124=True,hybrid_exact125=True,
        cutoff_and_no_cache_reads=True,active_actual_prefix=case.startswith('prefix'),decoded_frames=decoded,labels_read=False,source_hashes_unchanged=True))
    print(json.dumps(dict(case=case,frames=len(rr),checked=True)),flush=True)
def run(case):
    c=verify()['cases'][case];args=[sys.executable,str(ROOT/'core/run_support_forecast.py'),'--raw',c['raw'],'--output',str(OUT/'raw'/case),'--start',str(c['start']),'--duration',str(c['duration']),'--verify-frames']
    if not case.startswith('prefix'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as log:r=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,(case,r.returncode);check(case)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check']);p.add_argument('--case',choices=CASES);a=p.parse_args();globals()[a.action](a.case) if a.action!='freeze' else freeze()
