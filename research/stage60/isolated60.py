"""Case-isolated frontend controls following a retained cross-case regression failure."""
from pathlib import Path
import argparse,datetime,json,sys
import raw60 as previous
from frontend60 import LineFrontend
from event_drone_forecast.raw_io import open_raw,raw_chunks
ROOT=previous.current.ROOT;BASE=previous.OUT;OUT=BASE/'isolated_raw';PROTOCOL=OUT/'protocol.json'
load=previous.current.load;save=previous.current.save;sha=previous.current.sha;rows=previous.current.previous.rows
REF=previous.current.previous.previous.previous.OUT
CASES=previous.current.previous.previous.previous.CASES

def sources():
    r=previous.current.sources();r.update(previous.sources());r[Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__));return r
def freeze():
    previous.verify();assert not OUT.exists()
    save(BASE/'initial_interleaved_failure.json',dict(known_exit_code=1,batch_session=89399,
       exception='raw60.py current-vs-original assertion, during train3 after clear6 and side6 completed. First offending index/line was not saved; root cause unconfirmed.',
       retained_results=['raw_clear6.json','raw_side6.json'],
       followup='Fresh original, off adapter and interleaved train3 each exactly600 ticks against57; this does not identify the initial cause. Keep exact assertions. Case-isolated fresh processes match production worker lifecycle.'))
    save(PROTOCOL,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
       cases={c:CASES[c] for c in previous.current.CASES},
       design='Same frozen algorithm and full original57 intervals; each recording/case starts in a fresh SDK process as full production workers do. Off and paired frontend states still independent and interleaved within each recording. No assertions relaxed. Record complete offending observations on any mismatch, preserve original failure/results. No new parameter tuning or identity claims.'))
    print('60 case-isolated controls frozen; initial cause remains unknown',flush=True)
def verify():
    previous.verify();p=load(PROTOCOL);assert p['source_hashes']==sources();return p
def run(case):
    verify();path=OUT/f'raw_{case}.json';assert not path.exists();cfg=CASES[case];hints={}
    for p in rows(REF/case/'observations_forecasts.jsonl'):
        hint=p['body_hint_sent']
        if hint is not None:assert round(hint['t']*1e6)<=round((p['t']-.02)*1e6)
        for t in [p['t']-.01,p['t']]:hints[round(t*1e6)]=hint
    reference=rows(REF/case/'rotor_ticks.jsonl');reader=open_raw(Path(cfg['raw']));h,w=reader.get_size()
    off=LineFrontend(w,h,'neighbor10','off');paired=LineFrontend(w,h,'neighbor10','paired');records=[]
    for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
        hint=hints[round(t*1e6)];off.set_hint(e,hint,t,'body');paired.set_hint(e,hint,t,'body');old,_=off.detect(e,t);new,_=paired.detect(e,t)
        if old!=reference[i]['observation'] or round(t,8)!=round(reference[i]['t'],8):
            save(OUT/f'failure_{case}.json',dict(index=i,t=t,actual=old,expected=reference[i]));raise AssertionError('Isolated current off differs57; exact failure preserved')
        records.append(dict(t=t,off=old,paired=new,line_audit=paired.line_audit,changed=old!=new))
    assert len(records)==round(cfg['duration']*100)
    save(path,dict(records=records,original_geometry_exact57=True,labels_read=False));print(case+' isolated RAW complete: '+str(len(records)),flush=True)
def summary():
    verify();table=[]
    for case in previous.current.CASES:
        records=load(OUT/f'raw_{case}.json')['records']
        table.append(dict(case=case,native10=len(records),changed=sum(r['changed'] for r in records),
         profile_usable=sum(r['line_audit'].get('profile_usable',0) for r in records),
         inserted=sum(r['line_audit'].get('inserted',0) for r in records),fused=sum(r['line_audit'].get('fused',0) for r in records),
         original_four=sum(len(r['off']['rotors'])==4 for r in records),paired_four=sum(len(r['paired']['rotors'])==4 for r in records)))
    save(OUT/'raw_summary.json',dict(table=table,labels_used=False,off_exact57=True,first_failure_root_cause_confirmed=False));print(table,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','summary']);p.add_argument('--case',choices=list(previous.current.CASES));a=p.parse_args()
    if a.action=='run':run(a.case)
    else:globals()[a.action]()
