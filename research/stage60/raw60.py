"""Actual RAW paired/off observation check before expensive full forecasting."""
from pathlib import Path
import argparse,datetime,sys
import numpy as np
import study60 as current
from frontend60 import LineFrontend,PARAMS
import profile60
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=current.OUT;PROTOCOL=OUT/'raw_supplement.json'

def sources():return {p.relative_to(current.ROOT).as_posix():current.sha(p) for p in [Path(__file__),Path(__file__).with_name('frontend60.py'),Path(__file__).with_name('profile60.py')]}
def freeze():
    current.verify();assert not PROTOCOL.exists();profile_checks=profile60.checks()
    current.save(PROTOCOL,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),frontend_parameters=PARAMS,profile_parameters=profile60.PARAMS,profile_checks=profile_checks,
        cases={c:current.previous.previous.previous.CASES[c] for c in current.CASES},
        design='Actual full57 intervals: off and paired separate frontend states, same current RAW and cached57 NN body hints known before packet start only. No frequency/forecast fields/labels used. Off must exactly57 current observation at every native tick. Pairing/profile heuristic not true identity; matched thin ellipses get bounded convex center/axis observation fusion, distinct centerlines are raw candidates for unchanged grouping. No forced four or new semantic truth. Conditional gate: if geometry changes, proceed full frequency/15 forecasts and actual RAW prefix with independent protocol.'))
    print('60 actual RAW frontend controls frozen',flush=True)
def verify():
    current.verify();assert current.load(PROTOCOL)['source_hashes']==sources()
def run():
    verify();assert not (OUT/'raw_summary.json').exists();summary=[]
    for case,cfg in current.load(PROTOCOL)['cases'].items():
        refdir=current.previous.previous.previous.OUT/case
        parents=current.previous.rows(refdir/'observations_forecasts.jsonl');reference=current.previous.rows(refdir/'rotor_ticks.jsonl');hints={}
        for p in parents:
            hint=p['body_hint_sent']
            if hint is not None:assert round(hint['t']*1e6)<=round((p['t']-.02)*1e6)
            for t in [p['t']-.01,p['t']]:hints[round(t*1e6)]=hint
        reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();off=LineFrontend(w,h,'neighbor10','off');paired=LineFrontend(w,h,'neighbor10','paired');records=[]
        for index,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
            hint=hints[round(t*1e6)];off.set_hint(e,hint,t,'body');paired.set_hint(e,hint,t,'body')
            old,_=off.detect(e,t);new,_=paired.detect(e,t);assert old==reference[index]['observation'] and round(t,8)==round(reference[index]['t'],8)
            records.append(dict(t=t,off=old,paired=new,line_audit=paired.line_audit,changed=old!=new))
        assert len(records)==round(cfg['duration']*100)
        current.save(OUT/f'raw_{case}.json',dict(records=records,original_geometry_exact57=True))
        row=dict(case=case,native10=len(records),changed=sum(r['changed'] for r in records),
          profile_usable=sum(r['line_audit'].get('profile_usable',0) for r in records),
          inserted=sum(r['line_audit'].get('inserted',0) for r in records),fused=sum(r['line_audit'].get('fused',0) for r in records),
          original_four=sum(len(r['off']['rotors'])==4 for r in records),paired_four=sum(len(r['paired']['rotors'])==4 for r in records))
        summary.append(row);print(row,flush=True)
    current.save(OUT/'raw_summary.json',dict(table=summary,labels_used=False,off_exact57=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);globals()[p.parse_args().action]()
