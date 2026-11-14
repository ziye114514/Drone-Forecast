"""Preserve exact old-branch divergence; do not weaken geometry regression."""
from pathlib import Path
import argparse,datetime
import numpy as np
import raw60 as current
from frontend60 import LineFrontend
from gating57 import CandidateFrontend
from event_drone_forecast.raw_io import open_raw,raw_chunks

def run(kind):
    current.verify();out=current.OUT/f'diagnostic_train3_{kind}.json';assert not out.exists()
    cfg=current.current.previous.previous.previous.CASES['train3'];refdir=current.current.previous.previous.previous.OUT/'train3'
    rows=current.current.previous.rows;parents=rows(refdir/'observations_forecasts.jsonl');reference=rows(refdir/'rotor_ticks.jsonl');hints={}
    for p in parents:
        for t in [p['t']-.01,p['t']]:hints[round(t*1e6)]=p['body_hint_sent']
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();off=LineFrontend(w,h,'neighbor10','off') if kind!='original' else CandidateFrontend(w,h,'neighbor10')
    paired=LineFrontend(w,h,'neighbor10','paired') if kind=='interleaved' else None
    exact=0;first=None;result=[]
    for index,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
        off.set_hint(e,hints[round(t*1e6)],t,'body');actual,_=off.detect(e,t)
        expected=reference[index]
        if paired:
            paired.set_hint(e,hints[round(t*1e6)],t,'body');paired.detect(e,t)
        equal=actual==expected['observation'] and round(t,8)==round(expected['t'],8)
        if equal:exact+=1
        else:
            first=dict(index=index,t=t,expected_t=expected['t'],raw_events=len(e),expected_events=expected['event_count'],actual=actual,expected=expected['observation'])
            break
    current.current.save(out,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,exact_ticks_before_divergence=exact,first_divergence=first,all_complete=first is None))
    print(dict(kind=kind,exact=exact,first_index=first['index'] if first else None,actual_t=first['t'] if first else None),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['off','original','interleaved']);run(p.parse_args().kind)
