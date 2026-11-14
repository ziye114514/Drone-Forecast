"""Record every 10ms recovery event; frozen public stream is only 20ms."""
import json,time
from study import OUT,ROOT,CASES,verify,load,save,fronts,open_raw,raw_chunks,counts


if __name__=='__main__':
    verify();assert not (OUT/'side/event_audit.json').exists()
    cfg=CASES['side'];reader=open_raw(cfg['raw']);h,w=reader.get_size()
    model=fronts(w,h)['guard_neighbor10']
    old={round(r['t'],8):r for r in map(json.loads,(OUT/'side/observations.jsonl').read_text().splitlines())}
    start=cfg['start']-.03;end=cfg['start']+cfg['duration'];events=[];equal=0;t0=time.perf_counter()
    with (OUT/'side/audit_10ms.jsonl').open('w',encoding='utf-8') as file:
        for t,e in raw_chunks(reader,start,end-start,10000):
            o,_=model.detect_prepared(e,t,counts(e,w,h));a=model.audit
            row=dict(t=t,audit=a,observation=o)
            file.write(json.dumps(row,allow_nan=False)+'\n');events.append(row)
            if round(t,8) in old:
                assert o==old[round(t,8)]['methods']['guard_neighbor10']['observation']
                equal+=1
    save(OUT/'side/event_audit.json',dict(published_exact=equal,processed=len(events),wall_s=time.perf_counter()-t0,
        reference_active_frames=sum(bool(r['audit']['reference_active']) for r in events),
        scale_filter_frames=sum(r['audit'].get('scale_eligible',0)<r['audit'].get('original_candidates',0) for r in events),
        state_rejection_frames=sum(bool(r['audit']['state_rejection']) for r in events),
        challenge_frames=[dict(t=r['t'],count=r['audit']['challenge_count']) for r in events if 'challenge_count' in r['audit']],
        transfers=[dict(t=r['t'],center=r['observation'].get('center'),selected=r['audit']['selected_count']) for r in events if r['audit'].get('transferred')],
        note='Recovery may happen on an odd10ms frame and be absent from20ms output diagnostics. Same frozen algorithm; every published observation matches.'))
    print(json.dumps(load(OUT/'side/event_audit.json')),flush=True)
