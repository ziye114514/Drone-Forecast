"""Post-score hotspot diagnosis at predefined snapshots; no parameter changes."""
import collections,cProfile,pstats,time
import numpy as np
from study import OUT,ROOT,verify,save,CASES
from frontend import Frontend
from event_drone_forecast.raw_io import open_raw,raw_chunks


def main():
    verify();result=[]
    for case,t in [('clear6',23.),('side6',110.),('review4',75.)]:
        reader=open_raw(CASES[case]['raw']);h,w=reader.get_size();model=Frontend(w,h,'roi');history=collections.deque()
        prof=cProfile.Profile();last=None
        for end,e in raw_chunks(reader,t-.22,.22,20000):
            history.append((end,e))
            while history and history[0][0]<=end-.1+1e-8:history.popleft()
            if end<t-.12-1e-8:continue
            recent=np.concatenate([x for _,x in history])
            if end>=t-1e-8:prof.enable()
            last=model.step(recent,e,end)
            if end>=t-1e-8:prof.disable()
        stats=pstats.Stats(prof);entries=[]
        for (file,line,name),(primitive,calls,total,cumulative,callers) in stats.stats.items():
            if file.startswith(str(ROOT)):
                entries.append(dict(file=str(file),line=line,function=name,calls=calls,self_ms=total*1000,cumulative_ms=cumulative*1000))
        entries.sort(key=lambda x:-x['cumulative_ms'])
        result.append(dict(case=case,t=t,current_events=len(e),history_events=len(recent),search=last['search'],detected=last['observation']['detected'],
            timing_ms=last['timing_ms'],entries=entries[:20],note='One cProfile-instrumented call, cold short history; diagnosis only, not a benchmark or full-run replay.'))
        print(case,last['timing_ms'],flush=True)
    save(OUT/'hotspots.json',result)


if __name__=='__main__':main()
