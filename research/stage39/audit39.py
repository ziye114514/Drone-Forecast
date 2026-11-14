"""Read-only failure diagnosis; never modify frozen run/protocol."""
from pathlib import Path
import json,sys
import study39 as s

def difference(a,b,path=''):
    if type(a)!=type(b):return [dict(path=path,a=a,b=b)]
    if isinstance(a,dict):
        return sum((difference(a.get(k),b.get(k),path+'/'+k) for k in sorted(a.keys()|b.keys())),[])
    if isinstance(a,list):
        if len(a)!=len(b):return [dict(path=path,a=a,b=b)]
        return sum((difference(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
    return [] if a==b else [dict(path=path,a=a,b=b)]

def main():
    cfg=s.CASES['train3'];old={round(r['t'],8):r for r in map(json.loads,(s.ROOT.parent/'output'/cfg['observation']).read_text(encoding='utf-8').splitlines())}
    records=[]
    for mode in ['standalone','paired']:
        reader=s.open_raw(cfg['raw']);h,w=reader.get_size();front=s.RecoveryFrontend(w,h,'neighbor10');other=s.RecoveryFrontend(w,h,'neighbor10');noise=s.StreamFilter(w,h)
        diffs=[]
        for i,(t,e) in enumerate(s.raw_chunks(reader,25.,1.5,10000)):
            clean=noise.step(e)
            if mode=='paired' and i%2==0:other.detect(clean,t)
            o,_=front.detect(e,t)
            if mode=='paired' and i%2:other.detect(clean,t)
            delta=difference(o,old[round(t,8)]['observation'])
            if delta:diffs.append(dict(t=t,event_count=len(e),old_count=old[round(t,8)]['event_count'],differences=delta))
        records.append(dict(mode=mode,divergent_frames=len(diffs),first_differences=diffs[:4]))
    s.save(s.OUT/'baseline_failure_audit.json',records);print(json.dumps(records,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
