from pathlib import Path
from types import SimpleNamespace
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_publication import TimedPublisher

source=SimpleNamespace(epoch=time.perf_counter(),start=10.,done=False)
seen=[]
q=TimedPublisher(source,.8,5.,lambda state,wall,index,deadline:seen.append((state,wall,index,deadline)))
q.start();q.finish()
assert len(seen)==4 and all(row[0] is None for row in seen)
assert [row[2] for row in seen]==[1,2,3,4]
assert max(abs(row[3]-(10.+row[2]*.2)) for row in seen)<1e-12
source=SimpleNamespace(epoch=time.perf_counter(),start=0.,done=False)
seen=[]
def slow(state,wall,index,deadline):
    seen.append((wall,index))
    if index==1:time.sleep(.31)
q=TimedPublisher(source,.8,5.,slow);q.start();q.finish()
assert all(b[0]-a[0]>=.1 for a,b in zip(seen,seen[1:]))
assert q.missed>=1
print('PASS: independent5Hz deadlines; invalid heartbeats without measurements; skipped overruns; no catchup burst')
