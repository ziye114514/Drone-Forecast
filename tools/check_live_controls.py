"""Small deterministic controls for new scheduling and publication semantics."""
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from live_stream import LatestReplay,RateGate,MinimumIntervalGate
from live_motion import predictions

gate=RateGate(0.,5.)
assert gate.take(0.) and not gate.take(.1) and gate.take(.61)
assert gate.missed==2 and not gate.take(.62) and gate.take(.8)
gate=MinimumIntervalGate(0.,5.)
assert not gate.take(.19) and gate.take(.61) and not gate.take(.62) and not gate.take(.8) and gate.take(.81)

prepared=[]
def prepare(t,e):
    prepared.append(t)
    return e.copy(),None
source=LatestReplay(((i*.02,np.zeros(1)) for i in range(1,9)),0.,prepare)
received=[]
try:
    for packet in source:
        received.append(packet.t)
        assert packet.t<=source.source_now()+1e-6
        time.sleep(.06)
finally:source.close()
assert prepared==[i*.02 for i in range(1,9)]
assert received[-1]==.16 and len(received)<len(prepared)
assert source.overwritten>0 and not source.thread.is_alive()

def broken(t,e):raise RuntimeError('intentional producer failure')
source=LatestReplay(iter([(.02,np.zeros(1))]),0.,broken)
try:
    try:list(source)
    except RuntimeError as e:assert 'intentional producer failure' in str(e)
    else:raise AssertionError('Producer error was swallowed')
finally:source.close()

c=dict(position=[0.,0.],velocity=[10.,0.],acceleration=[0.,0.])
preferred,cv=predictions(c,.1)
assert np.allclose(cv['0.4'],[5.,0.])
assert np.allclose(preferred['0.4'],[5.*(1.-.9*.5*.5),0.])
print('PASS: all native batches processed; latest-slot overwrite; no future packet; producer shutdown/error; no catchup burst; publication-age compensation')
