"""Control the actual shared-memory lease/generation behavior without SDK IO."""
from multiprocessing import get_context
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_frames import SharedFrames

pool=SharedFrames(get_context('spawn'),height=8,width=12,slots=3,preview=True)
attached=SharedFrames.attach(pool.spec(),pool.locks,pool.generations)
try:
    with attached.locks[0]:
        attached.images[0,0].fill(17);attached.images[0,1].fill(29);attached.generations[0]=1
    packet=dict(slot=0,generation=1)
    frame,preview=pool.read(packet)
    assert (frame==17).all() and (preview==29).all()
    with attached.locks[0]:
        attached.images[0].fill(83);attached.generations[0]=2
    assert pool.read(packet) is None
    assert (frame==17).all() and (preview==29).all(),'Snapshot aliased a future generation'
    dest=np.zeros((8,12,3),np.uint8)
    direct,other=pool.read(dict(slot=0,generation=2),dest)
    assert direct is dest and (dest==83).all()
    try:pool.read(dict(slot=0,generation=2),np.zeros((4,4,3),np.uint8))
    except ValueError:pass
    else:raise AssertionError('Wrong shared destination accepted')
finally:attached.close();pool.close();pool.close()
print('PASS: cross-handle shared frames; stale generation rejected; stable snapshots; direct destination; cleanup',flush=True)
