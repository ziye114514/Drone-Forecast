"""Exact global-stride event selection before concatenating a large window."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage80')]
import exact80

PARAMS=dict(window_us=160000,maximum_events=450000,packet_us=20000,
    numerical_parameters_unchanged=True,
    selection='Count eligible packets; compute the same global stride/phase '
        'as concatenate/filter/[::stride]; concatenate only selected slices. '
        'No per-packet stride restart and no altered timestamps/order.',
    scope='Memory/copy optimization only, not a new periodic estimator or '
          'new future-position model. Keep all78 activity fields exact.')

def sample_window(buffer,start,end,cap=450000):
    """Equivalent to concatenate(valid_packets)[::ceil(count/cap)].

    Packet metadata certifies every event lies in [packet_end-20ms,end).
    A partial boundary packet alone needs a mask. Global offset carries the
    decimation phase across packets whose lengths are not stride multiples.
    """
    eligible=[]
    for packet_t,e in buffer:
        cutoff=round(packet_t*1e6)
        if cutoff<=start or cutoff-20000>=end:continue
        part=e if cutoff-20000>=start and cutoff<=end else e[(e['t']>=start)&(e['t']<end)]
        eligible.append(part)
    count=sum(len(e) for e in eligible);assert count<2**53 and cap>0
    stride=max(1,(count+cap-1)//cap);parts=[];offset=0
    for e in eligible:
        first=(-offset)%stride;parts.append(e[first::stride]);offset+=len(e)
    selected=np.concatenate(parts) if parts else buffer[-1][1][:0].copy()
    assert len(selected)==(count+stride-1)//stride
    return selected,count,stride

source=exact80.source
old="all_e=np.concatenate([q[1] for q in self.buffer]);start=round(t*1e6)-160000\n    all_e=all_e[(all_e['t']>=start)&(all_e['t']<round(t*1e6))]\n    count=len(all_e);stride=max(1,int(np.ceil(count/PARAMS['max_events'])));e=all_e[::stride]"
assert source.count(old)==1
source=source.replace(old,"start=round(t*1e6)-160000\n    e,count,stride=sample_window(self.buffer,start,round(t*1e6),PARAMS['max_events'])")
namespace=dict(exact80.namespace,sample_window=sample_window)
exec(compile(source,'<82 exact streamed global selection>','exec'),namespace)
class StreamingActivity(exact80.CachedActivity):step=namespace['step']
class OnlineExtras(exact80.OnlineExtras):
    def __init__(self,w,h):super().__init__(w,h);self.activity=StreamingActivity(w,h)

def controls():
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);rng=np.random.default_rng(8200);n=0
    buffers=[]
    for lengths in [[0]*10,[1]*10,[7,10,0,15,6,25,2,13,8,12],[130003,149999,160001,110003,200001,175003,90001,150007,100019,120007]]:
        buffer=[]
        for j,length in enumerate(lengths):
            e=np.zeros(length,dtype);e['t']=rng.integers(j*20000,(j+1)*20000,len(e));e['x']=rng.integers(0,100,len(e));e['y']=rng.integers(0,100,len(e));e['p']=rng.integers(0,2,len(e))
            # Deliberately unsorted within a valid packet: preserve original
            # order and support partial-boundary filtering without searchsorted.
            buffer.append(((j+1)*.02,e))
        buffers.append(buffer)
        for start,end in [(0,200000),(40000,200000),(45001,198777),(200000,220000)]:
            all_e=np.concatenate([e for _,e in buffer]);all_e=all_e[(all_e['t']>=start)&(all_e['t']<end)]
            for cap in [1,3,13,450000,2000000]:
                selected,count,stride=sample_window(buffer,start,end,cap)
                reference=all_e[::max(1,int(np.ceil(len(all_e)/cap)))];assert count==len(all_e)
                for field in dtype.names:np.testing.assert_array_equal(selected[field],reference[field])
                n+=1
    # Per-packet phase restart is a known bad implementation, not used.
    b=buffers[2];global_e,count,stride=sample_window(b,0,200000,13)
    restarted=np.concatenate([e[::stride] for _,e in b])
    assert len(global_e)!=len(restarted) or not np.array_equal(global_e['t'],restarted['t'])
    return dict(cases=n,all_fields_exact_original_selection=True,
        empty_singleton_partial_unsorted_large_windows=True,incorrect_packet_phase_restart_detected=True,
        parameters_changed=False,scope='Selection/indexing controls; real RAW activity equality follows.')
