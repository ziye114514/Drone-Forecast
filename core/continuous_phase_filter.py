"""Persistent native-timestamp phase state; validation never updates its copy."""
from pathlib import Path
import ctypes as C
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4)
def library(path):
    lib=C.CDLL(str(path));lib.cp_feed.argtypes=[C.POINTER(C.c_double),C.POINTER(C.c_int64),C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_int,C.c_int]
    lib.cp_feed.restype=C.c_int;return lib


def initial(time_us,phase,omega,alpha=0.):
    return np.r_[float(time_us),float(phase),float(omega),float(alpha),np.diag([1.,100.,10000.]).reshape(-1)].astype(np.float64)


def feed(state,times,phases,weight,*,dll,policy='mixture'):
    result=np.ascontiguousarray(state,dtype=np.float64).copy();tt=np.ascontiguousarray(times,dtype=np.int64)
    zz=np.ascontiguousarray(phases,dtype=np.float64);ww=np.ascontiguousarray(weight,dtype=np.float64)
    if not len(tt):return result
    if not (len(tt)==len(zz)==len(ww)) or np.any(np.diff(tt)<0):raise ValueError('Sorted equal-sized native observations required')
    lib=library(str(Path(dll).resolve()));mode={'mixture':1,'cohelix_dynamics':2}[policy]
    if mode==2 and (not hasattr(lib,'cp_version') or lib.cp_version()<2):raise ValueError('Rebuild phase DLL with dynamics-v2 support')
    status=lib.cp_feed(result.ctypes.data_as(C.POINTER(C.c_double)),tt.ctypes.data_as(C.POINTER(C.c_int64)),
        zz.ctypes.data_as(C.POINTER(C.c_double)),ww.ctypes.data_as(C.POINTER(C.c_double)),len(tt),mode)
    if status:raise RuntimeError(f'Continuous phase kernel failed: {status}')
    return result


def predict(state,times):
    age=(np.asarray(times,dtype=np.float64)-state[0])*1e-6
    return state[1]+state[2]*age+.5*state[3]*age*age


def score(state,times,angles,weights,*,order,sign):
    if len(times)<32:return dict(qualified=False,reason='few_current_ring_events',events=len(times))
    phase=predict(state,times);angle=sign*np.asarray(angles);weight=np.asarray(weights)
    value=np.sum(weight*np.exp(1j*(phase-order*angle)))/max(weight.sum(),1e-12)
    other=np.sum(weight*np.exp(1j*(phase+order*angle)))/max(weight.sum(),1e-12)
    sectors=len(np.unique(np.floor((angle[weight>=.1]+np.pi)*12/(2*np.pi)).astype(int)%12))
    return dict(complex_mean=[float(value.real),float(value.imag)],concentration=float(abs(value)),
        opposite_power=float(abs(other)**2),events=len(times),sectors=sectors)


def frozen_validation(state,times,angles,weights,early,*,order,sign):
    train=score(state,times[early],angles[early],weights[early],order=order,sign=sign)
    late=score(state,times[~early],angles[~early],weights[~early],order=order,sign=sign)
    if 'complex_mean' not in train or 'complex_mean' not in late:return dict(qualified=False,train=train,validation=late)
    a=complex(*train['complex_mean']);b=complex(*late['complex_mean'])
    agreement=float(np.real(b*np.conj(a))/max(abs(a),1e-12));power=abs(b)**2;wrong=late['opposite_power']
    contrast=float((power-wrong)/max(power+wrong,1e-12))
    return dict(qualified=bool(abs(a)>=.5 and abs(b)>=.5 and agreement>=.5 and contrast>=.5 and late['sectors']>=9),
        train=train,validation=late,phase_agreement=agreement,direction_contrast=contrast,
        validation_updates_phase_or_geometry=False,coherence_threshold=.5,sectors_threshold=9)
