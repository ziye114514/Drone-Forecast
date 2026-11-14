"""Bounded-memory exact angular/temporal Fourier projection.

The spatial order and temporal frequency factor, so repeated orders need not
repeat every costly temporal exponential. No event or timestamp is discarded.
"""
import numpy as np


def project(times_s,angles,weights,frequencies,orders):
    frequencies=np.asarray(frequencies);orders=np.asarray(orders)
    ff,fi=np.unique(frequencies,return_inverse=True);mm,mi=np.unique(orders,return_inverse=True)
    spatial=(np.asarray(weights)[:,None]*np.exp(1j*np.asarray(angles)[:,None]*mm[None])).T
    result=np.empty((len(mm),len(ff)),complex);tt=np.asarray(times_s)
    for begin in range(0,len(ff),32):
        fs=ff[begin:begin+32]
        temporal=np.exp(-2j*np.pi*tt[:,None]*fs[None])
        result[:,begin:begin+len(fs)]=spatial@temporal
    return result[mi,fi]/max(float(np.sum(weights)),1e-12)
