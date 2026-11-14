"""Match original top-five spectral candidates across two temporal halves."""
from pathlib import Path
import sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage66')]
import periodicity66 as original
PARAMS=dict(original.PARAMS,matching='Keep original top5 peaks with individual peak/noise>=8. '
    'A full-window peak must have a qualified peak within one half-window FFT '
    'bin in both temporal halves. Retain every matching frequency, no motor '
    'assignment or harmonic/mechanical RPM selection. Original dominant-only '
    'half-stability remains as a control, not overwritten.')


def peak(f,p):
    q=original.peak(f,p)
    if q is None:return None
    allowed=(f>=PARAMS['min_hz'])&(f<=PARAMS['max_hz'])
    noise=max(float(np.median(p[allowed])),1e-12)
    q['top_peaks_snr']=[float(p[int(np.argmin(abs(f-v)))]/noise) for v in q['top_peaks_hz']]
    return q


native_analyse=types.FunctionType(original.analyse.__code__,dict(original.__dict__,peak=peak))


def analyse(signals,event_counts,spans_us):
    rr=native_analyse(signals,event_counts,spans_us)
    for q in rr:
        matches=[]
        if q.get('valid'):
            early=q['first_half'];late=q['last_half']
            if early and late:
                tolerance=early['frequency_resolution_hz']
                for f,snr in zip(q['top_peaks_hz'],q['top_peaks_snr']):
                    if snr<PARAMS['snr_min']:continue
                    a=[(abs(h-f),h,s) for h,s in zip(early['top_peaks_hz'],early['top_peaks_snr']) if s>=PARAMS['snr_min'] and abs(h-f)<=tolerance]
                    b=[(abs(h-f),h,s) for h,s in zip(late['top_peaks_hz'],late['top_peaks_snr']) if s>=PARAMS['snr_min'] and abs(h-f)<=tolerance]
                    if a and b:
                        aa=min(a);bb=min(b)
                        matches.append(dict(frequency_hz=f,full_snr=snr,first_half_hz=aa[1],first_snr=aa[2],last_half_hz=bb[1],last_snr=bb[2]))
        q['multi_half_stable']=bool(matches);q['matched_frequencies']=matches
    return rr


native_tile=types.FunctionType(original.tile_measurement.__code__,dict(original.__dict__,analyse=analyse))


def tile_measurement(*args,**kwargs):
    q=native_tile(*args,**kwargs)
    q['multi_half_stable_candidates']=sum(v['multi_half_stable'] for v in q['tiles'])
    return q


controlled=types.FunctionType(original.controlled.__code__,dict(original.__dict__,analyse=analyse))


def strip_new(value):
    if isinstance(value,dict):return {k:strip_new(v) for k,v in value.items() if k not in ['top_peaks_snr','matched_frequencies','multi_half_stable','multi_half_stable_candidates']}
    if isinstance(value,list):return [strip_new(v) for v in value]
    return value
