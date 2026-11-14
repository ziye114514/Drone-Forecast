"""Does the front end keep rotor support across cutoffs, and can it carry its own motion?

Arms, all causal and label-free:
  A  per-frame support (one 33.3ms window)
  B  decayed accumulation, fixed pixels (no motion model)
  C  decayed accumulation moved by a GIVEN past velocity (frozen run; diagnostic ceiling)
  D  decayed accumulation moved by the front end's OWN phase-correlation estimate

Continuity is scored in object coordinates: the previous mask is moved by an
independent past velocity before overlap, because a tracked object legitimately
moves between cutoffs. Raw fixed-pixel IoU is reported too, to show why the
wrong metric flatters a smear.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from live_decode import native_batches
from rotor_support_front import (PARAMS,RotorSupportTracker,components,estimate_region_translation,
    rotor_support,shift_mask)

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
SEQUENCES=[dict(name='clear_26.4_26.8',raw='E:/dataset/train/6/media/event.raw',start=26.40,duration=.40,
        predictions='motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl',
        note='four discs, dense view'),
    dict(name='side_110.9_111.3',raw='E:/dataset/train/6/media/event.raw',start=110.90,duration=.40,
        predictions='motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl',
        note='side view, axis_only in the frozen run')]


def iou(a,b):
    union=np.logical_or(a,b).sum()
    return None if not union else float(np.logical_and(a,b).sum()/union)


def continuity(masks,times,speeds):
    raw=[iou(a,b) for a,b in zip(masks,masks[1:])]
    aligned=[]
    for index in range(1,len(masks)):
        dt=float(times[index]-times[index-1]);velocity=speeds[index-1]
        previous=masks[index-1]
        if velocity is not None:
            previous=shift_mask(previous,float(velocity[0])*dt,float(velocity[1])*dt)
        aligned.append(iou(previous,masks[index]))
    def median(values):
        values=[v for v in values if v is not None]
        return None if not values else float(np.nanmedian(values))
    counts=[int(np.count_nonzero(mask)) for mask in masks]
    return dict(iou_raw_median=median(raw),iou_object_median=median(aligned),
        iou_object=[None if v is None else round(v,3) for v in aligned],
        support_pixels=counts)


def estimator_check():
    on=np.zeros((720,1280),np.uint16);off=np.zeros((720,1280),np.uint16)
    on[320:380,570:630]=120;off[320:380,600:660]=90
    support=np.zeros((720,1280),bool);support[315:385,565:665]=True
    dx,dy=7,-4
    shifted=(np.roll(np.roll(on,dy,0),dx,1),np.roll(np.roll(off,dy,0),dx,1))
    estimate=estimate_region_translation((on,off),support,shifted)
    return dict(requested=[dx,dy],estimate=[estimate['dx'],estimate['dy']],
        exact=bool(estimate['dx']==dx and estimate['dy']==dy),method=estimate['method'],
        note='whole-frame phase correlation returned zero on real data because the static background dominates')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(sequences={},controlled=dict(estimator=estimator_check()),
        body_box_used=False,labels_used=False,identity_tracking=False,
        arm_C_note='given velocity comes from the frozen run: a diagnostic ceiling, not available to the front end',
        arm_D_note='own phase correlation between consecutive ON/OFF images; past frames only')
    for spec in SEQUENCES:
        started=time.perf_counter()
        rows=[json.loads(line) for line in (BASE/spec['predictions']).read_text('utf-8').splitlines()]
        trackers={key:RotorSupportTracker(self_motion=(key=='D')) for key in ('B','C','D')}
        times=[];speeds=[];masks={'A':[],'B':[],'C':[],'D':[]};shifts=[];sources=[]
        queue=deque();end=spec['start']+spec['duration']
        for t,events in native_batches(Path(spec['raw']),spec['start']-.25,spec['duration']+.3,False,
                33333,reader_backend='rolling'):
            queue.append(events.copy())
            while queue and (not len(queue[0]) or queue[0]['t'][-1]<(t-.22)*1e6):queue.popleft()
            if t<spec['start']-1e-9:continue
            if t>end+1e-9:break
            frame=rotor_support(np.concatenate(queue),float(t))
            row=min(rows,key=lambda r:abs(r['t']-t))
            velocity=np.asarray(row['coefficient']['velocity'],float) if abs(row['t']-t)<1/15 else None
            times.append(float(t));speeds.append(velocity)
            masks['A'].append(frame['period_mask']>0)
            for key in ('B','C','D'):
                copy=dict(frame)
                if key=='C':trackers[key].observe(copy,float(t),velocity=velocity)
                else:trackers[key].observe(copy,float(t))
                masks[key].append(copy['stable_support'])
                if key=='D':
                    shifts.append(copy['accumulation'].get('shift_px'))
                    sources.append(copy['accumulation'].get('source'))
        entry=dict(spec=spec,cutoffs=len(times),arms={})
        for key in ('A','B','C','D'):
            entry['arms'][key]=continuity(masks[key],times,speeds)
        entry['self_estimates_px']=shifts
        entry['self_estimate_sources']=sources
        given=[None if v is None else [round(float(x)*(times[i+1]-times[i]),2) for x in v]
            for i,v in enumerate(speeds[:-1])]
        errors=[]
        for estimate,reference in zip(shifts,given):
            if estimate and reference:
                errors.append(float(np.hypot(estimate[0]-reference[0],estimate[1]-reference[1])))
        entry['self_estimate_error_px']=dict(n=len(errors),
            median=None if not errors else float(np.median(errors)),
            max=None if not errors else float(max(errors)))
        entry['given_shift_px']=given
        entry['elapsed_s']=round(time.perf_counter()-started,2)
        result['sequences'][spec['name']]=entry
        print(spec['name'],{k:(round(v['iou_object_median'],3) if v['iou_object_median'] else None)
            for k,v in entry['arms'].items()},
            'estimate_err',entry['self_estimate_error_px'],flush=True)
    (OUT/'sequence.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print(json.dumps(result['controlled'],ensure_ascii=False))
    print('wrote',OUT/'sequence.json')


if __name__=='__main__':main()
