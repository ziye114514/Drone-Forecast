"""Offline test: fuse the absolute support centre with the displacement, causally.

Facts measured from the frozen stage164 run (all 2D, single camera, current frame
only, no labels in any input):

  * body.center == previous center + displacement  (61/61 frames), so the current
    position is dead reckoning with no absolute anchor;
  * the absolute per-frame support box centre is a SEPARATE observation: it is
    12.2px from the manual centre on average while the integrated centre is 9.2px,
    and the two differ by 18.5px, i.e. they are not the same number;
  * the observer fed the integrated centre as a 6px measurement while also feeding
    the displacement as a velocity row, so the two rows were degenerate and the
    filter was over-confident.

This compares, on the same 9 frozen queries:
  baseline   recorded 0.4s prediction
  fusion     a constant-velocity filter whose position row is the ABSOLUTE centre
             and whose velocity row is the displacement, then a 0.4s rollout
Nothing is fitted and no future value is read.
"""
import json,pathlib,bisect
import numpy as np

BASE=pathlib.Path(r"D:\drone forecast\output\side_temporal164_20261101")
LAB=pathlib.Path(r"D:\drone forecast\output\dense_labels_side_110_1195\labels.json")
TAU=0.4
POSITION_SIGMA=12.0
DISPLACEMENT_SIGMA=2.0
ACCEL_SIGMA=1000.0
CORRELATION=.35


def reference_at(records,t):
    stamps=[r['t'] for r in records];i=bisect.bisect_left(stamps,t)
    if i<len(records) and abs(stamps[i]-t)<1e-7:pair=[records[i],records[i]]
    elif 0<i<len(records) and stamps[i]-stamps[i-1]<=.1500001:pair=[records[i-1],records[i]]
    else:return None
    if any(r.get('state')!='present' or not r.get('box') for r in pair):return None
    centres=[(np.asarray(r['box'][:2])+np.asarray(r['box'][2:]))/2 for r in pair]
    alpha=0. if pair[0]['t']==pair[1]['t'] else (t-pair[0]['t'])/(pair[1]['t']-pair[0]['t'])
    return (1-alpha)*centres[0]+alpha*centres[1]


def run(rows,use_absolute,use_displacement,use_acceleration):
    """Tiny CV/CA filter over the saved frames; measurement rows are declared above."""
    x=None;P=None;previous_t=None;previous_centre=None;outputs={}
    for row in rows:
        body=row.get('body') or {}
        if not body.get('detected'):
            continue
        bb=np.asarray(body['bbox'],float);absolute=(bb[:2]+bb[2:])/2
        estimate=(row.get('motion') or {}).get('estimate') or {}
        displacement=np.array([estimate.get('dx',0.),estimate.get('dy',0.)]) if estimate else None
        acceleration=np.asarray((row.get('force') or {}).get('acceleration_px_s2') or [0.,0.],float)
        t=float(row['t'])
        if x is None:
            x=np.r_[absolute,np.zeros(2)];P=np.diag([POSITION_SIGMA**2]*2+[300.**2]*2)
            previous_t=t;previous_centre=absolute;outputs[t]=x[:2].copy();continue
        dt=float(t-previous_t)
        if dt<=0:continue
        F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
        q=ACCEL_SIGMA**2*CORRELATION
        block=np.array([[dt**4/4,dt**3/2],[dt**3/2,dt**2]])*q
        Q=np.zeros((4,4))
        for axis in range(2):
            idx=[axis,2+axis];Q[np.ix_(idx,idx)]=block
        x=F@x;P=F@P@F.T+Q
        blocks=[]
        if use_absolute:
            blocks.append((np.array([[1,0,0,0],[0,1,0,0]],float),absolute,
                np.eye(2)*POSITION_SIGMA**2))
        if use_displacement and displacement is not None and displacement.size==2 and dt<=.12:
            blocks.append((np.array([[0,0,1,0],[0,0,0,1]],float),displacement/dt,
                np.eye(2)*(DISPLACEMENT_SIGMA**2/(dt*dt))))
        if use_acceleration:
            blocks.append((np.array([[1,0,0,0],[0,1,0,0]],float),absolute+.5*acceleration*dt*dt,
                np.eye(2)*POSITION_SIGMA**2))
        if not blocks:
            outputs[t]=x[:2].copy();previous_t=t;previous_centre=absolute;continue
        H=np.vstack([block[0] for block in blocks])
        z=np.concatenate([np.asarray(block[1],float) for block in blocks])
        size=sum(block[2].shape[0] for block in blocks)
        R=np.zeros((size,size));offset=0
        for block in blocks:
            n=block[2].shape[0];R[offset:offset+n,offset:offset+n]=block[2];offset+=n
        S=H@P@H.T+R;K=np.linalg.solve(S,H@P).T
        x=x+K@(z-H@x);A=np.eye(4)-K@H;P=A@P@A.T+K@R@K.T
        outputs[t]=x[:2].copy()
        previous_t=t;previous_centre=absolute
    return outputs


def main():
    rows=[json.loads(x) for x in (BASE/'raw_side_final/predictions.jsonl').read_text('utf-8').splitlines()]
    records=sorted(json.loads(LAB.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    variants={
        'baseline_recorded':None,
        'absolute_only':dict(use_absolute=True,use_displacement=False,use_acceleration=False),
        'displacement_only':dict(use_absolute=False,use_displacement=True,use_acceleration=False),
        'fused':dict(use_absolute=True,use_displacement=True,use_acceleration=False),
        'fused_plus_recorded_acceleration':dict(use_absolute=True,use_displacement=True,use_acceleration=True),
    }
    results={}
    for name,config in variants.items():
        if config is None:
            errors=[]
            for row in rows:
                physical=row.get('physical') or {}
                future=physical.get('0.4')
                if future is None:continue
                truth=reference_at(records,row['t']+TAU)
                if truth is None:continue
                errors.append(float(np.linalg.norm(np.asarray(future,float)-truth)))
            results[name]=dict(n=len(errors),mean=float(np.mean(errors)),median=float(np.median(errors)),
                p95=float(np.percentile(errors,95)))
            continue
        outputs=run(rows,**config)
        errors=[];now=[]
        for t,position in outputs.items():
            truth=reference_at(records,t+TAU)
            current=reference_at(records,t)
            if truth is None or current is None:continue
            errors.append(float(np.linalg.norm(position-truth)))
            now.append(float(np.linalg.norm(position-current)))
        results[name]=dict(n=len(errors),mean=float(np.mean(errors)),median=float(np.median(errors)),
            p95=float(np.percentile(errors,95)),current_error_mean=float(np.mean(now)))
    print(json.dumps(results,indent=1,ensure_ascii=False))
    (BASE/'fusion_offline.json').write_text(json.dumps(dict(variants=results,
        position_sigma=POSITION_SIGMA,displacement_sigma=DISPLACEMENT_SIGMA,
        note='offline over saved frames; constant-velocity rollout of 0.4s; no future value used in any input'),
        indent=2),'utf-8')


if __name__=='__main__':
    main()
