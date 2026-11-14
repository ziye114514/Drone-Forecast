"""Short-pilot triplets for the rotor-only front end (no body box, no labels).

Each clip saves: raw ON/OFF, density filter, period-confirmed support. For clips
that also have a frozen structure run, the current rotor conics are used ONLY as
a diagnostic reference to describe what the filter kept or dropped; they are not
an input, not truth, and the rotor count is never claimed from them.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from live_decode import native_batches
from rotor_support_front import PARAMS,rotor_support

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
DTYPE=[('x','u2'),('y','u2'),('p','i2'),('t','i8')]
CLIPS=[dict(name='clear',raw='E:/dataset/train/6/media/event.raw',t=26.5,
        predictions='motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl',
        note='four discs clearly visible in the frozen run at this cutoff'),
    dict(name='partial',raw='E:/dataset/train/6/media/event.raw',t=26.700,
        predictions='motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl',
        note='frozen run reports partial_two here: overlap / partial visibility'),
    dict(name='side',raw='E:/dataset/train/6/media/event.raw',t=111.0,
        predictions='motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl',
        note='side view: frozen run reports axis_only'),
    dict(name='negative',raw='E:/dataset/train/5/media/event.raw',t=.5,predictions=None,
        note='background control clip used by earlier negative probes')]


def frozen_rotors(path,t):
    if path is None:return None
    rows=[json.loads(line) for line in (BASE/path).read_text('utf-8').splitlines()]
    row=min(rows,key=lambda r:abs(r['t']-t))
    if abs(row['t']-t)>1/15:return None
    return dict(source_t=row['t'],mode=row['structure'].get('mode'),
        rotors=row['geometry'].get('rotors',[]),body=row['body'],rates=row.get('rates',[]))


def ellipse_union(rotors,width,height):
    union=np.zeros((height,width),bool)
    used=[]
    for rotor in rotors:
        if not rotor.get('current_observation',True):continue
        try:cx,cy,angle=float(rotor['cx']),float(rotor['cy']),np.deg2rad(float(rotor['angle']))
        except (KeyError,TypeError,ValueError):continue
        a=float(rotor.get('major',0))/2;b=float(rotor.get('minor',0))/2
        if not (a>0 and b>0):continue
        pad=int(max(a,b))+3
        x0=max(0,int(cx)-pad);x1=min(width,int(cx)+pad+1)
        y0=max(0,int(cy)-pad);y1=min(height,int(cy)+pad+1)
        if x0>=x1 or y0>=y1:continue
        yy,xx=np.mgrid[y0:y1,x0:x1]
        u=(xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
        v=-(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)
        inside=(u/a)**2+(v/b)**2<=1.
        union[y0:y1,x0:x1]|=inside
        used.append(dict(cx=cx,cy=cy,major=a*2,minor=b*2,angle_deg=float(np.rad2deg(angle)),
            mode=rotor.get('mode')))
    return union,used


def overlap_stats(result,union):
    period=result['period_mask']>0
    dense=result['density']
    total_period=int(period.sum());total_dense=int(dense.sum())
    union_period=int((period&union).sum());union_dense=int((dense&union).sum())
    union_pixels=int(union.sum())
    return dict(
        period_pixels=total_period,
        period_inside_reference=union_period,
        period_outside_reference=total_period-union_period,
        period_inside_fraction=None if not total_period else union_period/total_period,
        reference_pixels=union_pixels,
        reference_period_fraction=None if not union_pixels else (period&union).sum()/union_pixels,
        density_pixels=total_dense,
        density_inside_reference=union_dense,
        density_inside_fraction=None if not total_dense else union_dense/total_dense,
        reference_density_fraction=None if not union_pixels else (dense&union).sum()/union_pixels,
        note='reference = frozen current rotor conics at the same cutoff; diagnostic only, not truth and not an input')


def distance_stats(result,rotors,width=1280,height=720):
    """How far the periodic support sits from the frozen rotor centres."""
    period=result['period_mask']>0
    ys,xs=np.nonzero(period)
    if not len(xs) or not rotors:return None
    normalised=[]
    for x,y in zip(xs,ys):
        best=None
        for rotor in rotors:
            a=max(float(rotor.get('major',0))/2,1e-6)
            distance=np.hypot(x-float(rotor['cx']),y-float(rotor['cy']))/a
            best=distance if best is None else min(best,distance)
        normalised.append(best)
    values=np.asarray(normalised)
    return dict(n=int(len(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)),
        within_1_semimajor=float(np.mean(values<=1.)),within_1_5_semimajor=float(np.mean(values<=1.5)),
        note='distance divided by that rotor semi-major; no truth labels involved')


def residual_stats(result,rotors):
    """Density-only pixels: how much survives the period test and where it sits."""
    dense=result['density'];period=result['period_mask']>0
    only=dense&~period
    ys,xs=np.nonzero(only)
    if not len(xs) or not rotors:return dict(density_only_pixels=int(only.sum()))
    far=0
    for x,y in zip(xs,ys):
        best=min(np.hypot(x-float(r['cx']),y-float(r['cy']))/
            max(float(r.get('major',0))/2,1e-6) for r in rotors)
        if best>3.:far+=1
    return dict(density_only_pixels=int(only.sum()),density_only_far_from_rotors=int(far),
        density_only_far_fraction=float(far/len(xs)),
        note='density passed but period did not; far means >3 semi-major from every frozen rotor centre')


def frequency_alignment(component,rates):
    """Nearest frozen per-rotor frequency, keeping harmonic ambiguity visible."""
    if not rates:return None
    values=[float(r['frequency_hz']) for r in rates if r.get('valid') and r.get('frequency_hz')]
    if not values:return None
    f=component.get('median_frequency_hz')
    if f is None:return None
    nearest=min(values,key=lambda v:abs(np.log(f/v)))
    ratio=f/nearest
    return dict(nearest_rate_hz=nearest,ratio=float(ratio),
        harmonic_tag='direct' if abs(np.log(ratio))<=.08 else
            'half' if abs(np.log(ratio*2))<=.08 else
            'double' if abs(np.log(ratio/2))<=.08 else 'other')


def panels(result,title,path):
    """Four panels: full frame, then a display-only crop of the support region.

    The crop is for the eye only; it is not an input and never feeds the filter.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':150})
    on=result['on'].astype(float);off=result['off'].astype(float)
    rgb=np.zeros(on.shape+(3,))
    rgb[:,:,0]=np.log1p(on)/max(np.log1p(on).max(),1e-9)
    rgb[:,:,2]=np.log1p(off)/max(np.log1p(off).max(),1e-9)
    def overlay(mask,color,alpha=.55):
        rgba=np.zeros(mask.shape+(4,),float)
        rgba[...,0]=color[0];rgba[...,1]=color[1];rgba[...,2]=color[2]
        rgba[...,3]=np.asarray(mask,float)*alpha
        return rgba
    region=result.get('bbox')
    if region is None:
        dense=result['density']
        ys,xs=np.nonzero(dense)
        region=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else [0,0,1279,719]
    pad=26
    x0=max(0,region[0]-pad);y0=max(0,region[1]-pad)
    x1=min(1280,region[2]+pad);y1=min(720,region[3]+pad)
    if x1-x0<40:x1=min(1280,x0+40)
    if y1-y0<40:y1=min(720,y0+40)
    crop=rgb[y0:y1,x0:x1]
    fig,axes=plt.subplots(1,4,figsize=(19,4.8))
    axes[0].imshow(rgb)
    axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='yellow',lw=1.2))
    axes[0].set_title(f'{title}：整帧（{result["counts"]["events"]} 事件）')
    axes[1].imshow(crop);axes[1].set_title(f'放大区（仅显示） {x1-x0}×{y1-y0}')
    axes[2].imshow(crop);axes[2].imshow(overlay(result['density'][y0:y1,x0:x1],(1.,.75,.1)))
    axes[2].set_title(f'密度筛选：{result["counts"]["density_pixels"]} px')
    axes[3].imshow(crop);axes[3].imshow(overlay(result['support'][y0:y1,x0:x1],(.1,.9,.9)))
    for component in result['components']:
        bx0,by0,bx1,by1=component['bbox']
        axes[3].add_patch(Rectangle((bx0-x0-1,by0-y0-1),bx1-bx0+2,by1-by0+2,fill=False,
            edgecolor='lime',lw=1.0))
    axes[3].set_title(f'周期确认：{result["counts"]["period_pixels"]} px / {result["component_count"]} 块')
    axes[0].set_xlim(0,1280);axes[0].set_ylim(720,0);axes[0].axis('off')
    for axis in axes[1:]:axis.axis('off')
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=dict(clips={},parameters=PARAMS,body_box_used=False,labels_used=False,
        grey_reconstruction_used=False,future_events_used=False,
        rotor_count_claimed=False,mechanical_RPM_measured=False)
    for clip in CLIPS:
        started=time.perf_counter();queue=deque();cutoff=clip['t']
        begin=cutoff-.35
        for t,events in native_batches(Path(clip['raw']),begin,.4,False,33333,reader_backend='rolling'):
            queue.append(events.copy())
            while queue and (not len(queue[0]) or queue[0]['t'][-1]<(cutoff-.2)*1e6):queue.popleft()
            if t>=cutoff-1e-9:break
        events=np.concatenate(queue) if queue else np.zeros(0,dtype=DTYPE)
        result=rotor_support(events,cutoff)
        entry=dict(clip=clip,cutoff=cutoff,counts=result['counts'],bbox=result['bbox'],
            components=[{k:v for k,v in c.items()} for c in result['components']],
            elapsed_s=round(time.perf_counter()-started,2))
        reference=frozen_rotors(clip['predictions'],cutoff)
        if reference:
            union,used=ellipse_union(reference['rotors'],1280,720)
            entry['frozen_reference']=dict(mode=reference['mode'],source_t=reference['source_t'],
                rotors=used,overlap=overlap_stats(result,union),
                support_distance=distance_stats(result,reference['rotors']),
                residual=residual_stats(result,reference['rotors']))
            rates=[r for r in (reference.get('rates') or []) if isinstance(r,dict)]
            entry['frequency_check']=[dict(component_pixels=c['pixels'],
                component_frequency_hz=c['median_frequency_hz'],
                alignment=frequency_alignment(c,rates)) for c in result['components']]
            entry['frozen_rates_hz']=[float(r['frequency_hz']) for r in rates
                if r.get('valid') and r.get('frequency_hz')]
        summary['clips'][clip['name']]=entry
        panels(result,f"{clip['name']} @ {cutoff:.3f}s",OUT/f"{clip['name']}_triplet.png")
        print(clip['name'],json.dumps({k:v for k,v in entry.items() if k!='components'},
            ensure_ascii=False)[:600],flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'summary.json')


if __name__=='__main__':main()
