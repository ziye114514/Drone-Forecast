"""Dense raw-outline diagnostic of an already exposed early-error example.

Fixed visualization ROI only, never an inference rule or new scoring label.
No parameter changes and no read of the reserved acquisition.
"""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches

def center(count,floor):
    k,lab,stats,_=cv2.connectedComponentsWithStats((count>=floor).astype(np.uint8),8)
    ids=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=16);ids=ids[ids!=0]
    if not len(ids):return None
    q=stats[ids];lo=q[:,:2].min(0);hi=(q[:,:2]+q[:,2:4]).max(0)
    return ((lo+hi-1)/2+[880,340]).tolist()

if __name__=='__main__':
    out=ROOT.parent/'output/early_error495_20261112';out.mkdir(exist_ok=False)
    x0,y0,X,Y=880,340,1280,470;records=[];images=[];cache=[]
    old=[json.loads(s) for s in (ROOT.parent/'output/raw_local_contrast487_20261112/4_local_contrast/predictions.jsonl').read_text('utf8').splitlines()]
    mapping={round(r['t']*1e6):r for r in old};chosen=[11.7,11.8,11.9,12.,12.1,12.2,12.3,12.4]
    selected={min(mapping,key=lambda q:abs(q-z*1e6)) for z in chosen}
    for t,all_events in native_batches('E:/dataset/experiment/4/media/event.raw',10.,2.4,False,33333,reader_backend='cooperative'):
        if t<11.5:continue
        end=round(t*1e6);e=all_events[(all_events['x']>=x0)&(all_events['x']<X)&(all_events['y']>=y0)&(all_events['y']<Y)].copy()
        np.save(out/f'{end}.npy',e);cache.append(dict(t=t,file=f'{end}.npy',events=len(e)))
        recent=e[e['t']>=end-20000];f=(recent['y'].astype(int)-y0)*(X-x0)+recent['x'].astype(int)-x0
        count=np.bincount(f,minlength=(Y-y0)*(X-x0)).reshape(Y-y0,X-x0)
        r=mapping.get(end);est=(r['motion'].get('estimate') or {}) if r else {};dt=r['motion'].get('dt_s') if r else None
        item=dict(t=t,outline8=center(count,8),outline4=center(count,4),
            match_velocity=None if not dt or 'dx' not in est else [est['dx']/dt,est['dy']/dt],
            state_velocity=None if not r or not r['coefficient'] else r['coefficient']['velocity'],
            state_acceleration=None if not r or not r['coefficient'] else r['coefficient']['acceleration'])
        # Display raw data without native mask; boxes are diagnostic weak outlines.
        if end in selected or abs(t-12.4)<1e-8:
            rgb=np.uint8(255-255*np.minimum(count/16.,1.))[:,:,None].repeat(3,2)
            images.append((t,rgb.copy()))
        records.append(item)
    (out/'events_manifest.json').write_text(json.dumps(dict(roi=[x0,y0,X,Y],fixed_diagnostic_region_not_inference=True,frames=cache),indent=2),'utf8')
    (out/'records.json').write_text(json.dumps(dict(records=records,dense_weak_outline_not_velocity_truth=True,
        existing_raw4_only=True,reserved_video_not_read=True,global102_score_unchanged=True),indent=2),'utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(4,2,figsize=(12,9),layout='constrained')
    for ax,(t,im) in zip(axes.flat,images):ax.imshow(im);ax.set_title(f'{t:.6f}s，原始20ms事件数');ax.axis('off')
    for ax in list(axes.flat)[len(images):]:ax.axis('off')
    fig.savefig(out/'raw_sequence.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for name in ('outline8','outline4'):
        good=[r for r in records if r[name] is not None]
        axes[0].plot([r['t'] for r in good],[r[name][0] for r in good],'.-',label=name+'弱轮廓x')
    for name in ('match_velocity','state_velocity'):
        good=[r for r in records if r[name] is not None]
        axes[1].plot([r['t'] for r in good],[r[name][0] for r in good],'.-',label=name+' x')
    axes[0].set_ylabel('弱轮廓中心x / px');axes[1].set_ylabel('速度x / px/s')
    for ax in axes:ax.axvline(12.,color='grey',linestyle='--');ax.legend();ax.set_xlabel('源时间 / s')
    fig.savefig(out/'dense_motion.png',dpi=150);plt.close(fig)
    print('SAVED',len(records),'packets',out,flush=True)
