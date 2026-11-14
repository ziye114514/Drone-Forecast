"""Post-inference native ON/OFF inspection and geometry/forecast timelines."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage123')]
from study123 import OUT,LABELS,CASES,METHODS,rows,save
from evaluate123 import state
from event_drone_forecast.raw_io import open_raw

def native(e,w,h):
    pix=e['y'].astype(np.int32)*w+e['x'];on=np.bincount(pix[e['p']>0],minlength=w*h).reshape(h,w);off=np.bincount(pix[e['p']<=0],minlength=w*h).reshape(h,w)
    im=np.full((h,w,3),255,np.uint8);im[(on>0)&(off==0)]=(40,80,220);im[(off>0)&(on==0)]=(220,170,20);im[(on>0)&(off>0)]=(15,15,15);return im

def snapshots():
    tiles=[];records=[]
    for case in ['extension1','extension2','natural6']:
        c=CASES[case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');labels=json.loads((LABELS/c['labels']).read_text('utf-8'))['labels'];reader=open_raw(c['raw']);h,w=reader.get_size()
        for offset in ([.5,2.,3.5,5.] if case!='natural6' else [.5,2.,4.,6.]):
            r=min(rr,key=lambda r:abs(r['t']-(c['start']+offset)));t=r['t'];begin=round((t-.02)*1e6);end=round(t*1e6);reader.seek_time(begin);e=reader.load_delta_t(20000);e=e[(e['t']>=begin)&(e['t']<end)];im=native(e,w,h)
            # Label-assisted cropping is for post-run inspection only. Prefer
            # the independently annotated target, so a missed NN is visible.
            nearby=min(labels,key=lambda l:abs(l['t']-t));box=None;crop_source='full sensor'
            if abs(nearby['t']-t)<=.26 and nearby.get('bbox') is not None:
                box=nearby['bbox'];crop_source='post-run sparse manual BOX'
            elif r['body_observation'].get('bbox') is not None:box=r['body_observation']['bbox'];crop_source='current NN BOX'
            if box is not None:
                b=np.array(box);lo=np.maximum(np.floor(b[:2]-35).astype(int),0);hi=np.minimum(np.ceil(b[2:]+35).astype(int),[w,h])
            else:lo=np.array([0,0]);hi=np.array([w,h])
            crop=im[lo[1]:hi[1],lo[0]:hi[0]];tile=np.full((320,480,3),245,np.uint8);scale=min(480/max(crop.shape[1],1),235/max(crop.shape[0],1));small=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);tile[:small.shape[0],:small.shape[1]]=small
            worker=state(r);d=worker['validated122']['center'];cv2.putText(tile,f'{case} t={t:.2f}s',(8,258),0,.59,(20,35,45),1)
            cv2.putText(tile,f'NN={r["body_observation"].get("detected",False)} rotors={len(worker["input_rotors"])} model={d["current"]}',(8,282),0,.47,(20,35,45),1)
            cv2.putText(tile,crop_source,(8,307),0,.43,(20,35,45),1);tiles.append(tile)
            records.append(dict(case=case,t=t,events=len(e),crop=np.r_[lo,hi].tolist(),crop_source=crop_source,no_fitted_overlays=True,no_G=True,current_observed_rotors=len(worker['input_rotors']),center_model_current=d['current'],human_visibility=None))
    grid=np.full((960,1920,3),255,np.uint8)
    for i,tile in enumerate(tiles):grid[i//4*320:(i//4+1)*320,i%4*480:(i%4+1)*480]=tile
    cv2.imwrite(str(OUT/'native_visibility_contact123.png'),grid);save(OUT/'native_snapshot_checks123.json',dict(records=records,post_inference_only=True,
        render='20ms polarity occupancy ON red/OFF cyan/both black; label-assisted crops never supplied to inference. No fitted ellipses, G reconstruction or completed motor points.',scope='Visual review, not new human ground truth.'))

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9))
    for ax,case in zip(axes,['extension1','extension2','natural6']):
        rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');t=[r['t'] for r in rr];workers=[state(r) for r in rr]
        ax.step(t,[len(w['input_rotors']) for w in workers],where='post',color='#b1bac0',label='current candidate count')
        ax.plot(t,[4.3 if w['validated122']['center']['current'] else np.nan for w in workers],'.',color='#169a7b',markersize=3,label='selected geometry current')
        ax.plot(t,[-.3 if r['body_observation'].get('detected') else np.nan for r in rr],'.',color='#5680c2',markersize=3,label='NN body current')
        ax.set_ylim(-.7,4.7);ax.set_yticks([0,1,2,3,4]);ax.set_title(case);ax.set_ylabel('Rotor candidates');ax.grid(alpha=.2)
    axes[-1].set_xlabel('RAW time (s)');fig.suptitle('Unchanged122, natural observations, NO artificial rotor removal');fig.legend(*axes[-1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,fontsize=9);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(OUT/'natural_coverage123.png',dpi=150);plt.close(fig)
    scores=json.loads((OUT/'scores123.json').read_text('utf-8'))['table'];fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    names=['body KF','body hold','affine3','conic3','validated'];colors=['#a0a8b0','#79838d','#5680c2','#df8827','#169a7b']
    for ax,case in zip(axes,['extension1','extension2','natural6']):
        ss=[next(z for z in scores if z['case']==case and z['h']==.4 and z['method']==m) for m in METHODS];values=[r['mean_px'] if r['mean_px'] is not None else 0 for r in ss]
        bars=ax.bar(names,values,color=colors);ax.bar_label(bars,labels=[f'{r["mean_px"]:.2f}' if r['mean_px'] is not None else 'N/A' for r in ss],fontsize=8);ax.tick_params(axis='x',labelrotation=35)
        ax.set_title(f'{case}: paired {ss[0]["paired_targets"]}/{ss[0]["eligible_present_targets"]} sparse targets',fontsize=10);ax.set_ylabel('Mean nominal0.4s endpoint error (px)');ax.set_ylim(0,max(max(values)*1.25,1));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same published origins; sparse manual BOX centers, NOT body COM');fig.tight_layout();fig.savefig(OUT/'sparse_forecast123.png',dpi=150);plt.close(fig)

def video(case):
    rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');cap=cv2.VideoCapture(str(OUT/'raw'/case/'forecasts.mp4'));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));dest=OUT/(case+'_validated123.mp4')
    writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w+450,h));assert writer.isOpened();poster=None
    for i,r in enumerate(rr):
        ok,im=cap.read();assert ok;display=np.full((h,w+450,3),248,np.uint8);display[:,:w]=im;q=r['frontends']['delayed_feedback'];worker=state(r);v=worker['validated122'];d=v['center'];s=v['selection']
        for z in worker['input_rotors']:cv2.ellipse(display,((z['cx'],z['cy']),(z['major'],z['minor']),z['angle']),(210,150,20),1)
        for method,color in [('center_body_kf107',(130,130,130)),('partial_conic3_kf120',(35,125,225)),('partial_validated_kf122',(65,165,20))]:
            c=q['coefficients'].get(method)
            if q['valid'] and c:
                points=np.rint([np.array(c['position'])+dt*np.array(c['velocity']) for dt in [0,.1,.2,.3,.4]]).astype(np.int32);cv2.polylines(display,[points],False,color,2);cv2.circle(display,tuple(points[-1]),4,color,1)
        lines=[f'{case} RAW t={r["t"]:.2f}s','Source50fps replay, NOT realtime','Natural input; NO feature removal','Green: validated0.4s forecast','Orange: fixed three-conic branch','Gray: body position/velocity KF','Cyan: current rotor candidates','',f'Body current: {r["body_observation"].get("detected",False)}',f'Current rotor candidates: {len(worker["input_rotors"])}',f'Structure learned: {d.get("structure_ready",False)}',f'Geometry current: {d["current"]}',f'Chosen label: {s["chosen"]}',f'Past validation checks: {s["checks_before"]}',d['measurement']['reason'],'No geometry means body fallback']
        for j,line in enumerate(lines):cv2.putText(display,line,(w+12,25+j*25),0,.45,(25,45,55),1)
        b=r['body_observation'].get('bbox')
        if b is not None:
            b=np.array(b);lo=np.maximum(np.floor(b[:2]-15).astype(int),0);hi=np.minimum(np.ceil(b[2:]+15).astype(int),[1280,720]);crop=display[lo[1]:hi[1],lo[0]:hi[0]]
            if crop.size:
                scale=min(425/crop.shape[1],230/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);display[475:475+crop.shape[0],w+12:w+12+crop.shape[1]]=crop
        writer.write(display)
        if i==len(rr)//2:poster=display.copy()
    cap.release();writer.release();cap=cv2.VideoCapture(str(dest));n=0
    while True:
        ok,_=cap.read()
        if not ok:break
        n+=1
    cap.release();assert n==len(rr);cv2.imwrite(str(OUT/(case+'_poster123.png')),poster);return dict(case=case,frames=n,path=str(dest))

def tracefigure():
    folder=OUT/'extraction_trace2';tiles=[];summary=[]
    for t in [43.5,45.]:
        d=np.load(folder/f'{t:.2f}_counts.npz');audit=json.loads((folder/f'{t:.2f}_audit.json').read_text('utf-8'));on=d['on'];off=d['off'];im=np.full((*on.shape,3),255,np.uint8)
        im[(on>0)&(off==0)]=(40,80,220);im[(off>0)&(on==0)]=(220,170,20);im[(on>0)&(off>0)]=(15,15,15)
        density=cv2.applyColorMap(np.uint8(np.clip(d['density']/(2*float(d['threshold'])),0,1)*255),cv2.COLORMAP_TURBO)
        raw=cv2.cvtColor(d['raw_mask'],cv2.COLOR_GRAY2BGR);closed=cv2.cvtColor(d['closed'],cv2.COLOR_GRAY2BGR)
        counts=dict(t=t,bipolar_occupied_crop_pixels=int(np.sum((on>0)&(off>0))),raw_threshold_crop_pixels=int(np.sum(d['raw_mask']>0)),opened_crop_pixels=int(np.sum(d['opened']>0)),closed_crop_pixels=int(np.sum(d['closed']>0)),threshold=float(d['threshold']),
            scope='Same NN crop plus50px; pixel counts include body/background, not true rotor area.')
        summary.append(counts)
        for j,(picture,label) in enumerate(zip([im,density,raw,closed],['Native ON/OFF occupancy','Bipolar density (count, NOT G)','Threshold >=2.0 (rate units)','After opening+closing'])):
            tile=np.full((300,480,3),245,np.uint8);scale=min(480/picture.shape[1],250/picture.shape[0]);small=cv2.resize(picture,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);tile[:small.shape[0],:small.shape[1]]=small
            cv2.putText(tile,f'{t:.2f}s {label}',(7,277),0,.43,(20,35,45),1);tiles.append(tile)
    grid=np.full((600,1920,3),255,np.uint8)
    for i,tile in enumerate(tiles):grid[i//4*300:(i//4+1)*300,i%4*480:(i%4+1)*480]=tile
    cv2.imwrite(str(OUT/'extraction_funnel123.png'),grid);save(OUT/'extraction_funnel123.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    figures();snapshots();save(OUT/'video_checks123.json',[video(c) for c in ['extension1','extension2','natural6']])
