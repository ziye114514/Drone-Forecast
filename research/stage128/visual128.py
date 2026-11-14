"""Native support, independent-model timelines and actual RAW forecast videos."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage124'),str(ROOT/'research/stage128')]
from study124 import OUT as INPUT,CASES,rows,save
from study128 import OUT
def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=json.loads((INPUT/'availability124.json').read_text('utf-8'));ind=json.loads((ROOT.parent/'output/stage127_independent_support_20261023/availability127.json').read_text('utf-8'))['table']
    cases=list(CASES);x=np.arange(len(cases));width=.24;fig,ax=plt.subplots(figsize=(11,4.8))
    for j,(mode,label,color) in enumerate([('count_global','Original count model','#8b969d'),('occupancy_roi','Occupancy model','#c96bab'),('independent','Independent models, shared KF','#139985')]):
        vals=[next(z for z in (ind if mode=='independent' else a) if z['case']==c and (mode=='independent' or z['mode']==mode))['current'] for c in cases]
        bars=ax.bar(x+(j-1)*width,vals,width,label=label,color=color);ax.bar_label(bars,fontsize=9)
    ax.set_xticks(x,[f'{c}\nn={round(CASES[c]["duration"]*50)}' for c in cases]);ax.set_ylabel('Current qualified model updates');ax.set_ylim(0,290);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'coverage128.png',dpi=150);plt.close(fig)
    rr=rows(INPUT/'natural6/records.jsonl');ii=json.loads((ROOT.parent/'output/stage127_independent_support_20261023/natural6.json').read_text('utf-8'));t=[r['t'] for r in rr]
    fig,axes=plt.subplots(3,1,figsize=(12,6),sharex=True)
    for ax,mode,label in zip(axes,['count_global','occupancy_roi','independent'],['Count model: own learned shape','Occupancy model: own learned shape','Single position filter: prefer count, fill from occupancy']):
        if mode=='independent':
            for name,col in [('count','#7b8792'),('occupancy','#c56caa')]:ax.plot(t,[1 if r['output']['chosen']==name else np.nan for r in ii],'.',color=col,markersize=5,label=name)
            ax.legend(loc='lower right',fontsize=9)
        else:ax.plot(t,[1 if r['outputs'][mode]['center']['current'] else np.nan for r in rr],'.',color='#139985',markersize=5)
        ax.set_ylim(.5,1.5);ax.set_xlim(22,30);ax.set_yticks([]);ax.set_title(label,loc='left',fontsize=10);ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('RAW time (s)');fig.suptitle('natural6: 236 -> 244 current updates; first22.80s ->22.68s');fig.tight_layout();fig.savefig(OUT/'timeline128.png',dpi=150);plt.close(fig)
    if not (OUT/'scores128.json').exists():return
    scores=json.loads((OUT/'scores128.json').read_text('utf-8'))['table'];fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,case in zip(axes,['extension1','extension2','natural6']):
        names=['body_KF','count_model','occupancy_model','mixed_template','independent_models'];ss=[next(z for z in scores if z['case']==case and z['h']==.4 and z['method']==m) for m in names];v=[z['mean_px'] or 0 for z in ss]
        bars=ax.bar(['body KF','count','occupancy','mixed','independent'],v,color=['#b0b8be','#7e8b94','#c86eac','#e9ab54','#159b89']);ax.bar_label(bars,labels=[f'{z["mean_px"]:.2f}' if z['mean_px'] is not None else 'N/A' for z in ss],fontsize=8)
        ax.set_title(f'{case}: paired{ss[0]["paired"]}/{ss[0]["eligible"]}');ax.tick_params(axis='x',labelrotation=35);ax.set_ylim(0,max(max(v)*1.2,1));ax.grid(axis='y',alpha=.2);ax.set_ylabel('Mean nominal0.4s BOX endpoint error (px)')
    fig.tight_layout();fig.savefig(OUT/'future_box128.png',dpi=150);plt.close(fig)
def video(case):
    rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');cap=cv2.VideoCapture(str(OUT/'raw'/case/'forecasts.mp4'));w=int(cap.get(3));h=int(cap.get(4));dest=OUT/(case+'_independent128.mp4');writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),50,(w+440,h));assert writer.isOpened()
    for i,r in enumerate(rr):
        ok,im=cap.read();assert ok;display=np.full((h,w+440,3),248,np.uint8);display[:,:w]=im;q=r['frontends']['delayed_feedback'];worker=q['learned_structure118'];v=worker['support128'];f=v['output']
        for name,col in [('partial_affine3_kf120',(100,100,100)),('support_independent_kf128',(40,165,30))]:
            c=q['coefficients'].get(name)
            if q['valid'] and c:
                p=np.rint([np.array(c['position'])+dt*np.array(c['velocity']) for dt in [0,.1,.2,.3,.4]]).astype(np.int32);cv2.polylines(display,[p],False,col,2);cv2.circle(display,tuple(p[-1]),5,col,2)
        for name,rotors,col in [('count',worker['input_rotors'],(0,170,230)),('occupancy',worker['support126']['occupancy_geometry'].get('rotors',[]),(210,30,175))]:
            for rot in rotors:cv2.ellipse(display,((rot['cx'],rot['cy']),(rot['major'],rot['minor']),rot['angle']),col,1)
        lines=[f'{case} t={r["t"]:.2f}s','RAW-only input + existing NN weights','Offline50fps replay, NOT realtime','Gray: original0.4s model forecast','Green: independent support forecast','Yellow: count rotor observations','Purple: occupancy observations','',f'Selected: {f["chosen"]}',f'Count current: {worker["outputs"]["affine3"]["center"]["current"]}',f'Occupancy current: {v["occupancy_model"]["current"]}',f'One position KF, no switch reset','No fabricated fourth rotor','Unchanged full-four / five-frame gates']
        for j,line in enumerate(lines):cv2.putText(display,line,(w+10,25+j*24),0,.44,(20,40,50),1)
        box=r['body_observation'].get('bbox')
        if box is not None:
            b=np.array(box);lo=np.maximum(np.floor(b[:2]-15).astype(int),0);hi=np.minimum(np.ceil(b[2:]+15).astype(int),[w,h]);crop=display[lo[1]:hi[1],lo[0]:hi[0]]
            if crop.size:
                scale=min(416/crop.shape[1],260/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);display[425:425+crop.shape[0],w+10:w+10+crop.shape[1]]=crop
        writer.write(display)
        if i==min(33,len(rr)-1):cv2.imwrite(str(OUT/(case+'_poster128.png')),display)
    cap.release();writer.release();cap=cv2.VideoCapture(str(dest));n=0
    while True:
        ok,_=cap.read()
        if not ok:break
        n+=1
    cap.release();assert n==len(rr);return dict(case=case,frames=n,path=str(dest))
def main():
    figures();save(OUT/'video_checks128.json',[video(c) for c in ['natural6','extension2']])
if __name__=='__main__':main()
