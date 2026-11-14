"""Correct INVALID tree-inclusive outline references, not model parameters.

After visual review of all27 native frames: drone appears in upper-right sky
14.6..16.0, automated whole-image count boxes otherwise follow foliage.
One image-only annotation enclosure is held fixed for ALL27 frames. It is
never given to the detector, forecast, or parameter learner. Not human COM GT.
"""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
from reference_boxes181 import outline_box
from score_rotor164 import score

if __name__=='__main__':
    BASE=ROOT.parent/'output/frozen_tangent287_20261108';OUT=ROOT.parent/'output/reference290_20261108';OUT.mkdir(exist_ok=False)
    enclosure=[700,0,1280,180]
    protocol=dict(scope='evaluation annotation ONLY; NEVER algorithm ROI',enclosure=enclosure,
        annotation_source='assistant visual inspection of all27 original native event overlays; not independent human labels',
        algorithm_candidate_frozen_before_annotation=True,counts_and_area_thresholds_unchanged=True,
        wrong_auto_boxes_not_used_to_tune_or_filter_predictions=True,
        keep_all25_original_query_origins=True,uncropped_frames_preserved=True)
    (OUT/'annotation_protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    records={};images=[]
    for t in np.round(np.arange(10.8,16.0001,.2),1):
        e=window(BASE/'cache',t);e=e[e['t']>=round(t*1e6)-20000]
        ids=e['y'].astype(np.int64)*1280+e['x'];counts=np.bincount(ids,minlength=720*1280).reshape(720,1280)
        x,y,X,Y=enclosure;masked=np.zeros_like(counts);masked[y:Y,x:X]=counts[y:Y,x:X]
        b=outline_box(masked,8);weak=outline_box(masked,4)
        disagreement=None if b is None or weak is None else float(np.linalg.norm((np.array(b[:2])+b[2:]-np.array(weak[:2])-weak[2:])/2))
        records[f'{t:.1f}']=dict(t=float(t),state='present' if b is not None else 'absent',box=b,weaker_outline_box=weak,
            threshold_center_disagreement_px=disagreement,evaluation_enclosure=enclosure,
            physical_COM_or_independent_human_GT=False)
        im=cv2.cvtColor(np.uint8(255-np.clip(counts*20,0,255)),cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im,tuple(enclosure[:2]),tuple(enclosure[2:]),(180,120,0),1)
        if b is not None:cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,160,0),1)
        images.append((t,im,disagreement))
    ref=OUT/'references.json';ref.write_text(json.dumps(dict(records=records,assistant_visual_review_pending=True,protocol=protocol),indent=2),'utf8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for page,start in enumerate(range(0,len(images),9)):
        fig,axes=plt.subplots(3,3,figsize=(14,9))
        for ax in axes.flat:ax.axis('off')
        for ax,(t,im,d) in zip(axes.flat,images[start:start+9]):
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'{t:.1f}s d={d:.1f}px' if d is not None else f'{t:.1f}s no target')
        fig.tight_layout();fig.savefig(OUT/f'review{page}.png',dpi=130);plt.close(fig)
    # Score only AFTER annotation, without changing either prediction file.
    paths=dict(control=BASE/'control.jsonl',biased_bias_prior286=BASE/'tangent.jsonl',
        identifiable289=ROOT.parent/'output/identifiable289_20261108/3_tangent.jsonl')
    origins=np.round(np.arange(10.8,15.6001,.2),1)
    result={name:score(path,ref,origins) for name,path in paths.items()}
    (OUT/'results.json').write_text(json.dumps(dict(scores=result,protocol=protocol,visual_review_pending=True),indent=2),'utf8')
    print({k:v['error'] for k,v in result.items()})
