"""Assistant-reviewed recording4 target regions, references only.

Full-frame automatic count boxes selected trees. Preserve invalid original
scores; coarse annotation regions selected from full native image review are
never read by inference. All25 prediction origins remain fixed.
"""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
from reference_boxes181 import outline_box
from score_rotor164 import score

if __name__=='__main__':
    src=ROOT.parent/'output/frozen324_20261109';out=ROOT.parent/'output/reference323_20261109';out.mkdir(exist_ok=False)
    old=json.loads((src/'references.json').read_text('utf8'));records={}
    # These are coarse target annotations from native full-frame images, not
    # model-selected ROIs and not detector coordinates or runtime patches.
    regions={10.8:[1120,0,1280,260],11.0:[1120,0,1280,260],11.2:[1120,0,1280,260],
        11.4:[1200,300,1280,440],13.6:[1000,400,1250,500],16.0:[1180,150,1280,270]}
    for k,previous in old['records'].items():
        t=previous['t'];r=dict(previous)
        if 14.4<=t<16.:
            r.update(state='absent',box=None,weaker_outline_box=None,visual_review='No drone visible in full native frame; original automatic box was background.')
        elif t in regions:
            e=window(src/'cache',t);e=e[e['t']>=round(t*1e6)-20000];count=np.bincount(e['y'].astype(np.int64)*1280+e['x'],minlength=1280*720).reshape(720,1280)
            x,y,X,Y=regions[t];crop=count[y:Y,x:X]
            def box(level):
                b=outline_box(crop,level);return None if b is None else (np.asarray(b)+[x,y,x,y]).tolist()
            b=box(8);weak=box(4);r.update(box=b,weaker_outline_box=weak,state='present' if b is not None else 'uncertain',
                coarse_visual_target_annotation_xyxy=regions[t],visual_review='Target localization reviewed on full native image; density rule unchanged.')
        else:r['visual_review']='Automatic count8 target box visually corresponds to drone.'
        r['threshold_center_disagreement_px']=None if r.get('box') is None or r.get('weaker_outline_box') is None else float(np.linalg.norm((np.asarray(r['box'][:2])+r['box'][2:]-np.asarray(r['weaker_outline_box'][:2])-r['weaker_outline_box'][2:])/2))
        records[k]=r
    path=out/'references.json';path.write_text(json.dumps(dict(records=records,annotation_scope='weak native visible outline, not COM; assistant coarse localization',
        current_detector_and_future_forecast_not_inputs_to_annotation_extractor=True,assistant_full_frame_review_completed=True,
        original_full_image_references_INVALID_background=True,original_invalid_reference_and_scores_preserved=True),indent=2),'utf8')
    scores={a:score(src/f'{a}.jsonl',path,np.round(np.arange(10.8,15.6001,.2),1)) for a in ('control','inverse320','combined')}
    (out/'results.json').write_text(json.dumps(dict(cases=scores,original216px_score_INVALID=True,
        inference_outputs_unchanged=True,fixed25_origins_preserved=True),indent=2),'utf8');print({a:s['error'] for a,s in scores.items()},flush=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(12,9))
    for ax,(t,region) in zip(axes.flat,regions.items()):
        e=window(src/'cache',t);e=e[e['t']>=round(t*1e6)-20000];counts=np.bincount(e['y'].astype(np.int64)*1280+e['x'],minlength=1280*720).reshape(720,1280);im=np.uint8(255-np.clip(counts*20,0,255));im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        r=records[f'{t:.1f}'];b=r.get('box')
        if b:cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,160,0),1)
        x,y,X,Y=region;ax.imshow(cv2.cvtColor(im[y:Y,x:X],cv2.COLOR_BGR2RGB));ax.set_title(f'{t:.1f}s {r["state"]}')
    for ax in axes.flat:ax.axis('off')
    fig.tight_layout();fig.savefig(out/'corrected_crops.png',dpi=130);plt.close(fig)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
