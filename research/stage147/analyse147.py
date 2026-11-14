"""Post-inference comparison: emitted forecasts, identical endpoints, no GT input."""
from pathlib import Path
from collections import Counter
import argparse,json,sys,html,hashlib
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT.parent;OUT=WORK/'output/stage147_curvature_split_20261024'
OLD=WORK/'output/stage146_threshold_branches_20261024'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage146'),str(ROOT/'research/stage147')]
from analyse146 import load,score,references,CASES as CASES146
from probe147 import js
CASES=dict(clear=('clear4',400),side=('side4',475),train2=('train2_4',300),fred0=('fred0_4',150))

def draw(im,row):
    for r in row['rotor_branch']['geometry']['observed_rotors']:
        center=tuple(np.rint([r['cx'],r['cy']]).astype(int));axes=tuple(np.maximum(1,np.rint([r['major']/2,r['minor']/2]).astype(int)))
        color=(30,180,20) if r.get('curvature_validated') else (20,120,230)
        cv2.ellipse(im,center,axes,r['angle'],0,360,color,1);cv2.circle(im,center,2,color,-1)
        for pt in r.get('original_arc_points',[]):cv2.circle(im,tuple(np.rint(pt).astype(int)),1,(255,90,10),-1)
    for d in row['rotor_branch'].get('curvature_split',{}).get('components',[]):
        if d['accepted']:
            for cut in d['cuts']:cv2.circle(im,tuple(cut['point']),3,(0,0,255),-1)
    return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',default='');a=p.parse_args()
    summary={};scores={};checks={};allrows={}
    for name,(oldname,n) in CASES.items():
        directory=OUT/(name+a.suffix);rows=load(directory/'predictions.jsonl');oldrows=load(OLD/(oldname+'_final')/'predictions.jsonl');allrows[name]=rows
        assert len(rows)==len(oldrows)==n
        completion=json.loads((directory/'completion.json').read_text());protocol=json.loads((directory/'protocol.json').read_text())
        baseline_difference=0.;orphans=0;paired=0;used=0;proposals=0;recovered=0;lost=0
        for r,b in zip(rows,oldrows):
            assert abs(r['t']-b['t'])<1e-8
            for h,pred in r['predictions'].get('optimized_geometry',{}).items():
                baseline_difference=max(baseline_difference,float(np.max(abs(np.asarray(pred)-b['predictions']['optimized_geometry'][h]))))
            observation=r['rotor_branch'];proposals+=observation['curvature_split']['accepted']
            counts=Counter(q['curvature_parent'] for q in observation['geometry']['observed_rotors'] if q.get('curvature_validated'))
            used+=bool(counts);orphans+=any(v==1 for v in counts.values());paired+=bool(counts) and all(v==2 for v in counts.values())
            current=observation['geometry']['state']=='four';previous=b['rotor_branch']['geometry']['state']=='four'
            recovered+=current and not previous;lost+=previous and not current
            assert r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6)
            assert observation['geometry']['observed_count']==len(observation['geometry']['observed_rotors'])
            assert observation['thrust_proxy']['actual_force_N'] is None
            for q in observation['geometry']['observed_rotors']:
                if q.get('curvature_validated'):
                    assert q['mode']=='arc' and q['inferred_shape'] and q['no_cut_chord']
        refs=references(CASES146[oldname],rows)
        scores[name]=dict(reference=CASES146[oldname]['reference'],reference_count=len(refs),
            baseline=score(refs,rows,'optimized_geometry'),old146_fused=score(refs,oldrows,'rotor_fused'),
            fused=score(refs,rows,'rotor_fused'),physical=score(refs,rows,'rotor_physics'),
            baseline_same_physical_origins=score(refs,rows,'optimized_geometry',require='rotor_physics'))
        summary[name]=dict(frames=n,states=dict(Counter(r['rotor_branch']['geometry']['state'] for r in rows)),
            old146_states=dict(Counter(r['rotor_branch']['geometry']['state'] for r in oldrows)),
            proposed_parent_splits=proposals,split_used_frames=used,paired_used_frames=paired,orphan_frames=orphans,
            four_gained_frames=recovered,four_lost_frames=lost,
            physical_ticks=sum(r['rotor_branch']['thrust_proxy']['valid'] for r in rows),
            active_fusion_ticks=sum(r['rotor_branch']['expert']['weight']>0 for r in rows),
            processing_median_ms=completion['processing_median_ms'],branch_median_ms=completion['rotor_branch_median_ms'],
            baseline_max_coordinate_difference_px=baseline_difference,run_directory=str(directory))
        digests=True
        for source,digest in protocol['sources'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
        video=cv2.VideoCapture(str(directory/'predictions.mp4'));decoded=0
        while True:
            ok,_=video.read()
            if not ok:break
            decoded+=1
        video.release();assert decoded==n
        checks[name]=dict(cutoff=True,partial_arcs_not_complete_rims=True,baseline_exact=baseline_difference==0,
                          sources_unchanged=completion['sources_unchanged'],source_hashes_current=True,video_frames=decoded,
                          no_orphan_split_children=orphans==0)
    # Same captured native frame in all three columns. Existing video is a
    # visualization source only, never an inference input or a scoring target.
    selected=[]
    for name,t,label in [('clear',26.66,'merge'),('side',112.02,'clear_C'),('side',118.12,'side_line')]:
        rows=allrows[name];index=min(range(len(rows)),key=lambda k:abs(rows[k]['t']-t));r=rows[index]
        oldname=CASES[name][0];old=load(OLD/(oldname+'_final')/'predictions.jsonl')[index]
        cap=cv2.VideoCapture(str(OUT/(name+a.suffix)/'predictions.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release();assert ok
        native=frame[:,:1280].copy();roi=r['rotor_branch']['roi'] or [0,0,1280,720]
        lo=np.maximum(np.asarray(roi[:2],int)-8,0);hi=np.minimum(np.asarray(roi[2:],int)+8,[1280,720])
        imgs=[native,draw(native.copy(),old),draw(native.copy(),r)]
        tiles=[]
        for title,img in zip(['Native ON/OFF + baseline overlay','146: high-threshold cores','147: original contour arcs'],imgs):
            crop=img[lo[1]:hi[1],lo[0]:hi[0]];scale=min(2.,570/max(1,crop.shape[1]),320/max(1,crop.shape[0]));crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            tile=np.full((380,590,3),255,np.uint8);tile[50:50+crop.shape[0],10:10+crop.shape[1]]=crop
            cv2.putText(tile,title,(10,25),0,.5,(20,30,40),1);tiles.append(tile)
        cv2.imwrite(str(OUT/f'comparison_{label}.png'),np.concatenate(tiles,axis=1))
        selected.append(dict(case=name,t=r['t'],index=index,old146=old['rotor_branch'],new147=r['rotor_branch']))
    (OUT/'selected_frames147.json').write_text(js(selected),encoding='utf-8')
    (OUT/'summary147.json').write_text(js(summary),encoding='utf-8');(OUT/'scores147.json').write_text(js(scores),encoding='utf-8');(OUT/'checks147.json').write_text(js(checks),encoding='utf-8')
    print(js(dict(summary=summary,scores={n:{k:v['0.4']['mean_px'] for k,v in s.items() if isinstance(v,dict)} for n,s in scores.items()},checks=checks)),flush=True)

if __name__=='__main__':main()
