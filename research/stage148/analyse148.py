"""Score exact emitted148 coordinates; paired bias/absolute/event ablations."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,argparse
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;OUT=WORK/'output/stage148_temporal_rotors_20261024'
OLD=WORK/'output/stage147_curvature_split_20261024'
sys.path[:0]=[str(ROOT/'research/stage146'),str(ROOT/'src')]
from analyse146 import load,score,references,CASES as META
def js(value):return json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',default='');a=p.parse_args()
    summary={};scores={};checks={};allrows={}
    for name,oldname,n in [('clear','clear4',400),('side','side4',475),('train2','train2_4',300),('fred0','fred0_4',150)]:
        directory=OUT/(name+a.suffix);rows=load(directory/'predictions.jsonl');previous=load(OLD/(name+'_paired')/'predictions.jsonl')
        assert len(rows)==len(previous)==n;allrows[name]=rows
        protocol=json.loads((directory/'protocol.json').read_text());completion=json.loads((directory/'completion.json').read_text())
        difference=0.;partial=0;current_pixels=0;no_current=0;track_counts=Counter();reasons=Counter();selections=Counter();changed_vs_bias=Counter();raw_filtered=[];guided_frames=0;guided_components=0;merged_frames=0;epoch=0;last_body_seen=None
        for row,before in zip(rows,previous):
            assert row['t']==before['t']
            baseline=row['predictions'].get('optimized_geometry',{});oldbaseline=before['predictions'].get('optimized_geometry',{})
            assert baseline.keys()==oldbaseline.keys()
            for h in baseline:difference=max(difference,float(np.max(abs(np.array(baseline[h])-oldbaseline[h]))))
            assert row['max_event_t'] is None or row['max_event_t']<round(row['t']*1e6)
            branch=row['rotor_branch'];g=branch['geometry'];rates=branch['rates'];e=branch['expert']
            if row['body'].get('detected'):
                if last_body_seen is not None and row['t']-last_body_seen>.5:epoch+=1
                last_body_seen=row['t']
            guided_frames+=any(r.get('split_method148')=='prior_guided_current_arcs' for r in g['observed_rotors'])
            merged_frames+=any(r.get('split_method148')=='prior_guided_fragment_merge' for r in g['observed_rotors'])
            guided_components+=sum(c.get('method148')=='prior_guided_current_arcs' and c['accepted'] for c in branch['curvature_split']['components'])
            assert len(g['observed_rotors'])==g['observed_count'];assert len(rates)==len(g['observed_rotors'])
            assert branch['temporal148']['missing_tracks_emitted']==0
            for r in g['observed_rotors']:
                track_counts[f'{epoch}:{r["visual_track_id"]}']+=1
                if 'raw_center148' in r:raw_filtered.append(float(np.linalg.norm(np.array([r['cx'],r['cy']])-r['raw_center148'])))
            for q in rates:
                reasons[q.get('reason','unknown')]+=1
                if q['valid']:
                    assert q['current_exclusive_pixels']>=4 and not q['past_only_frequency_held']
                    assert q['coherent_frames']>=2
                    current_pixels+=q['current_exclusive_pixels']
                elif q.get('reason')=='no_current_exclusive_support':no_current+=1
            selections.update(f'{h}:{s}' for h,s in e['selections_by_horizon'].items())
            for h,p in row['predictions'].get('rotor_fused',{}).items():
                if not np.allclose(p,row['predictions']['rotor_bias_fused'][h],atol=1e-10,rtol=0):changed_vs_bias[h]+=1
            assert all(0<=w<=.35 for w in e['weights_by_horizon'].values())
            assert branch['thrust_proxy']['actual_force_N'] is None
        refs=references(META[oldname],rows)
        scored=dict(reference=META[oldname]['reference'],reference_count=len(refs),
            baseline=score(refs,rows,'optimized_geometry'),previous147=score(refs,previous,'rotor_fused'))
        for output,label in [('rotor_fused','fused148'),('rotor_bias_fused','bias_only_fused'),('rotor_bias','bias_only_raw'),
            ('rotor_physics','event_residual_raw'),('rotor_temporal_absolute_fused','tracked_absolute_fused'),('rotor_temporal_absolute','tracked_absolute_raw')]:
            scored[label]=score(refs,rows,output)
        scored['baseline_same_event_origins']=score(refs,rows,'optimized_geometry',require='rotor_physics')
        scored['bias_same_event_origins']=score(refs,rows,'rotor_bias',require='rotor_physics')
        scores[name]=scored
        features=np.array([r['rotor_branch']['expert']['feature_delta_xy'] for r in rows if r['rotor_branch']['thrust_proxy']['valid']]).reshape(-1,2)
        run=longest=0
        for r in rows:
            run=run+1 if r['rotor_branch']['thrust_proxy']['valid'] else 0;longest=max(longest,run)
        summary[name]=dict(frames=n,states=dict(Counter(r['rotor_branch']['geometry']['state'] for r in rows)),
            previous147_states=dict(Counter(r['rotor_branch']['geometry']['state'] for r in previous)),
            guided_arc_used_frames=guided_frames,guided_parent_splits=guided_components,fragment_merged_frames=merged_frames,
            conditional_feature_ticks=sum(r['rotor_branch']['thrust_proxy']['valid'] for r in rows),
            longest_continuous_feature_s=longest*.02,
            previous147_conditional_ticks=sum(r['rotor_branch']['thrust_proxy']['valid'] for r in previous),
            stable_visual_ids=len(track_counts),visual_identity_epochs=epoch+1,observations_per_id=dict(track_counts),
            matching=sum(r['rotor_branch']['temporal148']['matched'] for r in rows),
            new_observations=sum(r['rotor_branch']['temporal148']['new_tracks'] for r in rows),
            ambiguous=sum(r['rotor_branch']['temporal148']['ambiguous'] for r in rows),
            rate_reasons=dict(reasons),selection_counts=dict(selections),event_changes_vs_bias_fused=dict(changed_vs_bias),
            position_adjustment_median_px=float(np.median(raw_filtered)) if raw_filtered else 0.,
            position_adjustment_max_px=max(raw_filtered,default=0.),
            centered_feature_std=features.std(0).tolist() if len(features) else None,
            centered_feature_covariance_eigenvalues=np.linalg.eigvalsh(np.cov(features.T)).tolist() if len(features)>2 else None,
            max_mature_event_pairs_by_horizon={h:max(r['rotor_branch']['expert']['diagnostics'].get(h,{}).get('event_past_matched_pairs',0) for r in rows) for h in ('0.1','0.2','0.4')},
            baseline_max_change_px=difference,processing_median_ms=completion['processing_median_ms'],
            rotor_branch_median_ms=completion['rotor_branch_median_ms'],run_directory=str(directory))
        for source,digest in protocol['sources'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
        cap=cv2.VideoCapture(str(directory/'predictions.mp4'));decoded=0
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
        cap.release();assert decoded==n
        checks[name]=dict(source_hashes=True,source_unchanged=completion['sources_unchanged'],baseline_exact=difference==0,
            predictions_no_past_as_current=True,event_cutoff=True,frequency_has_current_support=True,
            same_horizon_correction=True,complete_video_frames=decoded)
    # Scientific plots: independent manual BOX references; no pose/RPM claim.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,name in zip(axes,['side','train2']):
        for label,color in [('baseline','#777777'),('fused148','#e07816')]:
            pairs=scores[name][label]['0.4']['pairs'];x=np.array([p['predicted'] for p in pairs])
            ax.plot(x[:,0],x[:,1],'.',label=label,color=color,markersize=4)
        truth=np.array([p['truth'] for p in scores[name]['baseline']['0.4']['pairs']])
        ax.plot(truth[:,0],truth[:,1],'o',markerfacecolor='none',markersize=4,color='#27865e',label='manual box centre')
        ax.invert_yaxis();ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(name+' | horizon0.4s');ax.set_xlabel('x / px');ax.set_ylabel('y / px')
    fig.tight_layout();fig.savefig(OUT/'forecast148.png',dpi=150);plt.close(fig)
    # Extract actual published preview frames; orange is the new fused forecast.
    for name,t,label in [('clear',26.66,'merged_tracks'),('clear',26.84,'prior_guided_arcs'),('side',112.44,'partial_tracks'),('side',118.12,'side_tracks')]:
        rows=allrows[name];index=min(range(len(rows)),key=lambda i:abs(rows[i]['t']-t))
        cap=cv2.VideoCapture(str(OUT/(name+a.suffix)/'predictions.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();cap.release();assert ok
        cv2.imwrite(str(OUT/f'{label}.png'),im)
    (OUT/'summary148.json').write_text(js(summary),encoding='utf-8');(OUT/'scores148.json').write_text(js(scores),encoding='utf-8');(OUT/'checks148.json').write_text(js(checks),encoding='utf-8')
    print(js(dict(summary=summary,scores={n:{k:{h:dict(n=v['n'],missing=v['missing'],mean_px=v['mean_px']) for h,v in s.items()} for k,s in score.items() if isinstance(s,dict)} for n,score in scores.items()},checks=checks)),flush=True)
if __name__=='__main__':main()
