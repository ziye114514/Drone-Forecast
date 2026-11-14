"""Analysis is separate from frozen inference; never feed truth into RAW runs."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import OUT,ROOT,save,csvsave,verify
from event_drone_forecast.motion import extrapolate

def readcsv(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def traces(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]

def group_metrics(rows,keys,value):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    output=[]
    for key,q in groups.items():
        ok=[r for r in q if r['accepted']=='True'];errors=[float(r[value]) for r in ok if r.get(value)]
        output.append(dict(zip(keys,key),cases=len(q),accepted=len(ok),coverage=len(ok)/len(q),
            median_error_deg=float(np.median(errors)) if errors else None,
            mean_error_deg=float(np.mean(errors)) if errors else None,p95_error_deg=float(np.percentile(errors,95)) if errors else None,
            accepted_wrong_over10deg=sum(e>10 for e in errors),accepted_risk_over10deg=sum(e>10 for e in errors)/len(errors) if errors else None))
    return output

def modules():
    verify();(OUT/'figures').mkdir(exist_ok=True)
    single=readcsv(OUT/'single_cases.csv');multi=readcsv(OUT/'multi_cases.csv')
    sm=group_metrics(single,['method','arc_deg'],'best_candidate_error_deg');mm=group_metrics(multi,['method','condition','focal_scale'],'error_deg')
    detailed=group_metrics(single,['method','arc_deg','chord','diameter_px'],'best_candidate_error_deg')
    csvsave(OUT/'single_metrics.csv',sm);csvsave(OUT/'single_detailed_metrics.csv',detailed);csvsave(OUT/'multi_metrics.csv',mm)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in ('frozen_geometry','frozen_plus_audit','ams_plus_audit'):
        q=sorted([r for r in sm if r['method']==method],key=lambda r:float(r['arc_deg']),reverse=True)
        x=[float(r['arc_deg']) for r in q]
        axes[0].plot(x,[r['coverage']*100 for r in q],'-o',label=method)
        axes[1].plot(x,[r['accepted_risk_over10deg']*100 if r['accepted_risk_over10deg'] is not None else np.nan for r in q],'-o',label=method)
    axes[0].set(ylabel='Accepted pose coverage(%)',xlabel='Visible arc(deg)');axes[1].set(ylabel='Wrong>10deg among accepted(%)',xlabel='Visible arc(deg)')
    for ax in axes:ax.invert_xaxis();ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Single ellipse: sensitivity gating cannot certify short arcs');fig.tight_layout();fig.savefig(OUT/'figures/arc_risk_coverage.png',dpi=160);plt.close(fig)
    conditions=['clean','one_arc120','two_arcs180','one_wrong_plane','missing_one','missing_two','all_arcs120']
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for j,method in enumerate(('frozen_up_prior','joint_all','joint_audit')):
        q=[next(r for r in mm if r['method']==method and r['condition']==c and float(r['focal_scale'])==1) for c in conditions]
        x=np.arange(len(q))+(j-1)*.25
        axes[0].bar(x,[r['median_error_deg'] if r['median_error_deg'] is not None else np.nan for r in q],.25,label=method)
        axes[1].bar(x,[r['coverage']*100 for r in q],.25,label=method)
    axes[0].set_ylabel('Median accepted normal error(deg)');axes[1].set_ylabel('Accepted coverage(%)')
    for ax in axes:ax.set_xticks(range(len(conditions)),conditions,rotation=35,ha='right',fontsize=8);ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Calibrated synthetic four-circle normals; acceptance sets differ');fig.tight_layout();fig.savefig(OUT/'figures/multi_normal.png',dpi=160);plt.close(fig)
    cfg=json.loads((OUT/'protocol.json').read_text())['temporal'];methods=cfg['methods'];details=[];seed_errors={m:[] for m in methods}
    for seed in cfg['seeds']:
        a=np.load(OUT/f'temporal/seed_{seed}.npz');times=a['times'];truth=a['true_position']
        for h_index,h in enumerate(cfg['horizons_s']):
            offset=round(h/.02);n=min(len(times),len(truth)-offset)
            errors=np.linalg.norm(a['prediction'][:n,:,h_index]-truth[offset:offset+n,None,:],axis=-1)
            for j,method in enumerate(methods):
                details.append(dict(seed=seed,horizon_s=h,method=method,targets=n,error_px=float(np.mean(errors[:,j]))))
                if h==.4:seed_errors[method].append(float(np.mean(errors[:,j])))
    csvsave(OUT/'temporal_seed_metrics.csv',details);temporal=[]
    for method in methods:
        for h in cfg['horizons_s']:
            q=[r['error_px'] for r in details if r['method']==method and r['horizon_s']==h]
            temporal.append(dict(method=method,horizon_s=h,seeds=len(q),mean_error_px=float(np.mean(q)),seed_sd_px=float(np.std(q,ddof=1))))
    difference=np.array(seed_errors['joint_temporal_audit'])-seed_errors['frozen_axis'];rng=np.random.default_rng(832)
    samples=difference[rng.integers(0,len(difference),(2000,len(difference)))].mean(axis=1)
    comparison=dict(new_minus_frozen_error_px=float(difference.mean()),paired_seed_bootstrap95=np.percentile(samples,[2.5,97.5]).tolist(),
        caveat='Ten noise repetitions of one controlled trajectory with current oracle p/v and known gains; not independent flights.')
    a=np.load(OUT/f'temporal/seed_{cfg["seeds"][0]}.npz');n=101;t=a['times'][:n];truth=a['true_position'][20:121]
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for j,method in enumerate(methods):
        axes[0].plot(t,np.linalg.norm(a['prediction'][:n,j,-1]-truth,axis=-1),label=method)
        if j:axes[1].plot(a['times'],a['normal_error_deg'][:,j],label=method)
    axes[0].set_ylabel('0.4s endpoint error(px)');axes[1].set(ylabel='Normal axis error(deg)',xlabel='Time(s)')
    for ax in axes:
        ax.axvspan(.8,1.8,color='#e6b77b',alpha=.2);ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Preselected seed1000: controlled current-state diagnostic');fig.tight_layout();fig.savefig(OUT/'figures/temporal_mechanism.png',dpi=160);plt.close(fig)
    save(OUT/'module_analysis.json',dict(single=sm,multi=mm,temporal=temporal,temporal_pair=comparison,
        short_arc_confidence_failure_retained=True,no_real_flight_pose_score=True))
    print(json.dumps(dict(controlled_temporal=comparison)),flush=True)

def raw():
    protocol=json.loads((OUT/'raw_v2/protocol.json').read_text());details=[];summaries=[];absence=[];oldroot=OUT.parent/'paper_reproduction_v030'
    for case,meta in protocol['cases'].items():
        folder=OUT/'raw_v2'/case;summary=json.loads((folder/'summary.json').read_text());summaries.append(dict(case=case,**summary))
        observations=traces(folder/'observations.jsonl');sensor_times=np.array([r['t'] for r in observations])
        labels=json.loads((oldroot/f'evaluation_data/{"train2" if case=="train2" else "train6"}.json').read_text())['labels']
        for arm in ('frozen','consensus','consensus_audit'):
            records=traces(folder/arm/'forecasts.jsonl');times=np.array([r['t'] for r in records])
            if case in ('train6','train2') and arm=='frozen':
                previous=traces(oldroot/f'runs/{case}/forecasts.jsonl');assert len(records)==3000
                for a,b in zip(records,previous):
                    assert a['active']==b['active']
                    if a['states']:
                        for m in ('cv','paper_nominal','paper_adaptive','paper_fused'):assert a['states'][m]==b['states'][m],(case,a['t'],m)
            absent=[]
            for label in labels:
                target=label['t']
                if not meta['start']<target<=meta['start']+meta['duration']:continue
                if label['state']=='absent':
                    k=np.searchsorted(times,target,side='right')-1
                    if k>=0:absent.append(records[k]['active'])
                    continue
                if label['state']!='present':continue
                center=(np.array(label['bbox'])[:2]+np.array(label['bbox'])[2:])/2
                for h in (.1,.2,.4):
                    k=np.searchsorted(times,target-h,side='right')-1
                    if k<0:continue
                    q=records[k];four=observations[k]['sensing']['four_visible']
                    for method in ('cv','paper_nominal','paper_adaptive','paper_fused'):
                        r=dict(case=case,arm=arm,method=method,horizon_s=h,target_t=target,anchor_t=times[k],
                            available=q['active'],physics_active=q['physics_active'],four_visible=four)
                        if q['active']:
                            prediction=extrapolate(q['states'][method],target-times[k]);cv=extrapolate(q['states']['cv'],target-times[k])
                            r.update(error_px=float(np.linalg.norm(prediction-center)),equals_cv=bool(np.linalg.norm(prediction-cv)<1e-8))
                        details.append(r)
            absence.append(dict(case=case,arm=arm,absent_labels=len(absent),active_when_absent=sum(absent)))
    metrics=[]
    for case in protocol['cases']:
        for arm in ('frozen','consensus','consensus_audit'):
            for method in ('cv','paper_nominal','paper_adaptive','paper_fused'):
                for h in (.1,.2,.4):
                    for subset in ('all','four_visible'):
                        q=[r for r in details if r['case']==case and r['arm']==arm and r['method']==method and r['horizon_s']==h and (subset=='all' or r['four_visible'])]
                        ok=[r for r in q if r['available']]
                        metrics.append(dict(case=case,arm=arm,method=method,horizon_s=h,subset=subset,targets=len(q),available=len(ok),
                            coverage=len(ok)/len(q) if q else None,error_px=float(np.mean([r['error_px'] for r in ok])) if ok else None,
                            physics_at_scored_anchors=sum(r['physics_active'] for r in ok),cv_equal_endpoints=sum(r['equals_cv'] for r in ok)))
    csvsave(OUT/'raw_endpoint_details.csv',details);csvsave(OUT/'raw_metrics.csv',metrics)
    save(OUT/'raw_analysis.json',dict(metrics=metrics,summaries=summaries,absence=absence,
        frozen_replays_exact_for=['train6','train2'],labels_used_only_posthoc=True,
        caveat='Sparse existing human boxes, earlier-used development clips. Side run starts108s cold; compare its three arms only. No real pose/RPM truth.'))
    print(json.dumps([r for r in metrics if r['method']=='paper_fused' and r['horizon_s']==.4 and r['subset']=='all']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['modules','raw']);a=parser.parse_args();globals()[a.action]()
