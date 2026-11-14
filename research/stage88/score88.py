"""Post-inference scoring: independent targets never drive any forecaster."""
from pathlib import Path
import collections, datetime, json, sys
import numpy as np
from inventory88 import ROOT, OUT, save, load, sha, parse, unique_rows
from paired88 import METHODS, SEEDS, prior

SELF=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'


def rows(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]


def summarise(details):
    groups=collections.defaultdict(list)
    for r in details:
        for group in ['all']+(['turn'] if r.get('turn') else [])+(['physical_active'] if r.get('physical_active') else []):
            groups[(r['case'],r['method'],r['horizon'],group)].append(r)
    result=[]
    for (case,method,horizon,group),rr in groups.items():
        valid=[r for r in rr if r['valid']];errors=[r['fde'] for r in valid]
        result.append(dict(case=case,method=method,horizon=horizon,group=group,targets=len(rr),valid=len(valid),
            mean_fde=float(np.mean(errors)) if errors else None,median_fde=float(np.median(errors)) if errors else None,
            p95_fde=float(np.percentile(errors,95)) if errors else None,
            mean_ade=float(np.mean([r['ade'] for r in valid])) if valid and 'ade' in valid[0] else None))
    return result


def self_score(protocol):
    original=load(SELF/'scoring/analysis.json')['details'];details=[]
    for case in protocol['self_cases']:
        pred={round(r['t'],8):r for r in rows(OUT/'self'/case/'forecasts.jsonl')}
        reference=[r for r in original if r['case']==case and r['frontend']=='event_force' and r['model']=='mmae_state_fresh'
            and r['nominal_h'] in [.2,.4] and r['target_state']=='present']
        for ref in reference:
            a=pred.get(round(ref.get('anchor_t',-100),8));valid=bool(a is not None and a['valid'])
            for name in METHODS:
                r=dict(case=case,method=name,horizon=ref['nominal_h'],target_t=ref['target_t'],anchor_t=ref.get('anchor_t'),
                    valid=valid,physical_active=bool(a and a['diagnostics']['fresh']))
                if valid:
                    point=np.array(prior.evaluate(a['coefficients'][name],ref['actual_h']))
                    r['fde']=float(np.linalg.norm(point-ref['target_center']))
                    if name=='event':assert abs(r['fde']-ref['error_px'])<1e-9,(case,r,ref)
                details.append(r)
    return details


def nearest(rr,t,tolerance):
    if not rr:return None
    ts=np.array([r['t'] for r in rr]);j=int(np.searchsorted(ts,t));candidates=[i for i in [j-1,j] if 0<=i<len(rr)]
    k=min(candidates,key=lambda i:abs(ts[i]-t))
    return rr[k] if abs(ts[k]-t)<=tolerance+1e-8 else None


def center(r):
    b=np.array(r['bbox']);return (b[:2]+b[2:])/2


def fred_score(protocol):
    result=[];audit=[]
    for seq in protocol['fred_cases']:
        completion=load(OUT/'fred'/f'sequence{seq}'/'completion.json');assert completion['source_hashes_unchanged']
        truth,bad=parse((OUT/'annotations'/f'{seq}.txt').read_text(encoding='utf-8-sig'));assert not bad
        truth=unique_rows(truth);times=np.array([r['t'] for r in truth]);dropped=collections.Counter();anchors=0
        for p in rows(OUT/'fred'/f'sequence{seq}'/'forecasts.jsonl'):
            t=p['t']
            if abs(t*10-round(t*10))>1e-7 or t<20.6-1e-8 or t>49.6+1e-8:continue
            for h in protocol['scoring']['horizons']:
                target=nearest(truth,t+h,.018)
                if target is None:dropped['no_endpoint']+=1;continue
                left=np.searchsorted(times,t,side='right')-1
                right=np.searchsorted(times,target['t'],side='right')
                if left<0 or t-truth[left]['t']>.05:dropped['no_anchor_truth']+=1;continue
                segment=truth[left:right]
                if len(segment)<3 or max(np.diff([r['t'] for r in segment]))>.101:dropped['truth_gap']+=1;continue
                ids={r['identity'] for r in segment if r['identity'] is not None}
                if len(ids)>1:dropped['identity_change']+=1;continue
                future=[r for r in segment if r['t']>t+1e-9]
                assert future and future[-1]['t']==target['t']
                before=nearest(truth,t-.2,.05);middle=nearest(truth,t+.2,.05)
                turn=False;angle=None
                if h==.4 and before is not None and middle is not None and middle['t']<target['t']:
                    v0=(center(truth[left])-center(before))/max(truth[left]['t']-before['t'],.001)
                    v1=(center(target)-center(middle))/(target['t']-middle['t'])
                    if min(np.linalg.norm(v0),np.linalg.norm(v1))>=10.:
                        angle=float(np.degrees(np.arccos(np.clip(v0@v1/np.linalg.norm(v0)/np.linalg.norm(v1),-1,1))))
                        turn=angle>=30.
                anchors+=1
                for name in METHODS:
                    r=dict(case=f'fred{seq}',method=name,horizon=h,anchor_t=t,target_t=target['t'],actual_h=target['t']-t,
                           valid=p['valid'],turn=turn,turn_angle_deg=angle,physical_active=p['diagnostics']['fresh'],truth_points=len(future))
                    if p['valid']:
                        errors=[float(np.linalg.norm(np.array(prior.evaluate(p['coefficients'][name],g['t']-t))-center(g))) for g in future]
                        r['fde']=errors[-1];r['ade']=float(np.mean(errors));r['normalized_fde']=errors[-1]/np.linalg.norm(np.array(target['bbox'][2:])-target['bbox'][:2])
                    result.append(r)
        audit.append(dict(case=f'fred{seq}',candidate_anchors=anchors,dropped=dict(dropped),**completion))
    return result,audit


def paired_bootstrap(details):
    rr=[r for r in details if r['case'] in ['fred8','fred9'] and r['horizon']==.4 and r['valid']]
    slots=collections.defaultdict(dict)
    for r in rr:slots[(r['case'],r['anchor_t'])][r['method']]=r['fde']
    groups=collections.defaultdict(list)
    for (case,t),values in slots.items():
        assert set(values)==set(METHODS)
        groups[(case,int(t))].append(values)
    block=list(groups.values());rng=np.random.default_rng(8800);result=[]
    for control in ['constant','zero','lag200']+[f'past_shuffle{s}' for s in SEEDS]:
        samples=[]
        for _ in range(1000):
            chosen=rng.integers(len(block),size=len(block));values=[r['event']-r[control] for j in chosen for r in block[j]]
            samples.append(float(np.mean(values)))
        paired=[v['event']-v[control] for v in slots.values()]
        result.append(dict(control=control,anchors=len(paired),blocks=len(block),event_minus_control_px=float(np.mean(paired)),
            block_bootstrap_95_percentile_px=np.percentile(samples,[2.5,97.5]).tolist()))
    return result


def main():
    protocol=load(OUT/'protocol88.json');assert not (OUT/'scores88.json').exists()
    assert all((OUT/'self'/c/'completion.json').is_file() for c in protocol['self_cases'])
    assert all((OUT/'fred'/f'sequence{c}'/'completion.json').is_file() for c in protocol['fred_cases'])
    assert load(OUT/'fred/prefix8/completion.json')['prefix_exact']
    save(OUT/'scoring_protocol88.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),
        protocol_sha256=sha(OUT/'protocol88.json'),rules=protocol['scoring'],before_scoring=True))
    own=self_score(protocol);fred,audit=fred_score(protocol);table=summarise(own+fred)
    test=[r for r in fred if r['case'] in ['fred8','fred9'] and r['horizon']==.4]
    metrics={m:float(np.mean([r['fde'] for r in test if r['method']==m and r['valid']])) for m in METHODS}
    turns=sum(r['turn'] and r['valid'] for r in test if r['method']=='event')
    physics=sum(r['physical_active'] and r['valid'] for r in test if r['method']=='event')
    controls=['constant','zero','lag200'];gain={m:1-metrics['event']/metrics[m] if metrics[m]>0 else 0 for m in controls}
    gain['median_shuffle']=1-metrics['event']/float(np.median([metrics[f'past_shuffle{s}'] for s in SEEDS]))
    per_case={c:{m:float(np.mean([r['fde'] for r in test if r['case']==c and r['method']==m and r['valid']])) for m in METHODS} for c in ['fred8','fred9']}
    criterion=bool(all(v>.05 for v in gain.values()) and turns>=20 and all(v['event']<=v['constant'] and v['event']<=v['zero'] for v in per_case.values()))
    save(OUT/'endpoint_details88.json',dict(self=own,fred=fred))
    result=dict(table=table,fred_audit=audit,test_04_mean_fde=metrics,test_by_case=per_case,test_turn_anchors=turns,
        test_current_physics_anchors=physics,relative_event_gain=gain,predeclared_event_gain_criterion_met=criterion,
        paired_block_bootstrap=paired_bootstrap(fred),self_event_scores_exact78=True,
        conclusion='Independent periodic gain supported in this fixed pilot' if criterion else 'Independent periodic gain NOT established; retain matched negative controls.',
        no_official_full_benchmark=True,no_new_RPM_ground_truth=True,default_promoted=False)
    save(OUT/'scores88.json',result)
    print(json.dumps({k:result[k] for k in ['test_04_mean_fde','test_turn_anchors','test_current_physics_anchors','predeclared_event_gain_criterion_met','conclusion']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
