"""Retrospective root accounting of487; no annotation enters inference.

Past weak-outline derivatives are uncertain diagnostic references, not motion
truth or evidence that a future pilot command was predictable/unpredictable.
"""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from extend_inverse321 import references
from compare_final_pipeline import reference_at
from physical_publication import evaluate

def past_fit(refs,t):
    chosen=[r for r in refs if t-.60001<=r['t']<=t+1e-8 and r.get('state')=='present' and r.get('box')]
    if len(chosen)<4:return None
    times=np.asarray([r['t']-t for r in chosen]);
    if np.ptp(times)<.5999 or np.max(np.diff(times))>.20001:return None
    X=np.c_[np.ones(len(times)),times,.5*times*times]
    y=np.array([(np.asarray(r['box'][:2])+r['box'][2:])/2 for r in chosen]);b=np.linalg.lstsq(X,y,rcond=None)[0]
    variance=max(36.,float(np.sum((y-X@b)**2)/max(1,2*(len(times)-3))))
    C=np.linalg.inv(X.T@X)*variance
    return b,C,max(r['t'] for r in chosen)

if __name__=='__main__':
    out=ROOT.parent/'output/root_audit494_20261112';out.mkdir(exist_ok=False)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    cases={};allrows=[]
    for n in ('13','15','16','11','3','4'):
        path=ROOT.parent/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl'
        rows=[json.loads(s) for s in path.read_text('utf8').splitlines()]
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        refs=sorted(json.loads(ref.read_text('utf8'))['records'].values(),key=lambda q:q['t'])
        ids=sorted(q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']);rr=[]
        for t in ids:
            r=[q for q in rows if q['t']<=t+1e-9][-1];age=t-r['t'];target=reference_at(refs,t+.4)
            p=np.asarray(evaluate(r,[age])[str(float(age))]);v=np.asarray(r['coefficient']['velocity'])
            pred=np.asarray(evaluate(r,[age+.4])[str(float(age+.4))]);error=pred-target
            item=dict(recording=n,t=t,error_px=float(np.linalg.norm(error)),position_now=p.tolist(),
                state_velocity=v.tolist(),prediction=pred.tolist(),target=target.tolist(),
                mode=r['structure']['mode'],motion_method=(r['motion'].get('estimate') or {}).get('method'),
                current_position_error_px=None if reference_at(refs,t) is None else float(np.linalg.norm(p-reference_at(refs,t))))
            herrors={}
            for h in (.1,.2,.4):
                q=reference_at(refs,t+h)
                if q is not None:herrors[str(h)]=float(np.linalg.norm(np.asarray(evaluate(r,[age+h])[str(float(age+h))])-q))
            item['horizon_errors']=herrors
            fit=past_fit(refs,t)
            if fit:
                b,C,last=fit;assert last<=t+1e-8
                components=dict(position=p-b[0],velocity=.4*(v-b[1]),
                    continuation=pred-p-.4*v-.08*b[2],later_reference_innovation=b[0]+.4*b[1]+.08*b[2]-target)
                assert np.linalg.norm(sum(components.values())-error)<1e-7
                unit=error/max(np.linalg.norm(error),1e-9);vse=float(np.sqrt(C[1,1]));hrow=np.array([1.,.4,.08])
                item['past_reference_diagnostic']=dict(maximum_reference_t=last,velocity=b[1].tolist(),
                    acceleration=b[2].tolist(),velocity_sigma_proxy_px_s=vse,
                    future_sigma_proxy_px=float(np.sqrt(hrow@C@hrow+36.)),
                    signed_error_projection_px={k:float(a@unit) for k,a in components.items()},
                    component_vectors_px={k:a.tolist() for k,a in components.items()},
                    weak_reference_not_independent_COM_or_velocity_truth=True)
            rr.append(item);allrows.append(item)
        good=[r for r in rr if 'past_reference_diagnostic' in r]
        cases[n]=dict(n=len(rr),mean=float(np.mean([r['error_px'] for r in rr])),records=rr,
            diagnostic_n=len(good),signed_mean={k:float(np.mean([r['past_reference_diagnostic']['signed_error_projection_px'][k] for r in good])) for k in ('position','velocity','continuation','later_reference_innovation')})
    good=[r for r in allrows if 'past_reference_diagnostic' in r]
    payload=dict(cases=cases,n=len(allrows),mean=float(np.mean([r['error_px'] for r in allrows])),
        diagnostic_n=len(good),diagnostic_mean=float(np.mean([r['error_px'] for r in good])),
        signed_mean={k:float(np.mean([r['past_reference_diagnostic']['signed_error_projection_px'][k] for r in good])) for k in ('position','velocity','continuation','later_reference_innovation')},
        no_runtime_annotation_access=True,retrospective_only=True,reserved_video_not_read=True)
    (out/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False),'utf8')
    print(json.dumps({k:v for k,v in payload.items() if k!='cases'},indent=2))
    for n,s in cases.items():print(n,s['diagnostic_n'],s['signed_mean'])
    for r in sorted(allrows,key=lambda r:-r['error_px'])[:12]:
        print(r['recording'],r['t'],round(r['error_px'],2),r['horizon_errors'],
              r.get('past_reference_diagnostic',{}).get('signed_error_projection_px'))
