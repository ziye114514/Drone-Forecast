"""Matched body-only controls: isolate center correction from input-source change."""
from study104 import *
from online93 import Kinematic
from robust23 import Predictor as History


def main():
    verify();reference=load(EXTRA/'scoring/analysis.json')['details'];table=[];audit=[]
    for case in CASES:
        original=rows(BASE/case/'observations_forecasts.jsonl');corrected=load(OUT/'cached'/case/'forecasts.json')
        k=Kinematic();h=History();pred={};times=[]
        for r,q in zip(original,corrected):
            t=r['t'];body=r['body_observation'];z=body.get('center') if body.get('detected') else None
            ck=k.step(t,z);ch,_=h.step(t,z)
            coeff={}
            if q['valid'] and ch['valid']:
                coeff.update(body_only_kf=ck,body_only_history=ch['coefficients']['trusted_acc'],
                    corrected_kf=q['coefficients']['rigid_center_kf'],corrected_history=q['coefficients']['rigid_center_history'])
            pred[round(t,8)]=coeff;times.append(q['rigid_center_diagnostic']['processing_ms'])
        group=[]
        for r in reference:
            if r['case']!=case or r['frontend']!='event_force' or r['model']!='hold' or r['target_state']!='present':continue
            coeff=pred.get(round(r['anchor_t'],8),{}) if r.get('anchor_available') else {}
            from hybrid74 import prior
            for name in ['body_only_kf','body_only_history','corrected_kf','corrected_history']:
                c=coeff.get(name);error=None if c is None else float(np.linalg.norm(np.asarray(prior.evaluate(c,r['actual_h']))-r['target_center']))
                group.append(dict(h=r['nominal_h'],method=name,error=error))
        for horizon in [.1,.2,.4]:
            for method in ['body_only_kf','body_only_history','corrected_kf','corrected_history']:
                g=[r for r in group if r['h']==horizon and r['method']==method];v=[r['error'] for r in g if r['error'] is not None]
                table.append(dict(case=case,h=horizon,method=method,n=len(v),targets=len(g),mean_px=None if not v else float(np.mean(v))))
        if case!='clear6':
            # Neither clip initializes: all body-only / corrected coefficients
            # must be exactly equal, not credited as skeleton improvements.
            for values in pred.values():
                if values:assert values['body_only_kf']==values['corrected_kf'] and values['body_only_history']==values['corrected_history']
        audit.append(dict(case=case,body_control_uses_identical_detection_timestamps=True,
            geometry_added_median_ms=float(np.median(times)),geometry_added_p95_ms=float(np.quantile(times,.95)),
            no_initialization_control_exact=case!='clear6'))
    save(OUT/'matched_controls104.json',dict(table=table,audit=audit,
        source_sha256=sha(__file__),scope='Cached causal body observations, identical filter parameters/timestamps. No new RAW inference.'))
    print(json.dumps([r for r in table if r['h']==.4]),flush=True)


if __name__=='__main__':main()
