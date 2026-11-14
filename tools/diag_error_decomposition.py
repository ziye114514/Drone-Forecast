"""Where does the 0.4s error come from: current velocity, or the extrapolated acceleration?

Uses the frozen per-query state already saved by the stage164 run (position,
velocity, acceleration) and the manual whole-body box centres. No new inference,
no future event, and no paramter is fitted: the three predictors are fixed forms.
"""
import json,pathlib,bisect
import numpy as np
BASE=pathlib.Path(r"D:\drone forecast\output\side_temporal164_20261101")
LAB=pathlib.Path(r"D:\drone forecast\output\dense_labels_side_110_1195\labels.json")
TAU=0.4


def reference_at(records,t):
    stamps=[r['t'] for r in records];i=bisect.bisect_left(stamps,t)
    if i<len(records) and abs(stamps[i]-t)<1e-7:pair=[records[i],records[i]]
    elif 0<i<len(records) and stamps[i]-stamps[i-1]<=.1500001:pair=[records[i-1],records[i]]
    else:return None
    if any(r.get('state')!='present' or not r.get('box') for r in pair):return None
    centres=[(np.asarray(r['box'][:2])+np.asarray(r['box'][2:]))/2 for r in pair]
    alpha=0. if pair[0]['t']==pair[1]['t'] else (t-pair[0]['t'])/(pair[1]['t']-pair[0]['t'])
    return (1-alpha)*centres[0]+alpha*centres[1]


def main():
    diagnosis=json.loads((BASE/'error_cause_diagnosis.json').read_text('utf-8'))
    records=sorted(json.loads(LAB.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    rows=[]
    for item in diagnosis:
        t=item['t'];p=np.asarray(item['p'],float);v=np.asarray(item['v'],float)
        a=np.asarray(item['a'],float);truth=reference_at(records,t+TAU)
        if truth is None:continue
        cv=p+v*TAU
        ca=p+v*TAU+.5*a*TAU*TAU
        # fixed-form "acceleration halved / zeroed" probes, not fitted
        rows.append(dict(t=t,error_cv=float(np.linalg.norm(cv-truth)),
            error_ca=float(np.linalg.norm(ca-truth)),
            error_no_a_but_mean_v=None,
            speed_px_s=float(np.linalg.norm(v)),accel_px_s2=float(np.linalg.norm(a)),
            displacement_px=float(np.linalg.norm(v)*TAU),
            accel_contribution_px=float(.5*np.linalg.norm(a)*TAU*TAU)))
    def summarise(values):
        values=[v for v in values if v is not None]
        return dict(n=len(values),mean=float(np.mean(values)),
            median=float(np.median(values)),p95=float(np.percentile(values,95)))
    summary=dict(queries=len(rows),
        error_cv=summarise([r['error_cv'] for r in rows]),
        error_ca=summarise([r['error_ca'] for r in rows]),
        mean_speed=float(np.mean([r['speed_px_s'] for r in rows])),
        mean_accel=float(np.mean([r['accel_px_s2'] for r in rows])),
        mean_contribution_from_v=float(np.mean([r['displacement_px'] for r in rows])),
        mean_contribution_from_a=float(np.mean([r['accel_contribution_px'] for r in rows])),
        note='cv = p+v*0.4 ; ca = p+v*0.4+0.5*a*0.16 ; both fixed forms, nothing fitted')
    print(json.dumps(summary,indent=1,ensure_ascii=False))
    print()
    for row in rows:
        print(f"t={row['t']:.1f} CV {row['error_cv']:6.1f}  CA {row['error_ca']:6.1f}  "
              f"|v| {row['speed_px_s']:6.0f}  |a| {row['accel_px_s2']:6.0f}  "
              f"v-term {row['displacement_px']:6.0f}px  a-term {row['accel_contribution_px']:5.1f}px")
    (BASE/'error_decomposition.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2),'utf-8')


if __name__=='__main__':
    main()
