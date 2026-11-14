"""Is the declared 2px displacement noise honest?

The observer trusts the per-frame rotor-region displacement with a fixed 2px
variance. This compares each saved displacement with the position change over the
same interval, i.e. the disagreement between two independent estimates of the
SAME body motion. No future information and nothing is fitted: only the residual
distribution is reported, plus what a corrected noise prior would imply for the
0.4s velocity uncertainty.
"""
import json,pathlib
import numpy as np
BASE=pathlib.Path(r"D:\drone forecast\output\side_temporal164_20261101\raw_side_final")


def main():
    rows=[json.loads(x) for x in (BASE/'predictions.jsonl').read_text('utf-8').splitlines()]
    samples=[]
    for previous,current in zip(rows,rows[1:]):
        estimate=(current.get('motion') or {}).get('estimate') or {}
        if not estimate or estimate.get('score',0)<=0:continue
        dt=current['t']-previous['t']
        if not 0<dt<=.12:continue
        measured=np.array([estimate['dx'],estimate['dy']],float)
        body_previous=(previous.get('body') or {}).get('center')
        body_current=(current.get('body') or {}).get('center')
        if body_previous is None or body_current is None:continue
        observed=np.asarray(body_current,float)-np.asarray(body_previous,float)
        residual=measured-observed
        samples.append(dict(t=current['t'],dt=dt,measured=measured.tolist(),
            observed=observed.tolist(),residual=residual.tolist(),
            score=float(estimate.get('score',0)),
            speed=float(np.linalg.norm(observed)/dt)))
    residuals=np.array([s['residual'] for s in samples])
    speeds=np.array([s['speed'] for s in samples])
    dts=np.array([s['dt'] for s in samples])
    radial=np.linalg.norm(residuals,axis=1)
    summary=dict(pairs=len(samples),
        residual_px=dict(mean_abs=float(np.mean(np.abs(residuals))),
            rms=float(np.sqrt(np.mean(residuals**2))),
            radial_median=float(np.median(radial)),
            radial_p95=float(np.percentile(radial,95)),
            per_axis_sigma=float(np.sqrt(np.mean(residuals**2,axis=0)).mean())),
        speed_px_s=dict(median=float(np.median(speeds)),p95=float(np.percentile(speeds,95))),
        dt_s=dict(median=float(np.median(dts))),
        implied_velocity_sigma_px_s=None,
        declared_velocity_prior_from_2px=None,
        note='the residual is displacement-estimate minus position-difference for the same interval; '
             'both are causal, neither uses truth labels')
    sigma=summary['residual_px']['per_axis_sigma']
    dt=float(np.median(dts))
    summary['implied_velocity_sigma_px_s']=float(sigma/dt)
    summary['declared_velocity_prior_from_2px']=float(2.0/dt)
    summary['ratio_implied_over_declared']=float((sigma/2.0))
    print(json.dumps(summary,indent=1,ensure_ascii=False))
    for sample in samples[:6]:
        print(f"  t={sample['t']:.3f} dt={sample['dt']*1000:5.1f}ms measured=({sample['measured'][0]:6.2f},"
              f"{sample['measured'][1]:6.2f}) observed=({sample['observed'][0]:6.2f},{sample['observed'][1]:6.2f}) "
              f"residual={np.linalg.norm(sample['residual']):.2f}px")
    (BASE/'displacement_noise.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),'utf-8')


if __name__=='__main__':
    main()
