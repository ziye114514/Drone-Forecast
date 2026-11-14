"""Save the side-forward diagnostic without claiming RPM or forecast gain."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from line_phase_moments import measure_xy


if __name__=='__main__':
    out=ROOT.parent/'output/line370_20261109';out.mkdir(exist_ok=False)
    source=json.loads((ROOT.parent/'output/line369_20261109/results.json').read_text('utf8'))
    cases=source['cases'];controls=source['controls']
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    order=['rotating_line','opposite_phase_pair','translated_static_line','periodic_illumination','colored_nonperiodic_extent']
    for i,name in enumerate(order):
        parent=[r['late_R2'] for r in controls if r['case']==name and r['qualified']]
        local=[c['late_R2'] for r in controls if r['case']==name for c in r.get('spatial_cells',{}).values() if c['qualified']]
        axes[0,0].scatter(np.full(len(parent),i)-.1,parent,color='#677b92',s=16,label='pooled' if i==0 else None)
        axes[0,0].scatter(np.full(len(local),i)+.1,local,color='#008f7a',s=16,label='spatial cells' if i==0 else None)
    axes[0,0].set(xticks=range(5),xticklabels=['rotor','phase pair','static line','light flicker','colored noise'],ylabel='late partial R²',title='Controls:8 seeds per case; every score shown');axes[0,0].legend(fontsize=8)
    summaries={}
    for ax,(n,rr) in zip([axes[0,1],axes[1,0],axes[1,1]],cases.items()):
        parent=[(r['t'],r['late_R2']) for r in rr if r['qualified']]
        local=[(r['t'],c['late_R2']) for r in rr for c in r.get('spatial_cells',{}).values() if c['qualified']]
        if parent:ax.scatter(*np.array(parent).T,color='#677b92',marker='o',label='pooled')
        if local:ax.scatter(*np.array(local).T,color='#008f7a',marker='x',label='fixed early spatial cells')
        ax.set(title=f'Recording{n}: all16 fixed source queries',xlabel='source cutoff(s)',ylabel='late partial R²');ax.legend(fontsize=8)
        summaries[n]=dict(requested_frames=len(rr),global_geometry_eligible=len(parent),local_geometry_eligible=len(local),
            pooled_late_R2_max=max((v for t,v in parent),default=None),local_late_R2_max=max((v for t,v in local),default=None),
            rotation_detection_confirmed=False,physical_RPM_measured=False,forecast_input_enabled=False)
    for ax in axes.flat:ax.axhline(0,color='#444',ls='--',lw=1);ax.grid(alpha=.2)
    fig.savefig(out/'side_phase_diagnostic.png',dpi=150);plt.close(fig)
    # Mutating ONLY late spatial observations cannot alter the fitted clock.
    rng=np.random.default_rng(370);t=np.sort(rng.uniform(0,.033333,6000));r=rng.uniform(-1,1,len(t));xy=np.c_[60*r*np.cos(2*np.pi*250*t+.4),rng.normal(0,.8,len(t))];p=rng.integers(0,2,len(t))
    a,_=measure_xy(xy,t,p);changed=xy.copy();ids=np.flatnonzero(t>=.0166665);changed[ids]=changed[rng.permutation(ids)];b,_=measure_xy(changed,t,p)
    assert a['early_center_xy']==b['early_center_xy'];assert a['early_axis_xy']==b['early_axis_xy'];assert abs(a['frequency_proxy_hz']-b['frequency_proxy_hz'])<1e-10
    control_summary={}
    for name in order:
        aa=[r['late_R2'] for r in controls if r['case']==name and r['qualified']];bb=[c['late_R2'] for r in controls if r['case']==name for c in r.get('spatial_cells',{}).values() if c['qualified']]
        control_summary[name]=dict(global_median=float(np.median(aa)),local_median=float(np.median(bb)),local_max=float(max(bb)))
    result=dict(cases=summaries,controls=control_summary,late_spatial_suffix_changes_clock=False,
        frozen_clock_Hz=[a['frequency_proxy_hz'],b['frequency_proxy_hz']],
        independent_sensor_false_alarm_calibrated=False,first366_validation_confounded_by_trend=True,
        no_NEW_FDE_or_FPS_result=True,no_predictor_integration=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf8');print(result)
