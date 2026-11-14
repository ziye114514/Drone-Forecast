"""Audit saved75 errors; no detector or physical parameter changes."""
from pathlib import Path
import datetime,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage75'),str(ROOT/'research/stage74')]
import study75 as raw
import study74 as base
OUT=ROOT.parent/'output/stage76_fusion_failure_audit_20261021'
load=base.load;save=base.save;sha=base.sha;rows=base.rows

def run():
    raw.verify();base.verify();assert not OUT.exists()
    inputs={c:sha(raw.OUT/c/'observations_forecasts.jsonl') for c in base.CASES}
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hash=sha(__file__),
        input_hashes=inputs,analysis_hash=sha(base.OUT/'analysis.json'),
        scope='Post-hoc failure decomposition using saved forecasts and already seen '
              'sparse scores. Counterfactual components are diagnostics, not a new '
              'online predictor or held-out evaluation. No RAW re-read.'))
    analysis=load(base.OUT/'analysis.json');details=[];summaries=[]
    for case in base.CASES:
        rr=rows(raw.OUT/case/'observations_forecasts.jsonl');by={round(r['t'],8):(i,r) for i,r in enumerate(rr)}
        lookup={(d['target_t'],d['nominal_h'],d['model']):d for d in analysis['details'] if d['case']==case}
        for d in analysis['details']:
            if d['case']!=case or d['model']!='dual_cv_event_increment' or d['target_state']!='present' or not d['forecast_valid']:continue
            i,r=by[round(d['anchor_t'],8)];q=r['frontends']['delayed_feedback'];cv=q['coefficients']['cv'];c=q['coefficients'][d['model']];md=q['dual_branch_diagnostic'][d['model']]
            h=d['actual_h'];state=np.asarray(c['position'])-cv['position']+(np.asarray(c['velocity'])-cv['velocity'])*h
            force=np.asarray(d['predicted'])-c['position']-np.asarray(c['velocity'])*h
            basepoint=np.asarray(base.prior.evaluate(cv,h));np.testing.assert_allclose(basepoint+state+force,d['predicted'],atol=1e-9,rtol=0)
            previous=rr[i-1]['frontends']['delayed_feedback'] if i else None
            ha=np.asarray(previous['coefficients']['trusted_acc']['acceleration']) if previous and previous['valid'] else np.zeros(2)
            control=np.asarray(md.get('control_acceleration',[0,0]));event=control-ha if md['physical_input_active'] else np.zeros(2)
            gain=previous['increment_diagnostic']['models']['increment_periodic']['gain_px_s2'] if previous else None
            def error(m):return lookup[d['target_t'],d['nominal_h'],m]['error_px']
            target=np.asarray(d['target_center']);inc=q['increment_diagnostic']
            details.append(dict(case=case,anchor_t=d['anchor_t'],target_t=d['target_t'],nominal_h=d['nominal_h'],actual_h=h,
                cv_error=error('cv'),history_error=error('trusted_acc'),fused_error=d['error_px'],
                state_shift=state.tolist(),state_shift_norm=float(np.linalg.norm(state)),
                future_force_shift=force.tolist(),future_force_shift_norm=float(np.linalg.norm(force)),
                old_control=control.tolist(),previous_history_a=ha.tolist(),event_control_increment=event.tolist(),previous_event_gain=gain,
                physical_input_active=md['physical_input_active'],mode_weight=md['mode_probability'][1],
                axis_current=inc['direction']['current'],axis_age_s=inc['direction']['age_s'],
                periodic_current=inc['features']['periodic']['current'],periodic_age_s=inc['features']['periodic']['age_s'],
                no_state_shift_error=float(np.linalg.norm(basepoint+force-target)),
                no_future_force_error=float(np.linalg.norm(basepoint+state-target)),
                history_fallback_error=d['error_px'] if md['physical_input_active'] else error('trusted_acc')))
        dd=[d for d in details if d['case']==case and d['nominal_h']==.4]
        if dd:
            summaries.append(dict(case=case,n=len(dd),cv_error=float(np.mean([d['cv_error'] for d in dd])),
                history_error=float(np.mean([d['history_error'] for d in dd])),fused_error=float(np.mean([d['fused_error'] for d in dd])),
                inactive_targets=sum(not d['physical_input_active'] for d in dd),
                inactive_effectively_cv=sum(not d['physical_input_active'] and abs(d['cv_error']-d['fused_error'])<1e-5 for d in dd),
                active_but_stale_targets=sum(d['physical_input_active'] and not(d['axis_current'] and d['periodic_current']) for d in dd),
                mean_state_shift=float(np.mean([d['state_shift_norm'] for d in dd])),
                mean_future_force_shift=float(np.mean([d['future_force_shift_norm'] for d in dd])),
                history_fallback_error=float(np.mean([d['history_fallback_error'] for d in dd]))))
    save(OUT/'audit76.json',dict(table=summaries,details=details,component_reconstruction_exact=True,
        hypotheses=['Unavailable physical input falls to CV and loses strong history.',
            '74 uses previous_history_a for both interval state propagation and future rollout.',
            'Forecast residual gains are not validated instantaneous accelerations.',
            'IMM shares states, so erroneous controls can affect the otherwise neutral mode.'],
        controls_needed=['Refresh only future history input to current t.',
            'Matched fallback to history when physics is missing.',
            'Require current axis and periodic input, not just their .16s coast.',
            'Compare author independent MMAE versus interacting IMM with the same inputs.',
            'Compare state-control versus forecast-only residual use; keep constant inputs.']))
    print(summaries,flush=True)
if __name__=='__main__':run()
