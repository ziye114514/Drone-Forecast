"""Inspect the existing shared frequency gate before adding another axis observer."""
from pathlib import Path
import numpy as np
import forecast61 as current

def run():
    current.verify();out=current.current.OUT/'feature_gate_audit61.json';assert not out.exists()
    code=current.ROOT/'research/stage27/physics27.py';source=code.read_text(encoding='utf-8')
    assert "quality=float(sensing['frequency_quality']*obs.get('geometry_quality',0))" in source
    assert "model.update(t,state,direction,ratio,quality,bool(obs.get('detected')),bool(sensing['four_visible'] and active))" in source
    assert "ratio=1. if name=='adaptive_axis_only' else self.last_ratio" in source
    table=[];inputs={}
    for case in current.SCORING:
        if current.CASES[case]['reference']=='negative5':continue
        path=current.OUT/case/'observations_forecasts.jsonl';inputs[case]=current.sha(path);rr=current.rows(path);valid=[r for r in rr if r['frontends']['delayed_feedback']['valid']]
        fresh=[]
        for r in valid:
            d=r['frontends']['delayed_feedback']['feature_diagnostic'];m=d['models']['adaptive_axis_only']
            if m['normal'] is not None and d['axis_age_s'] is not None and d['axis_age_s']<=.16:fresh.append(d)
        table.append(dict(case=case,valid_parent_ticks=len(valid),fresh_axis_state_ticks=len(fresh),
            fresh_axis_but_frequency_absent_or_expired=sum(d['frequency_age_s'] is None or d['frequency_age_s']>.16 for d in fresh),
            fresh_axis_without_four_visible=sum(not d['four_visible'] for d in fresh),
            shared_gate_active_ticks=sum(d['models']['adaptive_axis_only']['active'] for d in fresh),
            max_axis_only_RLS_updates=max([r['frontends']['delayed_feedback']['feature_diagnostic']['models']['adaptive_axis_only']['updates'] for r in rr]+[0])))
    current.save(out,dict(helper_sha256=current.sha(Path(__file__)),input_hashes=inputs,table=table,
        source_hashes={name:current.sha(current.ROOT/name) for name in ['research/stage27/physics27.py','research/stage28/axis28.py','research/stage51/reference51.py','research/stage52/feedback52.py']},
        conclusion='Original27 intentionally used a shared feature gate for controlled ablations: even adaptive_axis_only ratio=1 still needs fresh four-frequency force timestamp and frequency_quality*geometry_quality, and learning queue launch requires four_visible. Current integrated52 retains it. Fresh stored axis state alone is not validated pose quality; do not simply remove all gates.',
        next='Before adding6 whole-body observer, isolate axis-only prediction/learning gates with geometry-only axis quality/current support while leaving frequency models exactly unchanged. Existing51 already preserves axis state/model across rotor-epoch changes; do not repeat that fix. Retain0.6s warmup,52 delayed feedback and equal-fallback/pure-history controls; no fabricated RPM or lower four-point layout thresholds.'))
    print(table,flush=True)

if __name__=='__main__':run()
