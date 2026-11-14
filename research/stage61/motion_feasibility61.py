"""Past-CV velocity vs native rotor size: feasibility, not measured blur/velocity truth."""
from pathlib import Path
import json,sys
import numpy as np
import forecast61 as current

def run():
    current.verify();out=current.current.OUT/'motion_feasibility61.json';assert not out.exists();summary=[];inputs={}
    for ref in ['clear6','side6','train3']:
        folder=current.OUT/f'off_{ref}';path=folder/'observations_forecasts.jsonl';tickpath=folder/'rotor_ticks.jsonl';inputs[ref]={p.name:current.sha(p) for p in [path,tickpath]}
        parents=current.rows(path);ticks={round(r['t'],8):r for r in current.rows(tickpath)};records=[]
        for i in range(1,len(parents)):
            previous=parents[i-1];q=previous['frontends']['delayed_feedback']
            if not q['valid']:continue
            v=np.asarray(q['coefficients']['cv']['velocity'],float)
            assert round(previous['t']*1e6)<=round((parents[i]['t']-.02)*1e6)
            for t in [parents[i]['t']-.01,parents[i]['t']]:
                r=ticks[round(t,8)];rotors=r['observation']['rotors'];d=float(np.linalg.norm(v)*.01)
                if not rotors:continue
                normal_ratios=[];major_ratios=[]
                for rotor in rotors:
                    a=np.deg2rad(rotor['angle']);u=np.array([np.cos(a),np.sin(a)]);n=np.array([-u[1],u[0]])
                    normal_ratios.append(abs(float(v@n))*.01/max(rotor['minor'],1.))
                    major_ratios.append(abs(float(v@u))*.01/max(rotor['major'],1.))
                records.append(dict(t=t,past_velocity_t=previous['t'],estimated_full10ms_translation_px=d,
                    median_normal_translation_over_observed_minor=float(np.median(normal_ratios)),median_axis_translation_over_observed_major=float(np.median(major_ratios))))
        summary.append(dict(case=ref,current_ticks_with_rotors_and_valid_past_CV=len(records),
            median_estimated10ms_translation_px=float(np.median([r['estimated_full10ms_translation_px'] for r in records])),
            p90_estimated10ms_translation_px=float(np.quantile([r['estimated_full10ms_translation_px'] for r in records],.9)),
            median_normal_translation_over_minor=float(np.median([r['median_normal_translation_over_observed_minor'] for r in records])),
            median_axis_translation_over_major=float(np.median([r['median_axis_translation_over_observed_major'] for r in records]))))
        current.save(current.current.OUT/f'motion_feasibility_{ref}.json',dict(records=records))
    current.save(out,dict(helper_sha256=current.sha(Path(__file__)),input_hashes=inputs,table=summary,
        method='Only preceding20ms-frame saved CV velocity, known by next packet start; existing current rotor geometry for scale comparison. Uniform-motion full10ms span estimate, not measured motion-blur length, velocity truth, correction efficacy or forecast gain. No future target labels/analysis input.',
        reuse='Existing21 windows21.warp already implements translation compensation, previouslywarp20 used rotor-detector velocity and had no uniform win. A future controlled test may reuse it with causal NN-body-derived CV velocity at10ms; do not claim compensation itself novel.'))
    print(summary,flush=True)

if __name__=='__main__':run()
