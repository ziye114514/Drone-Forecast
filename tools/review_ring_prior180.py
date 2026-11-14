"""One fixed four-visible native review, automatic ring-phase qualification."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from observed_ring_phase import ring_phase_prior
from pilot_author_phase180 import transformed
from pure_visibility170 import window

if __name__=='__main__':
    z=json.loads((ROOT.parent/'output/rim174_20261107/supported_rim_recovery.json').read_text('utf-8'))
    rotors=z['review_C']['rotors'];t=112.02
    rows=[json.loads(s) for s in (ROOT.parent/'output/layout176_20261107/integrated/side_fixed.jsonl').read_text('utf-8').splitlines()]
    old=max((r for r in rows if r['t']<=t-.033333),key=lambda r:r['t']);m=old['motion'];s=m['estimate'];v=np.array([s['dx'],s['dy']])/m['dt_s']
    events=window(ROOT.parent/'output/side_temporal164_20261101/cache_side_warm',t);out=[]
    for r in rotors:
        e=transformed(events,r,np.zeros(2),v,t);q=ring_phase_prior(e,r,t)
        out.append(dict(ellipse=r,phase=q));print(round(r['cx'],1),q['qualified'],q.get('candidate_signed_phase_hz'),q.get('validation_concentration'),q.get('phase_agreement'))
    path=ROOT.parent/'output/helix_bridge180_20261107/review_C_ring_phases.json'
    path.write_text(json.dumps(dict(t=t,previous_velocity_source_t=old['t'],rotors=out,
        qualification_thresholds_unchanged=True,labels_used_for_fitting=False),indent=2),'utf-8')
